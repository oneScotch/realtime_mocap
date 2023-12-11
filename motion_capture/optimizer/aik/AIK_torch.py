# Copyright (c) Zoetrope. All Rights Reserved.
import torch
import transforms3d

from .config import ID2ROT, SNAP_PARENT, kinematic_tree


def compute_root_rotation_svd(T, P, idxs):
    # Compute R0, here we think R0 is not only a Orthogonal matrix, but also a Rotation matrix. # noqa: E501
    # You can refer to paper "Least-Squares Fitting of Two 3-D Point Sets. K. S. Arun; T. S. Huang; S. D. Blostein" # noqa: E501
    # It is slightly different from  https://github.com/Jeff-sjtu/HybrIK/blob/main/hybrik/utils/pose_utils.py#L4, # noqa: E501
    # in which R0 is regard as orthogonal matrix only. Using their method might further boost accuracy. # noqa: E501

    P_0 = (P[idxs] - P[0]).squeeze().T  # (3, 5)
    T_0 = (T[idxs] - T[0]).squeeze().T  # (3, 5)

    H = torch.matmul(T_0, P_0.T)  # (3, 5) x (5, 3) -> (3, 3)

    if H.device != 'cpu':
        H = H.cpu()

    U, S, V_T = torch.linalg.svd(H)  # (3, 3), (3,), (3, 3)

    if H.device != 'cpu':
        U = U.to(T.device)
        V_T = V_T.to(T.device)

    R0 = torch.matmul(U, V_T).T

    det0 = torch.det(R0)

    if abs(det0 + 1) < 1e-6:
        V_ = V_T.T.clone()

        if (abs(S) < 1e-4).sum():
            V_[:, 2] = -V_[:, 2]
            R0 = torch.matmul(V_, U.T)
    return R0


@torch.no_grad()
def adaptive_IK(T_,
                P_,
                th_parallel=1e-5,
                eps=1e-8,
                device='cuda',
                enable_wrist_only=False):
    """Computes MANO hand pose parameters given template and predictions. We
    think the twist of hand bone could be omitted.

    :param T: template ,21*3 (tensor)
    :param P: target, 21*3 (tensor)
    :return: pose params.
    """

    idxs = [1, 5, 9, 13, 17]

    T = T_.clone().detach()[..., None]  # (21, 3, 1)
    P = P_.clone().detach()[..., None]  # (21, 3, 1)

    # some globals
    R = {}
    R_pa_k = {}
    q = {}
    q[0] = T[0]  # in fact, q[0] = P[0] = T[0].

    # compute root rotation
    R0 = compute_root_rotation_svd(T=T, P=P, idxs=idxs)

    # the bone from 1, 5, 9, 13, 17 to 0 has same rotations
    R = {idx: R0.clone() for idx in idxs}
    R[0] = R0

    # rotations
    pose_R = torch.zeros((1, 16, 3, 3)).to(device)
    pose_R[0, 0] = R[0]

    if enable_wrist_only:
        return pose_R

    # compute rotation along kinematics
    cos = torch.nn.CosineSimilarity(dim=0, eps=eps)
    for k in kinematic_tree:
        pa = SNAP_PARENT[k]
        pa_pa = SNAP_PARENT[pa]
        q[pa] = torch.matmul(R[pa], (T[pa] - T[pa_pa])) + q[pa_pa]
        delta_p_k = torch.matmul(torch.linalg.inv(R[pa]), P[k] - q[pa])
        delta_p_k = delta_p_k.reshape((3, ))
        delta_t_k = T[k] - T[pa]
        delta_t_k = delta_t_k.reshape((3, ))
        temp_axis = torch.cross(delta_t_k, delta_p_k)
        temp_axis_norm = torch.linalg.norm(temp_axis, axis=-1) + eps
        if temp_axis_norm >= th_parallel:
            # delta_t_k and delta_p_k is not parallel,
            # then computing the axis angle between them.
            # As the angle for axis is zero,
            # we directly assign D_twist to identity matrix.
            axis = temp_axis / temp_axis_norm
            cos_alpha = cos(delta_t_k, delta_p_k)
            alpha = torch.arccos(cos_alpha)

            # TODO: replace transforms3d for quicker aa_to_rotmat impl
            D_sw = transforms3d.axangles.axangle2mat(
                axis=axis.cpu(), angle=alpha.cpu(), is_normalized=False)

            D_sw = torch.from_numpy(D_sw).to(device)
            # D_tw = torch.eye(3).to(device)
            # R_pa_k[k] = torch.matmul(D_sw, D_tw)
            R_pa_k[k] = D_sw
            R[k] = torch.matmul(R[pa], R_pa_k[k])
        else:
            # delta_t_k and delta_p_k is parallel and thus alpha is zero
            R_pa_k[k] = torch.eye(3).to(device)
            R[k] = R[pa]

    for kp_idx, rot_idx in ID2ROT.items():
        pose_R[0, rot_idx] = R_pa_k[kp_idx]

    return pose_R
