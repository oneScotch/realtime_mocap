# Copyright (c) Hao Meng. All Rights Reserved.
import numpy as np
import transforms3d

from .config import ID2ROT, SNAP_PARENT, kinematic_tree


def compute_root_rotation_svd_np(T, P, idxs):
    # Compute R0, here we think R0 is not only a Orthogonal matrix, but also a Rotation matrix. # noqa: E501
    # You can refer to paper "Least-Squares Fitting of Two 3-D Point Sets. K. S. Arun; T. S. Huang; S. D. Blostein" # noqa: E501
    # It is slightly different from  https://github.com/Jeff-sjtu/HybrIK/blob/main/hybrik/utils/pose_utils.py#L4, # noqa: E501
    # in which R0 is regard as orthogonal matrix only. Using their method might further boost accuracy. # noqa: E501

    P_0 = (P[idxs] - P[0]).squeeze().T  # (3, 5)
    T_0 = (T[idxs] - T[0]).squeeze().T  # (3, 5)

    H = np.matmul(T_0, P_0.T)  # (3, 5) x (5, 3) -> (3, 3)

    U, S, V_T = np.linalg.svd(H)  # (3, 3), (3,), (3, 3)
    V = V_T.T
    R0 = np.matmul(V, U.T)

    det0 = np.linalg.det(R0)

    if abs(det0 + 1) < 1e-6:
        V_ = V.copy()

        if (abs(S) < 1e-4).sum():
            V_[:, 2] = -V_[:, 2]
            R0 = np.matmul(V_, U.T)
    return R0


def adaptive_IK_np(T_, P_, th_parallel=1e-5, eps=1e-8):
    """Computes MANO hand pose parameters given template and predictions. We
    think the twist of hand bone could be omitted.

    :param T: template ,21*3
    :param P: target, 21*3
    :return: pose params.
    """

    idxs = [1, 5, 9, 13, 17]

    T = T_.copy()[..., None]  # (21, 3, 1)
    P = P_.copy()[..., None]  # (21, 3, 1)

    # some globals
    R = {}
    R_pa_k = {}
    q = {}
    q[0] = T[0]  # in fact, q[0] = P[0] = T[0].

    # compute root rotation
    R0 = compute_root_rotation_svd_np(T=T, P=P, idxs=idxs)

    # the bone from 1, 5, 9, 13, 17 to 0 has same rotations
    R = {idx: R0.copy() for idx in idxs}
    R[0] = R0

    # compute rotation along kinematics
    for k in kinematic_tree:
        pa = SNAP_PARENT[k]
        pa_pa = SNAP_PARENT[pa]
        q[pa] = np.matmul(R[pa], (T[pa] - T[pa_pa])) + q[pa_pa]
        delta_p_k = np.matmul(np.linalg.inv(R[pa]), P[k] - q[pa])
        delta_p_k = delta_p_k.reshape((3, ))
        delta_t_k = T[k] - T[pa]
        delta_t_k = delta_t_k.reshape((3, ))
        temp_axis = np.cross(delta_t_k, delta_p_k)
        temp_axis_norm = np.linalg.norm(temp_axis, axis=-1) + eps
        if temp_axis_norm >= th_parallel:
            # delta_t_k and delta_p_k is not parallel,
            # then computing the axis angle between them.
            # As the angle for axis is zero,
            # we directly assign D_twist to identity matrix.
            axis = temp_axis / temp_axis_norm
            temp = np.linalg.norm(delta_t_k, axis=0) * \
                np.linalg.norm(delta_p_k, axis=0) + eps
            cos_alpha = np.dot(delta_t_k, delta_p_k) / temp
            alpha = np.arccos(cos_alpha)

            # TODO: replace transforms3d for quicker aa_to_rotmat impl
            D_sw = transforms3d.axangles.axangle2mat(
                axis=axis, angle=alpha, is_normalized=False)
            D_tw = np.eye(3)
            R_pa_k[k] = np.matmul(D_sw, D_tw)
            R[k] = np.matmul(R[pa], R_pa_k[k])
        else:
            # delta_t_k and delta_p_k is parallel and thus alpha is zero
            R_pa_k[k] = np.eye(3)
            R[k] = R[pa]

    pose_R = np.zeros((1, 16, 3, 3))
    pose_R[0, 0] = R[0]
    for kp_idx, rot_idx in ID2ROT.items():
        pose_R[0, rot_idx] = R_pa_k[kp_idx]

    return pose_R
