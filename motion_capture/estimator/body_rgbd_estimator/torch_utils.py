# yapf: disable
import math
import torch

try:
    from realtime_mocap.extern.smplx_kinect.smplx_kinect.common.angle_representation import (  # noqa: E501
        universe_convert,
    )
    has_smplx_kinect = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_smplx_kinect = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception
# yapf: enable


def get_smplx_init(kinect_joints, kinect_confs, betas, kintree_table, T, s2k,
                   J):
    parents = kintree_table[0]
    parents[0] = -1

    dtype = torch.float32
    kinect_confs = kinect_confs.repeat(1, 3)
    joints_viz_f = kinect_confs

    joints_kinect_m = kinect_joints @ T[0:3, 0:3].T + T[0:3, 3].reshape(1, 3)

    v_kin_flat = s2k @ torch.cat([
        betas.reshape(10),
        torch.ones(1, dtype=betas.dtype, device=betas.device)
    ])
    v_kin = v_kin_flat.reshape(-1, 3)

    rots, trans = initialize_pose_advanced(joints_kinect_m, joints_viz_f[:, 0],
                                           v_kin, J, parents, dtype)
    if rots is None:
        print('Perform pose init failed, taking default values')
        rots = torch.zeros((len(parents), 3),
                           dtype=dtype,
                           device=kinect_joints.device)
        trans = torch.zeros(3, dtype=dtype, device=kinect_joints.device)

    rot = rots[0]
    pose_body = rots[1:22].reshape(-1)

    return pose_body, rot, trans


def initialize_pose_advanced(joints_kinect, joints_viz, kinect_verts, J,
                             parents, dtype):
    #  bpid, vert_start, vert_end
    bp2kinect = torch.tensor([
        [1, 18, 19],  # left hip
        [2, 22, 23],  # right hip
        # [3, 0, 1],#spine
        [4, 19, 20],  # left knee
        [5, 23, 24],  # right knee
        [16, 5, 6],  # left shoulder
        [17, 12, 13],  # right shoulder
        [18, 6, 7],  # left elbow
        [19, 13, 14],  # right elbow,
        [12, 3, 26],  # neck
        [15, 26, 27]  # head
    ])

    #  we take 4 points: pelvis, chest, left shoulder, right shoulder
    #  using these points, we align absolute rotation
    kinect_ids = torch.tensor([0, 3, 5, 12])  # 4, 11, 18, 22

    if torch.sum(joints_viz[kinect_ids]) < 4:
        return None, None

    j_s = kinect_verts[kinect_ids]
    j_k = joints_kinect[kinect_ids].to(dtype=j_s.dtype)

    R_smpl = align_canonical(j_s[0] - j_s[1], j_s[3] - j_s[2])
    R_kin = align_canonical(j_k[0] - j_k[1], j_k[3] - j_k[2])
    R_glob = R_kin.T @ R_smpl

    t_glob = -R_glob @ torch.mean(j_s, axis=0) + torch.mean(j_k, axis=0)
    n_j = len(parents)
    rots = torch.zeros((n_j, 3), dtype=dtype)
    rots_assd = torch.zeros(n_j, dtype=torch.bool)

    rots[0] = universe_convert(R_glob, 'rotmtx', 'aa')
    t_glob = t_glob + R_glob @ J[0][0] - J[0][0]

    # return rots, t_glob

    for i in range(0, bp2kinect.shape[0]):
        if joints_viz[bp2kinect[i, 1]] + joints_viz[bp2kinect[i, 2]] < 2:
            continue
        model_dir = kinect_verts[bp2kinect[i, 2]] - kinect_verts[bp2kinect[i,
                                                                           1]]
        model_dir /= torch.linalg.norm(model_dir)
        skel_dir = joints_kinect[bp2kinect[i, 2]] - joints_kinect[bp2kinect[i,
                                                                            1]]
        skel_dir /= torch.linalg.norm(skel_dir)
        par_i = parents[bp2kinect[i, 0]]
        rot = torch.eye(3, dtype=skel_dir.dtype, device=skel_dir.device)
        while par_i != -1:
            rot_par = rots[par_i]
            rot_par_mat = universe_convert(rot_par, 'aa', 'rotmtx').to(
                dtype=skel_dir.dtype, device=skel_dir.device)
            rot = rot_par_mat @ rot
            par_i = parents[par_i]
        model_dir_in_joint = model_dir
        skel_dir_in_joint = rot.T @ skel_dir
        aa = rotate_a_b_axis_angle(
            skel_dir_in_joint.to(dtype=model_dir.dtype), model_dir_in_joint)
        rots[bp2kinect[i, 0]] = -aa
        rots_assd[bp2kinect[i, 0]] = True

    return rots, t_glob


def rotate_a_b_axis_angle(a, b):
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    rot_axis = torch.cross(a, b)
    # find a proj onto b
    a_proj = b * (a.dot(b))
    a_ort = a - a_proj
    theta = torch.atan2(torch.linalg.norm(a_ort), torch.linalg.norm(a_proj))
    if a.dot(b) < 0:
        theta = math.pi - theta
    aa = rot_axis / torch.linalg.norm(rot_axis) * theta
    return aa


def align_canonical(x_dir, y_dir):
    x_dir /= torch.linalg.norm(x_dir)
    z_dir = torch.cross(x_dir, y_dir)
    z_dir /= torch.linalg.norm(z_dir)
    y_dir = torch.cross(z_dir, x_dir)
    y_dir /= torch.linalg.norm(y_dir)
    R = torch.stack([x_dir, y_dir, z_dir], dim=0)
    return R


def orthoprocrustes(xs, ys):
    n = xs.shape[0]
    xsm = torch.mean(xs, 0)
    ysm = torch.mean(ys, 0)
    q = xs.shape[1]
    dxs = xs - torch.tile(xsm.reshape(1, -1), (n, 1))
    dys = ys - torch.tile(ysm.reshape(1, -1), (n, 1))
    dxst = torch.tile(dxs.reshape(-1, 1, q), (1, q, 1))
    dyst = torch.tile(dys.reshape(-1, q, 1), (1, 1, q))
    S = torch.sum(dyst * dxst, 0)
    u, s, v = torch.linalg.svd(S)

    s = torch.sign(s)
    s = torch.diag(s)
    R = u @ s @ v
    cnt = 0
    while (torch.linalg.det(R) < 0):
        s[2 - cnt] *= -1
        R = u @ s @ v

    t = ysm - R @ xsm
    return R, t
