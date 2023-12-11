import cv2
import numpy as np

BP2KINECT = np.asarray([
    [1, 18, 19],  # left hip
    [2, 22, 23],  # right hip
    # [3, 0, 1],  # spine
    [4, 19, 20],  # left knee
    [5, 23, 24],  # right knee
    [16, 5, 6],  # left shoulder
    [17, 12, 13],  # right shoulder
    [18, 6, 7],  # left elbow
    [19, 13, 14],  # right elbow,
    [12, 3, 26],  # neck
    [15, 26, 27]  # head
])


def get_smplx_init(kinect_joints, kinect_confs, betas, kintree_table, T, s2k,
                   J):

    parents = kintree_table[0].astype(np.int32)
    parents[0] = -1

    kinect_confs = kinect_confs.reshape((32, 1))
    kinect_confs = np.repeat(kinect_confs, 3, axis=1)

    joints_kinect_d = kinect_joints
    joints_viz_f = kinect_confs

    joints_kinect_m = joints_kinect_d @ T[0:3, 0:3].T + T[0:3, 3].reshape(1, 3)

    dtype = np.float32
    v_kin_flat = s2k @ np.concatenate([betas.reshape(10), np.ones(1)])
    v_kin = v_kin_flat.reshape(-1, 3).astype(dtype)

    rots, trans = initialize_pose_advanced(joints_kinect_m, joints_viz_f[:, 0],
                                           v_kin, J, parents, dtype)
    if rots is None:
        print('Perform pose init failed, taking default values')
        rots = np.zeros((len(parents), 3), dtype=dtype)
        trans = np.zeros(3, dtype=dtype)

    rot = rots[0]
    pose_body = rots[1:22].reshape(-1)

    return pose_body, rot, trans


def initialize_pose_advanced(joints_kinect, joints_viz, kinect_verts, J,
                             parents, dtype):
    # bpid, vert_start, vert_end
    bp2kinect = BP2KINECT

    kinect_ids = np.asarray([0, 3, 5, 12], dtype=int)

    if np.sum(joints_viz[kinect_ids]) < 4:
        return None, None

    j_k = joints_kinect[kinect_ids]
    j_s = kinect_verts[kinect_ids]
    R_smpl = align_canonical(j_s[0] - j_s[1], j_s[3] - j_s[2])
    R_kin = align_canonical(j_k[0] - j_k[1], j_k[3] - j_k[2])

    R_glob = R_kin.T @ R_smpl

    rot, jac = cv2.Rodrigues(R_glob)
    t_glob = -R_glob @ np.mean(j_s, axis=0) + np.mean(j_k, axis=0)

    dtype = dtype
    n_j = len(parents)
    rots = np.zeros((n_j, 3), dtype=dtype)
    rots_assd = np.zeros(n_j, dtype=np.bool)

    rots[0] = rot.reshape(-1)
    t_glob = t_glob + R_glob @ J[0][0] - J[0][0]

    # return rots, t_glob

    for i in range(0, bp2kinect.shape[0]):
        if joints_viz[bp2kinect[i, 1]] + joints_viz[bp2kinect[i, 2]] < 2:
            continue
        model_dir = kinect_verts[bp2kinect[i, 2]] - kinect_verts[bp2kinect[i,
                                                                           1]]
        model_dir /= np.linalg.norm(model_dir)
        skel_dir = joints_kinect[bp2kinect[i, 2]] - joints_kinect[bp2kinect[i,
                                                                            1]]
        skel_dir /= np.linalg.norm(skel_dir)
        par_i = parents[bp2kinect[i, 0]]
        rot = np.eye(3)
        while par_i != -1:
            rot_par = rots[par_i]
            rot_par_mat, jac = cv2.Rodrigues(rot_par)
            rot = rot_par_mat @ rot
            par_i = parents[par_i]
        model_dir_in_joint = model_dir
        skel_dir_in_joint = rot.T @ skel_dir
        aa = rotate_a_b_axis_angle(skel_dir_in_joint, model_dir_in_joint)
        rots[bp2kinect[i, 0]] = -aa
        rots_assd[bp2kinect[i, 0]] = True

    return rots, t_glob


def align_canonical(x_dir, y_dir):
    x_dir /= np.linalg.norm(x_dir)
    z_dir = np.cross(x_dir, y_dir)
    z_dir /= np.linalg.norm(z_dir)
    y_dir = np.cross(z_dir, x_dir)
    y_dir /= np.linalg.norm(y_dir)
    R = np.stack([x_dir, y_dir, z_dir], axis=0)
    return R


def rotate_a_b_axis_angle(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    rot_axis = np.cross(a, b)
    a_proj = b * (a.dot(b))
    a_ort = a - a_proj
    theta = np.arctan2(np.linalg.norm(a_ort), np.linalg.norm(a_proj))
    if a.dot(b) < 0:
        theta = np.pi - theta
    aa = rot_axis / np.linalg.norm(rot_axis) * theta
    return aa


def orthoprocrustes(xs, ys):
    n = xs.shape[0]
    xsm = np.mean(xs, 0)
    ysm = np.mean(ys, 0)
    q = xs.shape[1]
    dxs = xs - np.tile(xsm.reshape(1, -1), (n, 1))
    dys = ys - np.tile(ysm.reshape(1, -1), (n, 1))
    dxst = np.tile(dxs.reshape(-1, 1, q), (1, q, 1))
    dyst = np.tile(dys.reshape(-1, q, 1), (1, 1, q))
    S = np.sum(dyst * dxst, 0)
    u, s, v = np.linalg.svd(S)
    s = np.sign(s)
    s = np.diag(s)
    R = u @ s @ v
    cnt = 0
    while (np.linalg.det(R) < 0):
        s[2 - cnt] *= -1
        R = u @ s @ v

    t = ysm - R @ xsm
    return R, t
