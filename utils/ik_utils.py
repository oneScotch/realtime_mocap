import numpy as np
import torch


def calc_bone_len(kp3d, index=9):
    if isinstance(kp3d, np.ndarray):
        return np.linalg.norm(kp3d[index] - kp3d[0])
    elif isinstance(kp3d, torch.Tensor):
        return torch.linalg.norm(kp3d[index] - kp3d[0])
    else:
        raise TypeError(f'Unsupported type: {type(kp3d)}')


def normalize_with_target(kp3d, kp3d_tgt):
    ratio = calc_bone_len(kp3d_tgt) / calc_bone_len(kp3d)
    kp3d_norm = kp3d * ratio
    kp3d_norm = kp3d_norm - kp3d_norm[0] + kp3d_tgt[0]
    return kp3d_norm


def align_bone_len(opt_, pre_):
    opt = opt_.copy()
    pre = pre_.copy()

    opt_align = opt.copy()
    for i in range(opt.shape[0]):
        ratio = pre[i][6] / opt[i][6]
        opt_align[i] = ratio * opt_align[i]

    err = np.abs(opt_align - pre).mean(0)

    return err


def rotation_local2global(rot_mats, parents, left_mult=False):
    out = np.zeros_like(rot_mats)
    n_rotation = rot_mats.shape[-3]
    for j in range(n_rotation):
        if parents[j] < 0:
            # root rotation
            out[..., j, :, :] = rot_mats[..., j, :, :]
        else:
            parent_rot = out[..., parents[j], :, :]
            local_rot = rot_mats[..., j, :, :]
            lm = local_rot if left_mult else parent_rot
            rm = parent_rot if left_mult else local_rot
            out[..., j, :, :] = np.matmul(lm, rm)
    return out


def rotation_local2global_torch(rot_mats, parents, left_mult=False):
    n_rotation = rot_mats.shape[-3]
    ret_list = []
    for j in range(n_rotation):
        if parents[j] < 0:
            # root rotation
            joint_rotmat = rot_mats[..., j, :, :]
        else:
            parent_rot = ret_list[parents[j]].squeeze(1)
            local_rot = rot_mats[..., j, :, :]
            lm = local_rot if left_mult else parent_rot
            rm = parent_rot if left_mult else local_rot
            joint_rotmat = torch.matmul(lm, rm)
        ret_list.append(joint_rotmat.unsqueeze(1))
    out = torch.cat(ret_list, dim=1)
    return out


def rotation_global2local_sjoint(joint_index, global_rot_mats, parents):
    parent_global_rotmtx = global_rot_mats[parents[joint_index]]
    joint_local_rotmtx = parent_global_rotmtx.T.dot(
        global_rot_mats[joint_index])
    return joint_local_rotmtx


def rotation_global2local(global_rot_mats, parents, left_mult=False):
    parents = np.array(parents)
    n_joints = parents.shape[0]
    orig_shape = global_rot_mats.shape
    global_rot_mats = global_rot_mats.reshape((-1, n_joints, 3, 3))

    out = np.zeros_like(global_rot_mats)
    for j in range(n_joints):
        # for j in prange(n_joints):
        if parents[j] < 0:
            out[:, j, :, :] = global_rot_mats[:, j, :, :]
        else:
            parent_rot = global_rot_mats[:, parents[j], :, :]
            parent_rot = np.transpose(parent_rot, (0, 2, 1))

            local_rot = global_rot_mats[:, j, :, :]

            lm = local_rot if left_mult else parent_rot
            rm = parent_rot if left_mult else local_rot
            out[:, j, :, :] = np.matmul(lm, rm)
    out = out.reshape(*orig_shape)
    return out


def rotation_global2local_torch(global_rot_mats, parents, left_mult=False):
    n_joints = len(parents)
    orig_shape = global_rot_mats.shape
    global_rot_mats = global_rot_mats.reshape((-1, n_joints, 3, 3))

    ret_list = []
    for j in range(n_joints):
        # for j in prange(n_joints):
        if parents[j] < 0:
            joint_rotmat = global_rot_mats[:, j, :, :]
        else:
            parent_rot = global_rot_mats[:, parents[j], :, :]
            parent_rot = torch.transpose(parent_rot, 1, 2)

            local_rot = global_rot_mats[:, j, :, :]

            lm = local_rot if left_mult else parent_rot
            rm = parent_rot if left_mult else local_rot
            joint_rotmat = torch.matmul(lm, rm)
        ret_list.append(joint_rotmat.unsqueeze(1))
    out = torch.cat(ret_list, dim=1).reshape(*orig_shape)
    return out
