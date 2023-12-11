import numpy as np
import torch
from mmhuman3d.utils.transforms import rotmat_to_aa
from torch.nn import functional as F

DEFAULT_TRANSL = np.array([0.0044228160, 0.63850945, 1.0851063]).reshape(1, 3)


def transform_rotation_frankbody(body_rot6d):
    ret_dict = dict()
    if not isinstance(body_rot6d, torch.Tensor):
        body_rot6d = torch.from_numpy(body_rot6d)
    body_rot6d = body_rot6d.view(-1, 3, 2)
    a1 = body_rot6d[:, :, 0]
    a2 = body_rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    pred_rotmat = torch.stack((b1, b2, b3), dim=-1)
    body_rotaa = rotmat_to_aa(pred_rotmat)
    body_rotaa = body_rotaa.reshape(1, 66)

    ret_dict['body_pose'] = body_rotaa[:, 3:].cpu().numpy()
    ret_dict['global_orient'] = body_rotaa[:, :3].cpu().numpy()
    return ret_dict
