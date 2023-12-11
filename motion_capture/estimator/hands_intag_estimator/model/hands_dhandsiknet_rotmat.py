# yapf: disable
import torch
from einops import rearrange
from manopth import manolayer
from manopth.rot6d import compute_rotation_matrix_from_ortho6d
from xrprimer.utils.log_utils import get_logger

from realtime_mocap.motion_capture.optimizer.iknet_optimizer import (
    SHAPE_MEAN, SHAPE_STD, IKNet,
)
from realtime_mocap.utils.ik_utils import (
    rotation_global2local_torch, rotation_local2global_torch,
)

# yapf: enable
MANO_PARENTS = [
    -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15
][:16]
FLIP_MAT = torch.eye(3)
FLIP_MAT[0, 0] = -1


class HandsDhandsIKNetRotmatModule(torch.nn.Module):
    """IKNet for optimizing rotations from given keypoints.

    It is only for right MANO model.
    """

    def __init__(self,
                 device: str,
                 iknet_path: str,
                 mano_root: str,
                 logger=None) -> None:
        """Initialization of AIKOptimizer.

        Args:
            logger: Logger of the optimizer. Defaults to None.
        """
        super().__init__()
        self.logger = get_logger(logger)
        self.logger = logger
        self.device = device
        self.left_iknet = IKNet(inc=21 * 3, depth=6, width=1024).to(device)
        self.left_iknet.load_state_dict(torch.load(iknet_path))
        self.left_iknet.eval()
        self.right_iknet = IKNet(inc=21 * 3, depth=6, width=1024).to(device)
        self.right_iknet.load_state_dict(torch.load(iknet_path))
        self.right_iknet.eval()
        self.right_mano_layer = manolayer.ManoLayer(
            mano_root=mano_root,
            side='right',
            use_pca=False,
            flat_hand_mean=True,
            root_rot_mode='rotmat',
            joint_rot_mode='rotmat').to(device)

    def forward(self, left_kps3d, right_kps3d):
        """Forward function of double hand iknet.

        Args:
            kps3d (np.ndarray): (b, 21, 3) positions.
        """
        shape_mean = torch.Tensor(SHAPE_MEAN)[None, :].to(self.device)
        shape_std = torch.Tensor(SHAPE_STD)[None, :].to(self.device)
        results = []
        # flip left kps like right
        flipped_left_kps3d_x = left_kps3d[:, :, 0:1] * (-1)
        wrist_j = left_kps3d[:, 0, 0]
        offset_j = wrist_j + left_kps3d[:, 0, 0]
        flipped_left_kps3d_x += offset_j
        flipped_left_kps3d_yz = left_kps3d[:, :, 1:]
        flipped_left_kps3d = torch.cat(
            (flipped_left_kps3d_x, flipped_left_kps3d_yz), dim=2)
        for kps3d, iknet in zip([flipped_left_kps3d, right_kps3d],
                                [self.left_iknet, self.right_iknet]):
            ref_bone_length = norm_vec(kps3d[:, 0] - kps3d[:, 9]).reshape(
                1, 1, 1).repeat(1, 21, 3)
            kps3d = (kps3d - kps3d[:, 9][:, None, :].repeat(
                1, 21, 1)) / ref_bone_length

            rot6d, shape = iknet(kps3d)
            shape = shape_mean.repeat(len(shape), 1) + shape[:, None].repeat(
                1, 10) * shape_std.repeat(len(shape), 1)

            rot6d = rearrange(rot6d, 'b j c -> (b j) c')
            rotmat = compute_rotation_matrix_from_ortho6d(rot6d)
            rotmat = rearrange(rotmat, '(b j) h w -> b j h w', j=16)
            results.append(rotmat)
        # flip left rotation
        results[0] = flip_rotmat_torch(results[0])
        return results[0], results[1]


def norm_vec(vec):
    vec = vec.reshape(3)
    norm = vec[0]**2 + vec[1]**2 + vec[2]**2
    norm = torch.sqrt(norm)
    return norm


def flip_rotmat_torch(local_rotmat):
    global_rotmat = rotation_local2global_torch(
        rot_mats=local_rotmat, parents=MANO_PARENTS)
    flip_mat = FLIP_MAT.to(
        device=local_rotmat.device, dtype=local_rotmat.dtype)
    ret_list = []
    for joint_index in range(len(global_rotmat)):
        flipped_mat = flip_mat @ global_rotmat[joint_index] @ flip_mat
        ret_list.append(flipped_mat.unsqueeze(0))
    global_rotmat = torch.cat(ret_list, dim=0)
    local_rotmat = rotation_global2local_torch(
        global_rotmat, parents=MANO_PARENTS)
    return local_rotmat
