import torch
import torch.nn as nn
from einops import rearrange
from manopth import manolayer
from manopth.rot6d import compute_rotation_matrix_from_ortho6d

from .base_optimizer import BaseOptimizer

SHAPE_MEAN = [
    0.15910966, 1.24914071, -0.64187749, 0.5400079, -3.29426494, -0.83807857,
    0.31873315, -0.22016137, 1.33633995, 0.83711511
]

SHAPE_STD = [
    0.99800293, 1.66176237, 1.40137928, 1.46973533, 2.23193049, 1.57620704,
    1.62400539, 2.8165403, 1.76102397, 1.30528381
]


class dense_bn(nn.Module):

    def __init__(self, inc, ouc):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(inc, ouc, True), nn.BatchNorm1d(ouc), nn.Sigmoid())

    def forward(self, x):
        return self.dense(x)


class IKNet(nn.Module):

    def __init__(self, inc, depth, width, joints=16):
        super().__init__()
        self.dense = dense_bn(inc, width)

        self.dense_1 = dense_bn(width, width)
        self.dense_2 = dense_bn(width, width)
        self.dense_3 = dense_bn(width, width)
        self.dense_4 = dense_bn(width, width)
        self.dense_5 = dense_bn(width, width)

        # joints * 6D rotation and 1 for shape estimation
        self.dense_6 = nn.Linear(width, joints * 6 + 1)

    def forward(self, x):
        x = rearrange(x, 'b j c -> b (j c)', c=3)
        x = self.dense(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.dense_5(x)
        x = self.dense_6(x)

        theta_raw = x[:, :-1]
        shape = x[:, -1]
        theta_raw = rearrange(theta_raw, 'b (j n) -> b j n', n=6)
        return theta_raw, shape


class IKNetOptimizer(BaseOptimizer):
    """IKNet for optimizing rotations from given keypoints.

    It is only for right MANO model.
    """

    def __init__(self,
                 device: str,
                 side: str,
                 iknet_path: str,
                 mano_root: str,
                 logger=None) -> None:
        """Initialization of AIKOptimizer.

        Args:
            logger: Logger of the optimizer. Defaults to None.
        """
        super().__init__(logger=logger)
        self.logger = logger
        self.device = device
        self.iknet = IKNet(inc=21 * 3, depth=6, width=1024).to(device)
        self.iknet.load_state_dict(torch.load(iknet_path))
        self.iknet.eval()
        if side != 'right':
            self.logger.error(
                'The IKNet is trained for right hand, please set it to left.')
            raise TypeError
        self.right_mano_layer = manolayer.ManoLayer(
            mano_root=mano_root,
            side='right',
            use_pca=False,
            flat_hand_mean=True,
            root_rot_mode='rotmat',
            joint_rot_mode='rotmat').to(device)
        self.shape_mean = torch.Tensor(SHAPE_MEAN)[None, :].to(self.device)
        self.shape_std = torch.Tensor(SHAPE_STD)[None, :].to(self.device)

    def forward(self, kps3d, side='right', return_joints=False, **kwargs):
        """Forward function of AIKOptimizer.

        Args:
            kps3d (np.ndarray): (b, 21, 3) positions.
        """
        ref_bone_length = torch.linalg.norm(
            kps3d[:, 0] - kps3d[:, 9], dim=1,
            keepdim=True)[:, None, :].repeat(1, 21, 3)
        kps3d = (kps3d -
                 kps3d[:, 9][:, None, :].repeat(1, 21, 1)) / ref_bone_length

        # IKNet forwarding
        with torch.no_grad():
            rot6d, shape = self.iknet(kps3d)

        shape = self.shape_mean.repeat(len(shape), 1) + shape[:, None].repeat(
            1, 10) * self.shape_std.repeat(len(shape), 1)
        rot6d = rearrange(rot6d, 'b j c -> (b j) c')
        rotmat = compute_rotation_matrix_from_ortho6d(rot6d)
        rotmat = rearrange(rotmat, '(b j) h w -> b j h w', j=16)
        if return_joints:
            verts, xyz_ik = self.right_mano_layer(rotmat, shape)
            verts = verts.cpu().numpy()
            xyz_ik = xyz_ik.cpu().numpy()
            if side == 'left':
                wrist_j = xyz_ik[:, 0, 0].copy()
                xyz_ik[:, :, 0] = -xyz_ik[:, :, 0]
                offset_j = wrist_j - xyz_ik[:, 0, 0]
                xyz_ik[:, :, 0] += offset_j
            return rotmat.cpu().numpy(), xyz_ik
        else:
            return rotmat.cpu().numpy()

    def __call__(self, kps3d, side, return_joints=False, **kwargs):
        return self.forward(kps3d, side, return_joints, **kwargs)
