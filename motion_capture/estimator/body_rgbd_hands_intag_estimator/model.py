# yapf: disable
import torch

from ..body_rgbd_estimator.nn_module import BodyRgbdRotmatModule
from ..hands_intag_estimator.model.hands_intag_rotmat import (
    HandsIntagRotmatModule,
)

# yapf: enable


class RgbdIntagHandModule(torch.nn.Module):

    def __init__(self,
                 handnet_gcn,
                 mano_root,
                 iknet_path,
                 rnn_checkpoint_path,
                 device='cuda') -> None:
        super().__init__()
        self.intaghand_iknet = HandsIntagRotmatModule(
            handnet_gcn=handnet_gcn,
            mano_root=mano_root,
            device=device,
            iknet_path=iknet_path)
        self.rgbd_bodypose_module = BodyRgbdRotmatModule(
            rnn_checkpoint_path=rnn_checkpoint_path, device=device)

    def forward(self, img_tensor, kinect_kp, pose_init, twists, hidden):
        left_rotmat, right_rotmat = self.intaghand_iknet(img_tensor)
        body_pose_rotmat, hidden = self.rgbd_bodypose_module(
            kinect_kp, pose_init, twists, hidden)
        return left_rotmat, right_rotmat, body_pose_rotmat, hidden
