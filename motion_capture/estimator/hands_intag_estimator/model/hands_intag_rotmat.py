import torch

from .hands_dhandsiknet_rotmat import HandsDhandsIKNetRotmatModule
from .model import HandsIntagKps3dModule


class HandsIntagRotmatModule(torch.nn.Module):

    def __init__(self, handnet_gcn, mano_root, device: str,
                 iknet_path: str) -> None:
        super().__init__()
        self.img2kps_model = HandsIntagKps3dModule(handnet_gcn,
                                                   mano_root).to(device)
        self.kps2rot_model = HandsDhandsIKNetRotmatModule(
            device, iknet_path, mano_root)

    def forward(self, img_tensor):
        left_kps3d, right_kps3d = self.img2kps_model(img_tensor)
        left_rotmat, right_rotmat = self.kps2rot_model(left_kps3d, right_kps3d)
        return left_rotmat, right_rotmat
