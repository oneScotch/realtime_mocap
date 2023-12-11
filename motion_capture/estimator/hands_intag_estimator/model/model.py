# yapf: disable
import torch
import torch.nn as nn
from manopth import manolayer
from mmhuman3d.core.conventions.keypoints_mapping.mano import MANO_REORDER_MAP

from realtime_mocap.motion_capture.estimator.hands_intag_estimator.model.decoder import (  # noqa: E501
    load_decoder,
)
from realtime_mocap.motion_capture.estimator.hands_intag_estimator.model.encoder import (  # noqa: E501
    load_encoder,
)

# yapf: enable


class HandNET_GCN(nn.Module):

    def __init__(self, encoder, mid_model, decoder):
        super(HandNET_GCN, self).__init__()
        self.encoder = encoder
        self.mid_model = mid_model
        self.decoder = decoder
        self.sum_encoder_time = 0.0
        self.sum_time = 0.0
        self.iter = 0

    def forward(self, img):
        hms, mask, dp, img_fmaps, hms_fmaps, dp_fmaps = self.encoder(img)
        global_feature, fmaps = self.mid_model(img_fmaps, hms_fmaps, dp_fmaps)
        # global_feature 1, 2048
        # len(fmaps) = 4
        # [1, 256, 8, 8], [1, 256, 16, 16], [1, 256, 32, 32], [1, 256, 64, 64]
        result = self.decoder(global_feature, fmaps)

        return result


class HandsIntagKps3dModule(nn.Module):

    def __init__(self, handnet_gcn, mano_root) -> None:
        super().__init__()
        # Get J_regressor
        mano_layer = {
            'left':
            manolayer.ManoLayer(
                mano_root=mano_root,
                side='left',
                center_idx=None,
                flat_hand_mean=True,
                use_pca=False),
            'right':
            manolayer.ManoLayer(
                mano_root=mano_root,
                side='right',
                center_idx=None,
                flat_hand_mean=True,
                use_pca=False)
        }
        for side in ['left', 'right']:
            J_regressor = mano_layer[side].th_J_regressor.clone().detach()
            tip_regressor = torch.zeros_like(J_regressor[:5])
            tip_regressor[0, 745] = 1.0  # thumb_tip
            tip_regressor[1, 317] = 1.0  # index_tip
            tip_regressor[2, 444] = 1.0  # middle_tip
            tip_regressor[3, 556] = 1.0  # ring_tip
            tip_regressor[4, 673] = 1.0  # pinky_tip
            J_regressor = torch.cat([J_regressor, tip_regressor], dim=0)
            J_regressor = J_regressor[MANO_REORDER_MAP].contiguous()
            setattr(self, f'{side}_J_regressor', J_regressor)
        self.handnet_gcn = handnet_gcn

    def forward(self, img_tensor):
        verts_dict = self.handnet_gcn(img_tensor)
        left_kps = torch.matmul(self.left_J_regressor, verts_dict['left'])
        right_kps = torch.matmul(self.left_J_regressor, verts_dict['right'])
        return left_kps, right_kps


def load_model(model, train, path):
    encoder, mid_model = load_encoder(model)
    decoder = load_decoder(model, train, path, mid_model.get_info())
    model = HandNET_GCN(encoder, mid_model, decoder)
    return model
