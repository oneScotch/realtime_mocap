import numpy as np
import os
import torch

from realtime_mocap.utils.geometry_utils import batch_rodrigues


class HandsFrankRotmatModule(torch.nn.Module):

    def __init__(self, h3d_encoder, mean_hand_dir, device) -> None:
        super().__init__()
        self.device = device
        self.h3d_encoder = h3d_encoder.to(self.device)
        self.mean_hand_list = []
        for side in ('left', 'right'):
            mean_hand = np.load(
                os.path.join(mean_hand_dir,
                             f'{side}_hand_pose_mean.npy')).reshape(1, 45)
            mean_hand = torch.from_numpy(mean_hand).to(self.device)
            self.mean_hand_list.append(mean_hand)

    def forward(self, dhands_img):
        frank_output = self.h3d_encoder(dhands_img)
        pred_pose = frank_output[:, 3:51]
        # for left hand
        left_pose = pred_pose[0:1, ...].reshape(1, 16, 3)
        left_pose_0 = left_pose[:, :, 0:1]
        left_pose_1 = left_pose[:, :, 1:2] * (-1)
        left_pose_2 = left_pose[:, :, 2:3] * (-1)
        left_pose = torch.cat((left_pose_0, left_pose_1, left_pose_2),
                              dim=2).reshape(1, -1)
        left_wrist = left_pose[:, :3]
        left_hand_pose = left_pose[:, 3:] + self.mean_hand_list[0]
        left_pose = torch.cat((
            left_wrist,
            left_hand_pose,
        ), dim=1).reshape(1, -1)
        left_rotmat = batch_rodrigues(
            left_pose.view(-1, 3), dtype=left_pose.dtype).view([1, -1, 3, 3])
        # for right hand
        right_pose = pred_pose[1:2, ...].reshape(1, -1)
        right_wrist = right_pose[:, :3]
        right_hand_pose = right_pose[:, 3:] + self.mean_hand_list[1]
        right_pose = torch.cat((
            right_wrist,
            right_hand_pose,
        ), dim=1).reshape(1, -1)
        right_rotmat = batch_rodrigues(
            right_pose.view(-1, 3), dtype=right_pose.dtype).view([1, -1, 3, 3])
        ret_tensor = torch.cat((left_rotmat, right_rotmat), dim=0)
        return ret_tensor
