# yapf: disable
import numpy as np
from scipy.spatial.transform import Rotation as scipy_Rotation
from xrmocap.data_structure.body_model import SMPLXData

from realtime_mocap.motion_capture.aggregator.base_aggregator import (
    BaseAggregator,
)
from realtime_mocap.utils.ik_utils import (
    rotation_global2local_sjoint, rotation_local2global,
)

# yapf: enable


class Mano16Aggregator(BaseAggregator):

    def __init__(self,
                 mapping_dict: dict,
                 left_hand_mean_path,
                 right_hand_mean_path,
                 joint_parents_path,
                 use_tracking=False,
                 betas_path=None,
                 logger=None):
        super().__init__(logger=logger)
        self.mapping_dict = mapping_dict
        self.hand_mean = dict(
            left=np.load(left_hand_mean_path),
            right=np.load(right_hand_mean_path))
        self.joint_parents = np.load(joint_parents_path)
        self.smplx_hand_joints = [[21, 'right_wrist'], [20, 'left_wrist']]

        if betas_path is None:
            self.betas = np.zeros(shape=(1, 10))
        else:
            self.betas = np.load(betas_path)

        # tracking
        self.tracking_hand = dict(
            left=np.load(left_hand_mean_path),
            right=np.load(right_hand_mean_path))
        self.use_tracking = use_tracking
        self.tracking_cnt = 0
        self.tracking_wrist = dict(right=None, left=None)

    def forward(self, smplx_nested_dict, **kwargs):
        merged_dict = dict()
        for key, mapping_list in self.mapping_dict.items():
            merged_dict[key] = smplx_nested_dict[mapping_list[0]][
                mapping_list[1]]
        body_pose = merged_dict['body_pose']
        global_orient = merged_dict['global_orient']
        transl = merged_dict['transl']
        img_arr = merged_dict['img_arr']
        frame_idx = merged_dict['frame_idx']
        timestamp = merged_dict['timestamp']
        hand_pose = dict()
        wrist_rot_mat = dict()
        for side in ['left', 'right']:
            hand_pose_mat = merged_dict[f'{side}_hand_pose']
            if hand_pose_mat is not None:
                hand_pose[side] = scipy_Rotation.from_matrix(
                    hand_pose_mat[:, 1:, ...].reshape(15, 3, 3)).as_rotvec()
                wrist_rot_mat[side] = hand_pose_mat[:, 0, ...].reshape(1, 3, 3)
                self.tracking_hand[side] = hand_pose[side]
                self.tracking_wrist[side] = wrist_rot_mat[side]
                self.tracking_cnt = 0
            else:
                if self.use_tracking and self.tracking_cnt < 3:
                    hand_pose[side] = self.tracking_hand[side].reshape(
                        -1) * 0.8 + self.hand_mean[side] * 0.2
                    self.tracking_hand[side] = hand_pose[side]
                    if self.tracking_wrist[side] is not None:
                        wrist_rot_mat[side] = self.tracking_wrist[side] * 0.8
                    else:
                        wrist_rot_mat[side] = None
                    self.tracking_wrist[side] = wrist_rot_mat[side]
                    self.tracking_cnt += 1
                else:
                    hand_pose[side] = self.hand_mean[side]
                    wrist_rot_mat[side] = None
                    self.tracking_hand[side] = hand_pose[side]
                    self.tracking_wrist[side] = wrist_rot_mat[side]

        # get wrist rotation
        local_body_pose = np.concatenate((global_orient, body_pose),
                                         axis=0).reshape(-1, 3)
        body_pose = body_pose.reshape(-1, 3)
        local_body_pose = scipy_Rotation.from_rotvec(
            local_body_pose).as_matrix()
        global_body_pose = rotation_local2global(
            local_body_pose,
            self.joint_parents).reshape(*local_body_pose.shape)
        for joint_index, joint_name in self.smplx_hand_joints:
            side = joint_name.split('_')[0]
            wrist_global_rotmat = wrist_rot_mat[side]
            if wrist_global_rotmat is None:
                body_pose[joint_index - 1] = np.zeros(
                    shape=(3, ), dtype=global_orient.dtype)
            else:
                global_body_pose[joint_index] = wrist_global_rotmat
                wrist_local_rotmat = rotation_global2local_sjoint(
                    joint_index=joint_index,
                    global_rot_mats=global_body_pose,
                    parents=self.joint_parents)
                body_pose[joint_index - 1] = scipy_Rotation.from_matrix(
                    wrist_local_rotmat.reshape(3, 3)).as_rotvec().reshape(3)
        fullpose = np.concatenate((
            global_orient.reshape(1, 1, 3),
            body_pose.reshape(1, 21, 3),
            np.zeros(shape=(1, 3, 3), dtype=global_orient.dtype),
            hand_pose['left'].reshape(1, 15, 3),
            hand_pose['right'].reshape(1, 15, 3),
        ),
                                  axis=1)
        smplx_data = SMPLXData(
            fullpose=fullpose,
            transl=np.expand_dims(transl, axis=0),
            betas=self.betas)
        return dict(
            smplx_data=smplx_data,
            frame_idx=frame_idx,
            timestamp=timestamp,
            img_arr=img_arr)
