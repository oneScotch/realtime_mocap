import numpy as np
import time
from scipy.spatial.transform import Rotation as scipy_Rotation
from typing import List
from xrmocap.data_structure.body_model.smplx_data import SMPLXData

from .base_source import BaseSource


class SMPLXHandposeSource(BaseSource):
    BASE_TIME = 0.033333

    def __init__(self,
                 global_orient_order: str,
                 global_orient_eular: List[int],
                 logger=None) -> None:
        BaseSource.__init__(self=self, logger=logger)
        self.global_orient = scipy_Rotation.from_euler(
            seq=global_orient_order,
            angles=np.array(global_orient_eular),
            degrees=True).as_rotvec()
        self.last_left_hand_pose = None
        self.last_left_hand_pose_time = None
        self.prev_time = None

    def get_data(self, **kwargs):
        if self.prev_time is None:
            self.prev_time = time.time()
        else:
            time_to_wait = self.__class__.BASE_TIME + \
                self.prev_time - time.time()
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            self.prev_time = time.time()
        # might be None
        ret_dict = {}
        if self.last_left_hand_pose is None:
            left_hand_pose = np.zeros(shape=(1, 15, 3))
            self.last_left_hand_pose = left_hand_pose
            self.last_left_hand_pose_time = time.time()
        else:
            left_hand_pose = self.last_left_hand_pose.copy()
            if time.time() - self.last_left_hand_pose_time > 2.0:
                for i in range(left_hand_pose.shape[1]):
                    for j in range(left_hand_pose.shape[2]):
                        offset = (np.random.rand() - 0.5) * 0.25
                        left_hand_pose[0, i, j] += offset
                self.last_left_hand_pose = left_hand_pose
                self.last_left_hand_pose_time = time.time()
        right_hand_pose = np.zeros(shape=(1, 15, 3))
        fullpose = np.concatenate((
            self.global_orient.reshape(1, 1, 3),
            np.zeros(shape=(1, 21, 3)),
            np.zeros(shape=(1, 3, 3)),
            left_hand_pose,
            right_hand_pose,
        ),
                                  axis=1)
        smplx_data = SMPLXData(
            fullpose=fullpose,
            transl=np.array([[0.17405687, 0.76408563, 1.43093603]]),
            betas=np.zeros(shape=(1, 10)))
        ret_dict['smplx_data'] = smplx_data
        return ret_dict
