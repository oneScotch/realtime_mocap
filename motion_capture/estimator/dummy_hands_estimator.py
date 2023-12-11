import numpy as np
import time

from .base_estimator import BaseEstimator


class DummyHandsEstimator(BaseEstimator):
    BASE_TIME = 0.028

    def __init__(self, logger) -> None:
        BaseEstimator.__init__(self, logger=logger)

    def forward(self, img_arr, k4a_pose, **kwargs):
        self.crop_hands(img_arr, k4a_pose)
        time_offset = (np.random.rand() - 0.5) * 0.5
        forward_time = self.__class__.BASE_TIME * (1 + time_offset)
        time.sleep(forward_time)
        left_hand_pose = np.zeros(shape=(1, 15, 3))
        right_hand_pose = np.zeros(shape=(1, 15, 3))
        smplx_dict = dict(
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )
        return smplx_dict

    def crop_hands(self, img_arr, k4a_pose):
        time.sleep(0.005)
