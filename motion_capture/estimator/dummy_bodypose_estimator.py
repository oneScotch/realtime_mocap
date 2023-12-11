import numpy as np
import time

from .base_estimator import BaseEstimator


class DummyRgbdPytorchBodyPoseEstimator(BaseEstimator):
    BASE_TIME = 0.01883333

    def __init__(self, logger) -> None:
        BaseEstimator.__init__(self, logger=logger)

    def forward(self, img_arr, **kwargs):
        time_offset = (np.random.rand() - 0.5) * 0.5
        forward_time = self.__class__.BASE_TIME * (1 + time_offset)
        time.sleep(forward_time)
        bodypose = np.zeros(shape=(1, 21, 3))
        transl = np.zeros(shape=(1, 3))
        global_orient = np.zeros(shape=(1, 3))
        smplx_dict = dict(
            global_orient=global_orient, transl=transl, bodypose=bodypose)
        return smplx_dict
