# yapf: disable
import numpy as np
from typing import Union

from realtime_mocap.motion_capture.optimizer.smooth_filter.builder import (
    build_smooth_filter,
)
from .base_optimizer import BaseOptimizer

# yapf: enable


class K4abtSmoothOptimizer(BaseOptimizer):
    ELBOWS_INDEXES = np.array([6, 13])
    HANDS_INDEXES = np.array([7, 8, 9, 10, 14, 15, 16, 17])
    HEAD_INDEXES = np.array([3, 26, 27, 28, 29, 30, 31])

    def __init__(self,
                 elbow_filter: Union[None, dict] = None,
                 hand_filter: Union[None, dict] = None,
                 head_filter: Union[None, dict] = None,
                 default_filter: Union[None, dict] = None,
                 logger=None) -> None:
        BaseOptimizer.__init__(self, logger=logger)

        self.elbow_filter = None
        self.hand_filter = None
        self.head_filter = None
        self.default_filter = None
        if isinstance(elbow_filter, dict):
            elbow_filter['logger'] = self.logger
            self.elbow_filter = build_smooth_filter(elbow_filter)
        if isinstance(hand_filter, dict):
            hand_filter['logger'] = self.logger
            self.hand_filter = build_smooth_filter(hand_filter)
        if isinstance(head_filter, dict):
            head_filter['logger'] = self.logger
            self.head_filter = build_smooth_filter(head_filter)
        if isinstance(default_filter, dict):
            default_filter['logger'] = self.logger
            self.default_filter = build_smooth_filter(default_filter)

    def forward(self, kps3d: np.ndarray, **kwargs):
        ret_dict = dict()
        ret_dict.update(kwargs)
        if self.default_filter is not None:
            new_kps3d = self.default_filter(kps3d)
        else:
            new_kps3d = kps3d.copy()
        if self.elbow_filter is not None:
            # overwrite elbow joints in new_fullpose
            elbow_pose = kps3d[self.__class__.ELBOWS_INDEXES, :]
            new_elbow_pose = self.elbow_filter(elbow_pose)
            new_kps3d[self.__class__.ELBOWS_INDEXES, :] = new_elbow_pose
        if self.hand_filter is not None:
            # overwrite wrist joints in new_fullpose
            hand_pose = kps3d[self.__class__.HANDS_INDEXES, :]
            new_hand_pose = self.hand_filter(hand_pose)
            new_kps3d[self.__class__.HANDS_INDEXES, :] = new_hand_pose
        if self.head_filter is not None:
            # overwrite head joints in new_fullpose
            head_pose = kps3d[self.__class__.HEAD_INDEXES, :]
            new_head_pose = self.head_filter(head_pose)
            new_kps3d[self.__class__.HEAD_INDEXES, :] = new_head_pose
        ret_dict['kps3d'] = new_kps3d
        return ret_dict
