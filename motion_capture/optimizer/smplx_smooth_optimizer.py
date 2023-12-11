# yapf: disable
import numpy as np
from scipy.spatial.transform import Rotation as scipy_Rotation
from typing import Union
from xrmocap.data_structure.body_model import SMPLXData

from realtime_mocap.motion_capture.optimizer.smooth_filter.builder import (
    build_smooth_filter,
)
from .base_optimizer import BaseOptimizer

# yapf: enable


class SmplxSmoothOptimizer(BaseOptimizer):
    ELBOWS_INDEXES = np.array([18, 19])
    WRISTS_INDEXES = np.array([20, 21])
    HEAD_INDEXES = np.array([12, 15])

    def __init__(self,
                 elbow_filter: Union[None, dict] = None,
                 wrist_filter: Union[None, dict] = None,
                 head_filter: Union[None, dict] = None,
                 fullpose_filter: Union[None, dict] = None,
                 transl_filter: Union[None, dict] = None,
                 logger=None) -> None:
        """Smooth optimizer for smplx.

        Args:
            elbow_filter (Union[None, dict], optional):
                Filter for left and right elbows. If elbow_filter
                is not None, wrists will not be smoothed by
                fullpose_filter.
                Defaults to None.
            wrist_filter (Union[None, dict], optional):
                Filter for left and right wrists. If wrist_filter
                is not None, wrists will not be smoothed by
                fullpose_filter.
                Defaults to None.
            head_filter (Union[None, dict], optional):
                Filter for head and neck. If head_filter
                is not None, head joints will not be smoothed by
                fullpose_filter.
                Defaults to None.
            fullpose_filter (Union[None, dict], optional):
                Filter for smplx fullpose. It has a lower
                priority than the filters above.
                Defaults to None.
            transl_filter (Union[None, dict], optional):
                Filter for smplx transl. It has a lower
                priority than the filters above.
                Defaults to None.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseOptimizer.__init__(self, logger=logger)

        self.elbow_filter = None
        self.wrist_filter = None
        self.head_filter = None
        self.fullpose_filter = None
        self.transl_filter = None
        if isinstance(elbow_filter, dict):
            elbow_filter['logger'] = self.logger
            self.elbow_filter = build_smooth_filter(elbow_filter)
        if isinstance(wrist_filter, dict):
            wrist_filter['logger'] = self.logger
            self.wrist_filter = build_smooth_filter(wrist_filter)
        if isinstance(head_filter, dict):
            head_filter['logger'] = self.logger
            self.head_filter = build_smooth_filter(head_filter)
        if isinstance(fullpose_filter, dict):
            fullpose_filter['logger'] = self.logger
            self.fullpose_filter = build_smooth_filter(fullpose_filter)
        if isinstance(transl_filter, dict):
            transl_filter['logger'] = self.logger
            self.transl_filter = build_smooth_filter(transl_filter)

    def forward(self, smplx_data: SMPLXData, **kwargs):
        ret_dict = dict()
        ret_dict.update(kwargs)
        fullpose = smplx_data.get_fullpose()
        fullpose_rotmat = scipy_Rotation.from_rotvec(fullpose.reshape(
            -1, 3)).as_matrix()
        transl = smplx_data.get_transl()
        if self.fullpose_filter is not None:
            new_fullpose_rotmat = self.fullpose_filter(fullpose_rotmat)
        else:
            new_fullpose_rotmat = fullpose_rotmat
        if self.elbow_filter is not None:
            # overwrite elbow joints in new_fullpose
            elbow_pose = fullpose_rotmat[self.__class__.ELBOWS_INDEXES, :]
            new_elbow_pose = self.elbow_filter(elbow_pose)
            new_fullpose_rotmat[
                self.__class__.ELBOWS_INDEXES, :] = new_elbow_pose
        if self.wrist_filter is not None:
            # overwrite wrist joints in new_fullpose
            wrist_pose = fullpose_rotmat[self.__class__.WRISTS_INDEXES, :]
            new_wrist_pose = self.wrist_filter(wrist_pose)
            new_fullpose_rotmat[
                self.__class__.WRISTS_INDEXES, :] = new_wrist_pose
        if self.head_filter is not None:
            # overwrite head joints in new_fullpose
            head_pose = fullpose_rotmat[self.__class__.HEAD_INDEXES, :]
            new_head_pose = self.head_filter(head_pose)
            new_fullpose_rotmat[self.__class__.HEAD_INDEXES, :] = new_head_pose
        new_fullpose = scipy_Rotation.from_matrix(
            new_fullpose_rotmat).as_rotvec().reshape(1, -1, 3)
        smplx_data.set_fullpose(new_fullpose)
        if self.transl_filter is not None:
            transl = self.transl_filter(transl)
            smplx_data.set_transl(transl)
        ret_dict['smplx_data'] = smplx_data
        return ret_dict
