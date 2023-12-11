# yapf: disable
import numpy as np
from typing import Union

from realtime_mocap.motion_capture.optimizer.smooth_filter.builder import (
    build_smooth_filter,
)
from .base_optimizer import BaseOptimizer


# yapf: enable
class BboxSmoothOptimizer(BaseOptimizer):

    def __init__(self,
                 size_filter: Union[None, dict] = None,
                 location_filter: Union[None, dict] = None,
                 logger=None) -> None:
        BaseOptimizer.__init__(self, logger=logger)
        self.size_filter_cfg = size_filter
        self.size_filter = None
        self.location_filter_cfg = location_filter
        self.location_filter = None
        self.filter_life = 0
        self.last_size = None
        self.last_location = None
        self.ready_for_reset = False

    def forward(self, bbox_xyxy: np.ndarray, **kwargs):
        if bbox_xyxy is None:
            # filter memory valid, but mark ready_for_reset
            if self.filter_life >= 2 and not self.ready_for_reset:
                new_bbox_xyxy = np.zeros(shape=(5, ))
                half_wid = self.last_size[0] / 2
                new_bbox_xyxy[0] = self.last_location[0] - half_wid
                new_bbox_xyxy[2] = self.last_location[0] + half_wid
                half_hei = self.last_size[1] / 2
                new_bbox_xyxy[1] = self.last_location[1] - half_hei
                new_bbox_xyxy[3] = self.last_location[1] + half_hei
                new_bbox_xyxy[4] = 0.0123
                bbox_xyxy = new_bbox_xyxy
                self.ready_for_reset = True
                return bbox_xyxy
            else:
                self.filter_life = 0
                self.last_size = None
                self.last_location = None
                self.size_filter = None
                self.location_filter = None
                self.ready_for_reset = False
                return None
        # get a valid bbox, not ready for reset
        self.ready_for_reset = False
        wid_hei = bbox_xyxy[2:4] - bbox_xyxy[0:2]
        center = (bbox_xyxy[0:2] + bbox_xyxy[2:4]) / 2
        # the first bbox, create filter and return None
        if self.last_location is None:
            self.last_size = wid_hei
            self.last_location = center
            if self.size_filter_cfg is not None:
                self.size_filter = build_smooth_filter(self.size_filter_cfg)
                self.size_filter.forward(wid_hei)
            if self.location_filter_cfg is not None:
                self.filter_life = 1
                self.location_filter = build_smooth_filter(
                    self.location_filter_cfg)
                self.location_filter.forward(center)
            return None
        # smooth the current bbox with filter
        else:
            self.last_size = wid_hei
            self.last_location = center
            if self.size_filter is not None:
                wid_hei = self.size_filter.forward(wid_hei)
            if self.location_filter is not None:
                self.filter_life += 1
                center = self.location_filter.forward(center)
            new_bbox_xyxy = np.zeros(shape=(5, ))
            half_wid = wid_hei[0] / 2
            new_bbox_xyxy[0] = center[0] - half_wid
            new_bbox_xyxy[2] = center[0] + half_wid
            half_hei = wid_hei[1] / 2
            new_bbox_xyxy[1] = center[1] - half_hei
            new_bbox_xyxy[3] = center[1] + half_hei
            new_bbox_xyxy[4] = bbox_xyxy[4]
            bbox_xyxy = new_bbox_xyxy
            return bbox_xyxy
