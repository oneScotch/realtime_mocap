import numpy as np
from typing import Union

from ..base_estimator import BaseEstimator
from ..detect_boost.builder import build_detect_boost
from .bbox_assign import build_bbox_assign

try:
    from xrmocap.human_perception.builder import build_detector
    has_xrmocap = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_xrmocap = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception


class HandsMmdetEstimator(BaseEstimator):

    def __init__(self,
                 bbox_assign: Union[dict, None],
                 bbox_threshold: float = 0.1,
                 xrmocap_detector: Union[dict, None] = None,
                 detect_boost: Union[dict, None] = None,
                 video_writer: Union[dict, None] = None,
                 logger=None) -> None:
        BaseEstimator.__init__(self, logger=logger)
        if not has_xrmocap:
            self.logger.error(import_exception)
            raise ImportError
        self.bbox_threshold = bbox_threshold
        bbox_assign['logger'] = self.logger
        self.bbox_assign = build_bbox_assign(bbox_assign)

        xrmocap_detector['logger'] = self.logger
        self.xrmocap_detector = build_detector(xrmocap_detector)
        if detect_boost is None:
            detect_boost['logger'] = self.logger
            self.detect_boost = detect_boost
        else:
            self.detect_boost = build_detect_boost(detect_boost)

    def forward(self, img_arr, k4a_pose, **kwargs):
        ret_dict = dict(img_arr=img_arr, k4a_pose=k4a_pose)
        ret_dict.update(kwargs)
        if self.detect_boost is not None:
            img = self.detect_boost.get_image(img_arr, k4a_pose)
        ret_list = self.xrmocap_detector.infer_array(
            image_array=np.expand_dims(img, axis=0),
            disable_tqdm=True,
            multi_person=True)
        hands_bboxes = []
        for idx in range(len(ret_list[0])):
            if ret_list[0][idx][4] > self.bbox_threshold:
                points2d = np.array(
                    [ret_list[0][idx][:2],
                     ret_list[0][idx][2:4]]).reshape(1, 1, 2, 2)
                points2d = self.detect_boost.get_points(points2d)
                bbox_xyxys = ret_list[0][idx].copy()
                bbox_xyxys[:4] = points2d.reshape(-1)
                hands_bboxes.append(bbox_xyxys)
        if len(hands_bboxes) > 0:
            hands_bboxes = self.bbox_assign.assign_bboxes(
                hands_bboxes, k4a_pose)
        else:
            hands_bboxes = dict(left=None, right=None, together=None)
        ret_dict['hands_bboxes'] = hands_bboxes
        return ret_dict
