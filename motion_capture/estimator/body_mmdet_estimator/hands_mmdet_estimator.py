# yapf: disable
import cv2
import numpy as np
import torch
from typing import Union

from realtime_mocap.motion_capture.optimizer.builder import build_optimizer
from realtime_mocap.utils.ffmpeg_utils import (
    build_video_writer, try_to_write_frame,
)
from ..base_estimator import BaseEstimator
from ..detect_boost.builder import build_detect_boost
from .bbox_assign import process_mmdet_results, non_max_suppression

try:
    from mmdet.apis import init_detector, inference_detector
    has_mmdet = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception

# yapf: enable


class BodyMmdetEstimator(BaseEstimator):

    def __init__(self,
                 bbox_assign: dict,
                 model_cfg: str,
                 backend_files: str,
                 bbox_threshold: float = 0.1,
                 device: str = 'cuda',
                 bbox_optim: Union[dict, None] = None,
                 detect_boost: Union[dict, None] = None,
                 video_writer: Union[dict, None] = None,
                 logger=None) -> None:
        BaseEstimator.__init__(self, logger=logger)
        if not has_mmdet:
            self.logger.error(import_exception)
            raise ImportError
        self.bbox_threshold = bbox_threshold
        bbox_assign['logger'] = self.logger
        self.bbox_assign = build_bbox_assign(bbox_assign)

        model_cfg = load_config(model_cfg)
        self.model = init_detector(model_cfg, backend_files, device)
        self.input_shape = get_input_shape(deploy_cfg)

        if detect_boost is None:
            self.detect_boost = detect_boost
        else:
            detect_boost['logger'] = self.logger
            self.detect_boost = build_detect_boost(detect_boost)

        if bbox_optim is None:
            self.bbox_optim = bbox_optim
        else:
            bbox_optim['logger'] = self.logger
            bbox_optim_cfg = bbox_optim
            self.bbox_optim = build_optimizer(bbox_optim_cfg)

        self.video_writer_cfg = video_writer
        self.video_writer = None

    def forward(self, img_arr, k4a_pose, **kwargs):
        ret_dict = dict(img_arr=img_arr, k4a_pose=k4a_pose)
        ret_dict.update(kwargs)
        if self.detect_boost is not None:
            img = self.detect_boost.get_image(img_arr, k4a_pose)
        mmdet_results = inference_detector(self.model, img)
        mmdet_box = process_mmdet_results(mmdet_results, cat_id=0, multi_person=True)
        if len(mmdet_box) > 0:
            mmdet_box = non_max_suppression(mmdet_box[0], args.iou_thr)
        else:
            mmdet_box = dict()
        if self.bbox_optim is not None:
            mmdet_box = optim.forward(mmdet_box)
        if self.video_writer_cfg is not None \
                and self.video_writer is None:
            self.video_writer_cfg['logger'] = self.logger
            self.video_path = self.video_writer_cfg['output_path']
            self.video_writer = build_video_writer(self.video_writer_cfg)
            self.video_writer_resolution = [
                self.video_writer_cfg['resolution'][1],  # w
                self.video_writer_cfg['resolution'][0]  # h
            ]
            # re-set self.video_writer_cfg to prevent duplicate build
            self.video_writer_cfg = None
        if self.video_writer is not None:
            canvas = img_arr.copy()
            for bbox_xyxy in mmdet_box:
                if bbox_xyxy is None:
                    continue
                cv2.rectangle(
                    img=canvas,
                    pt1=bbox_xyxy[:2].astype(np.int32),
                    pt2=bbox_xyxy[2:4].astype(np.int32),
                    color=(0, 0, 255),
                    thickness=4,
                )
                conf = float(bbox_xyxy[4])
                cv2.putText(
                    canvas,
                    text=f'{key}: {conf:.2f}',
                    org=bbox_xyxy[:2].astype(np.int32),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=canvas.shape[1] / 1500,
                    color=(0, 0, 255),
                    thickness=2)
            canvas = cv2.resize(canvas, self.video_writer_resolution)
            write_success = try_to_write_frame(
                self.video_writer, img_arr=canvas)
            if not write_success:
                self.video_writer = None
                self.logger.info('Video has been written to ' +
                                 f'{self.video_path}.')
        ret_dict['body_bboxes'] = mmdet_box
        return ret_dict
