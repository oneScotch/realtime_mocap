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
from .bbox_assign import build_bbox_assign

try:
    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape, load_config
    has_mmdeploy = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mmdeploy = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception

# yapf: enable


class HandsMmdetTRTEstimator(BaseEstimator):

    def __init__(self,
                 bbox_assign: dict,
                 deploy_cfg: str,
                 model_cfg: str,
                 backend_files: str,
                 bbox_threshold: float = 0.1,
                 device: str = 'cuda',
                 bbox_optim: Union[dict, None] = None,
                 detect_boost: Union[dict, None] = None,
                 video_writer: Union[dict, None] = None,
                 logger=None) -> None:
        BaseEstimator.__init__(self, logger=logger)
        if not has_mmdeploy:
            self.logger.error(import_exception)
            raise ImportError
        self.bbox_threshold = bbox_threshold
        bbox_assign['logger'] = self.logger
        self.bbox_assign = build_bbox_assign(bbox_assign)

        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
        self.task_processor = build_task_processor(model_cfg, deploy_cfg,
                                                   device)
        self.mmdet_trt = self.task_processor.init_backend_model(backend_files)
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
            self.bbox_optim = dict(
                left=build_optimizer(bbox_optim_cfg),
                right=build_optimizer(bbox_optim_cfg),
                together=build_optimizer(bbox_optim_cfg))

        self.video_writer_cfg = video_writer
        self.video_writer = None

    def forward(self, img_arr, k4a_pose, **kwargs):
        ret_dict = dict(img_arr=img_arr, k4a_pose=k4a_pose)
        ret_dict.update(kwargs)
        if self.detect_boost is not None:
            img = self.detect_boost.get_image(img_arr, k4a_pose)
        model_inputs, _ = self.task_processor.create_input(
            img, self.input_shape)
        with torch.no_grad():
            ret_list = self.task_processor.run_inference(
                self.mmdet_trt, model_inputs)

        hands_bboxes = []
        for idx in range(len(ret_list[0][0])):
            if ret_list[0][0][idx][4] > self.bbox_threshold:
                points2d = np.array(
                    [ret_list[0][0][idx][:2],
                     ret_list[0][0][idx][2:4]]).reshape(1, 1, 2, 2)
                points2d = self.detect_boost.get_points(points2d)
                bbox_xyxys = ret_list[0][0][idx].copy()
                bbox_xyxys[:4] = points2d.reshape(-1)
                hands_bboxes.append(bbox_xyxys)
        if len(hands_bboxes) > 0:
            hands_bboxes = self.bbox_assign.assign_bboxes(
                hands_bboxes, k4a_pose, img_arr.shape[1], img_arr.shape[0])
        else:
            hands_bboxes = dict(left=None, right=None, together=None)
        if self.bbox_optim is not None:
            for k, optim in self.bbox_optim.items():
                hands_bboxes[k] = optim.forward(hands_bboxes[k])
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
            for key, bbox_xyxy in hands_bboxes.items():
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
        ret_dict['hands_bboxes'] = hands_bboxes
        return ret_dict
