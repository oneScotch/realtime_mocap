# yapf: disable
import cv2
import numpy as np
import torch
from typing import Union
from xrmocap.data_structure.body_model import SMPLXData
from xrprimer.data_structure.camera import PinholeCameraParameter

from realtime_mocap.utils.ffmpeg_utils import (
    build_video_writer, try_to_write_frame,
)
from .base_application import BaseApplication
from .body_model.builder import build_body_model

try:
    from smplx_mpr.render import MinimalVerticesRenderer
    has_smplx_mpr = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_smplx_mpr = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception
# yapf: enable


class SmplxMprVisualization(BaseApplication):

    def __init__(self,
                 device: str,
                 body_model: dict,
                 pinhole_param_path: str,
                 window_size: Union[int, None] = 1280,
                 video_writer: Union[dict, None] = None,
                 original_video_writer: Union[dict, None] = None,
                 logger=None) -> None:
        BaseApplication.__init__(self, logger=logger)
        if not has_smplx_mpr:
            self.logger.error(import_exception)
            raise ImportError
        self.device = device
        self.pinhole_param = PinholeCameraParameter.fromfile(
            pinhole_param_path)
        # set device before build always fails
        # body_model['device'] = device
        body_model = build_body_model(body_model)
        self.body_model = body_model.to(self.device)
        self.window_size = window_size
        self.new_size = None
        self.renderer = MinimalVerticesRenderer(
            device=device,
            faces=body_model.faces_tensor.clone().detach(),
            pinhole_param=self.pinhole_param,
            logger=logger)

        self.video_writer_cfg = video_writer
        self.video_writer = None
        self.original_video_writer_cfg = original_video_writer
        self.original_video_writer = None

    def forward(self,
                smplx_data: SMPLXData,
                img_arr: np.ndarray = None,
                **kwargs):
        with torch.no_grad():
            param_dict = smplx_data.to_tensor_dict(device=self.device)
            mesh_result = self.body_model(
                betas=param_dict['betas'],
                global_orient=param_dict['global_orient'],
                body_pose=param_dict['body_pose'],
                left_hand_pose=param_dict['left_hand_pose'].reshape(1, 45),
                right_hand_pose=param_dict['right_hand_pose'].reshape(1, 45),
                transl=param_dict['transl'],
                expression=param_dict['expression'],
                jaw_pose=param_dict['jaw_pose'],
                leye_pose=param_dict['leye_pose'],
                reye_pose=param_dict['reye_pose'],
                **kwargs)
            verts = mesh_result['vertices'][0]
            img = self.renderer(vertices=verts, bg=img_arr)
            if self.window_size is not None:
                if self.new_size is None:
                    resize_ratio = \
                        self.window_size / max(img.shape[:2])
                    self.new_size = (int(img.shape[1] * resize_ratio),
                                     int(img.shape[0] * resize_ratio))
                img = cv2.resize(img, self.new_size)
                cv2.imshow('SmplxMprVisualization', img)
                cv2.waitKey(1)
            # if there's a writer to be constructed
            if self.video_writer_cfg is not None \
                    and self.video_writer is None:
                self.video_writer_cfg['logger'] = self.logger
                self.video_path = self.video_writer_cfg['output_path']
                self.video_writer = build_video_writer(self.video_writer_cfg)
                self.video_writer_resolution = [
                    self.video_writer_cfg['resolution'][1],
                    self.video_writer_cfg['resolution'][0]
                ]
                # re-set self.video_writer_cfg to prevent duplicate build
                self.video_writer_cfg = None
            # write images to video and destroy it at last
            if self.video_writer is not None:
                canvas = img.copy()
                canvas = cv2.resize(canvas, self.video_writer_resolution)
                write_success = try_to_write_frame(
                    self.video_writer, img_arr=canvas)
                if not write_success:
                    self.video_writer = None
                    self.logger.info('Video has been written to ' +
                                     f'{self.video_path}.')

            if self.original_video_writer_cfg is not None \
                    and self.original_video_writer is None:
                self.original_video_writer_cfg['logger'] = self.logger
                self.original_video_path = self.original_video_writer_cfg['output_path']
                self.original_video_writer = build_video_writer(self.original_video_writer_cfg)
                self.original_video_writer_resolution = [
                    self.original_video_writer_cfg['resolution'][1],
                    self.original_video_writer_cfg['resolution'][0]
                ]
                # re-set self.video_writer_cfg to prevent duplicate build
                self.original_video_writer_cfg = None
            # write images to video and destroy it at last
            if self.original_video_writer is not None:
                canvas_arr = img_arr.copy()
                canvas_arr = cv2.resize(canvas_arr, self.original_video_writer_resolution)
                write_success = try_to_write_frame(
                    self.original_video_writer, img_arr=canvas_arr)
                if not write_success:
                    self.original_video_writer = None
                    self.logger.info('Video has been written to ' +
                                     f'{self.original_video_path}.')
        return img

    def __del__(self):
        cv2.destroyAllWindows()
