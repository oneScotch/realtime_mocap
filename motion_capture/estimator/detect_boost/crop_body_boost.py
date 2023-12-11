# yapf: disable
import cv2
import numpy as np
from typing import Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.ops.projection.builder import OpencvProjector

from realtime_mocap.utils.ffmpeg_utils import (
    build_video_writer, try_to_write_frame,
)
from .base_boost import BaseBoost

# yapf: enable


class CropBodyBoost(BaseBoost):

    def __init__(self,
                 cam_param_path: str,
                 scale: float = 1.2,
                 video_writer: Union[dict, None] = None,
                 logger=None) -> None:
        BaseBoost.__init__(self, logger)
        try:
            cam_param = PinholeCameraParameter.fromfile(cam_param_path)
        except ValueError:
            self.logger.warning('Log error [File content is not correct.] ' +
                                'is for loading PinholeCameraParameter. ' +
                                'If you see [FisheyeCameraParameter ' +
                                'has been loaded.], ' +
                                'this msg can be ignored.')
            cam_param = FisheyeCameraParameter.fromfile(cam_param_path)
            self.logger.info('FisheyeCameraParameter has been loaded.')
        self.projector = OpencvProjector(
            camera_parameters=[
                cam_param,
            ], logger=self.logger)

        self.scale = scale
        self.last_offset = None

        self.video_writer_cfg = video_writer
        self.video_writer = None

    def get_image(self, img_arr: np.ndarray, k4a_pose: np.ndarray, **kwargs):
        kps3d = (k4a_pose[0, :, 2:5] / 1000).astype(np.float32)
        height = img_arr.shape[0]
        width = img_arr.shape[1]
        kps2d = self.projector.project(points=kps3d)[0]
        # this bbox may go out of boundary
        bbox_xyxy = np.array([
            np.min(kps2d[:, 0]),
            np.min(kps2d[:, 1]),
            np.max(kps2d[:, 0]),
            np.max(kps2d[:, 1]),
        ])
        center = (bbox_xyxy[:2] + bbox_xyxy[2:]) / 2.0
        bbox_half_width = (bbox_xyxy[2] - bbox_xyxy[0]) / 2.0 * self.scale
        bbox_half_height = (bbox_xyxy[3] - bbox_xyxy[1]) / 2.0 * self.scale
        half_edge = max(bbox_half_width, bbox_half_height)
        # this bbox will be always inside img_arr
        bbox_xyxy_origin = np.array([
            max(0, center[0] - bbox_half_width),
            max(0, center[1] - bbox_half_height),
            min(width - 1, center[0] + bbox_half_width),
            min(height - 1, center[0] + bbox_half_height),
        ]).astype(np.int32)
        bbox_xyxy = np.array([
            max(0, center[0] - half_edge),
            max(0, center[1] - half_edge),
            min(width - 1, center[0] + half_edge),
            min(height - 1, center[0] + half_edge),
        ]).astype(np.int32)
        area = (bbox_xyxy[2] - bbox_xyxy[0]) * \
            (bbox_xyxy[3] - bbox_xyxy[1])
        if area > 0:
            self.last_offset = bbox_xyxy[:2]
            ret_img = img_arr[bbox_xyxy[1]:bbox_xyxy[3],
                              bbox_xyxy[0]:bbox_xyxy[2], :]
        else:
            self.last_offset = np.zeros_like(bbox_xyxy[:2])
            ret_img = np.zeros_like(img_arr[:128, :128, :])
        # if there's a writer to be constructed
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
        # write images to video and destroy it at last
        if self.video_writer is not None:
            if area > 0:
                canvas = img_arr.copy()
                cv2.rectangle(
                    img=canvas,
                    pt1=bbox_xyxy_origin[:2],
                    pt2=bbox_xyxy_origin[2:4],
                    color=(0, 255, 0),
                    thickness=4,
                )
                cv2.rectangle(
                    img=canvas,
                    pt1=bbox_xyxy[:2],
                    pt2=bbox_xyxy[2:4],
                    color=(0, 0, 255),
                    thickness=2,
                )
                canvas = cv2.resize(canvas, self.video_writer_resolution)
            else:
                canvas = np.zeros(
                    shape=(self.video_writer_resolution[1],
                           self.video_writer_resolution[0], 3),
                    dtype=np.uint8)
            write_success = try_to_write_frame(
                self.video_writer, img_arr=canvas)
            if not write_success:
                self.video_writer = None
                self.logger.info('Video has been written to ' +
                                 f'{self.video_path}.')
        return ret_img

    def get_points(self, points2d, **kwargs):
        if self.last_offset is None:
            self.logger.error('No offset has been recorded. ' +
                              'Please call boost.get_image() ' +
                              'first before getting points.')
        offset = np.expand_dims(
            self.last_offset, axis=(0, 1, 2)).repeat(points2d.shape[2], 2)
        ret_points2d = points2d.copy()
        ret_points2d[..., :2] += offset
        return ret_points2d
