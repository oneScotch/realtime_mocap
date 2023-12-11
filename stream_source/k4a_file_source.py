# yapf: disable
import cv2
import numpy as np
import time
from typing import Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.ops.projection.builder import OpencvProjector

from realtime_mocap.motion_capture.optimizer.builder import build_optimizer
from realtime_mocap.utils.ffmpeg_utils import (
    build_video_writer, try_to_write_frame,
)
from realtime_mocap.utils.img_utils import get_undistort_maps
from .base_source import BaseSource

# yapf: enable


class K4AFileSource(BaseSource):
    BASE_TIME = 0.032

    def __init__(self,
                 k4a_file_path: str,
                 cam_param_path: Union[None, str] = None,
                 undistort_img: bool = True,
                 k4abt_optimizer: Union[dict, None] = None,
                 video_writer: Union[dict, None] = None,
                 logger=None) -> None:
        BaseSource.__init__(self=self, logger=logger)
        npz_file = np.load(k4a_file_path, allow_pickle=True)
        self.mframe_dict = dict(npz_file)
        self.frame_idx = 0
        self.prev_time = None
        self.video_writer_cfg = video_writer
        self.video_writer = None
        try:
            self.cam_param = PinholeCameraParameter.fromfile(cam_param_path)
        except ValueError:
            self.logger.warning('Log error [File content is not correct.] ' +
                                'is for loading PinholeCameraParameter. ' +
                                'If you see [FisheyeCameraParameter ' +
                                'has been loaded.], ' +
                                'this msg can be ignored.')
            self.cam_param = FisheyeCameraParameter.fromfile(cam_param_path)
            self.logger.info('FisheyeCameraParameter has been loaded.')
        self.projector = OpencvProjector(
            camera_parameters=[
                self.cam_param,
            ], logger=self.logger)
        if undistort_img is True and \
                (self.cam_param is None or
                 not isinstance(
                    self.cam_param, FisheyeCameraParameter)):
            self.logger.warning('Cannot undistort images' +
                                ' without cam_param_path,' +
                                ' reset undistort_img to False.')
            undistort_img = False
        self.undistort_img = undistort_img
        if self.undistort_img:
            map1, map2 = get_undistort_maps(self.fisheye_param)
            self.undist_map1 = map1
            self.undist_map2 = map2

        if k4abt_optimizer is None:
            self.k4abt_optimizer = k4abt_optimizer
        else:
            self.k4abt_optimizer = build_optimizer(k4abt_optimizer)

    def get_data(self, **kwargs):
        # might be None
        data_dict = self.mframe_dict[str(self.frame_idx)].item()
        # ['img_arr', 'timestamp', 'k4a_pose', 'frame_idx']
        # data_dict['img_arr'].shape: (1536, 2048, 3)
        # data_dict['k4a_pose'].shape: (1, 32, 10)
        # type(data_dict['timestamp']): float
        if self.undistort_img and data_dict is not None:
            img_arr = cv2.remap(
                data_dict['img_arr'],
                self.undist_map1,
                self.undist_map2,
                interpolation=cv2.INTER_NEAREST)
            data_dict = data_dict.copy()
            data_dict['img_arr'] = img_arr
        self.frame_idx = (self.frame_idx + 1) % len(self.mframe_dict)
        if self.frame_idx >= len(self.mframe_dict):
            self.logger.warning('oops')
        if self.k4abt_optimizer is not None and data_dict is not None:
            kps3d = (data_dict['k4a_pose'][0, :, 2:5] / 1000).astype(
                np.float32)
            kps3d = self.k4abt_optimizer.forward(kps3d)['kps3d']
            data_dict['k4a_pose'][0, :, 2:5] = kps3d * 1000
        # if there's a writer to be constructed
        if self.video_writer_cfg is not None \
                and self.video_writer is None:
            self.video_writer_cfg['logger'] = self.logger
            self.video_path = self.video_writer_cfg['output_path']
            self.video_writer = build_video_writer(self.video_writer_cfg)
            self.video_writer_cfg = None
        # write images to video and destroy it at last
        if self.video_writer is not None and data_dict is not None:
            canvas = data_dict['img_arr'].copy()
            kps3d = (data_dict['k4a_pose'][0, :, 2:5] / 1000).astype(
                np.float32)
            kps2d = self.projector.project(points=kps3d)[0]
            for kp in kps2d:
                cv2.circle(
                    canvas, (int(kp[0]), int(kp[1])),
                    radius=10,
                    color=(0, 0, 255),
                    thickness=-1)
            canvas = cv2.resize(canvas, self.video_writer_cfg['resolution'])
            write_success = try_to_write_frame(
                self.video_writer, img_arr=canvas)
            if not write_success:
                self.video_writer = None
                self.logger.info('Video has been written to ' +
                                 f'{self.video_path}.')
        if self.prev_time is None:
            self.prev_time = time.time()
        else:
            time_to_wait = self.__class__.BASE_TIME + \
                self.prev_time - time.time()
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            self.prev_time = time.time()
        if data_dict is not None:
            data_dict['timestamp'] = time.time()
        return data_dict
