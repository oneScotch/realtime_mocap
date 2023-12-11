import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation as scipy_Rotation
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

# yapf: enable
try:
    import pyk4a
    has_pyk4a = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_pyk4a = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception

from .base_source import BaseSource


def get_xrprimer_fisheye_param(cam_params):
    K_rgb_distorted = np.eye(3, dtype=np.float32)
    for [i, j, k] in [[0, 0, 'fx'], [1, 1, 'fy'], [0, 2, 'cx'], [1, 2, 'cy']]:
        K_rgb_distorted[i, j] = cam_params['rgb_intrinsics'][k]

    rgb_distortion = np.array([
        cam_params['rgb_intrinsics'][k]
        for k in ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
    ],
                              dtype=np.float32)

    rvec = np.array([cam_params['depth_to_rgb']['r'][i] for i in range(3)],
                    dtype=np.float32)

    tvec = np.array([cam_params['depth_to_rgb']['t'][i] for i in range(3)],
                    dtype=np.float32) / 1000
    rot_mat = scipy_Rotation.from_rotvec(rvec).as_matrix()
    translation = tvec
    k33 = K_rgb_distorted
    h = cam_params['color_resolution']['h']
    w = cam_params['color_resolution']['w']
    dist_ks = [
        rgb_distortion[0],
        rgb_distortion[1],
        rgb_distortion[4],
        rgb_distortion[5],
        rgb_distortion[6],
        rgb_distortion[7],
    ]
    dist_ps = [
        rgb_distortion[2],
        rgb_distortion[3],
    ]
    fisheye_param = FisheyeCameraParameter(
        K=k33,
        R=rot_mat,
        T=translation,
        name='k4a_fisheye_param',
        height=h,
        width=w,
        world2cam=True,
        convention='opencv',
        dist_coeff_k=dist_ks,
        dist_coeff_p=dist_ps)
    return fisheye_param


class K4AStreamSource(BaseSource):

    def __init__(self,
                 necessary_keys=['pose', 'color'],
                 undistort_img: bool = True,
                 k4abt_optimizer: Union[dict, None] = None,
                 video_writer: Union[dict, None] = None,
                 fps: int = 30,
                 skip_old_atol_ms: Union[float, None] = None,
                 get_color_timestamp: bool = True,
                 parallel_bt: bool = True,
                 log_level: int = 0,
                 gpu_id: int = 0,
                 logger=None) -> None:
        BaseSource.__init__(self=self, logger=logger)
        if not has_pyk4a:
            self.logger.error(import_exception)
            raise ImportError
        self.video_writer_cfg = video_writer
        self.video_writer = None

        self.necessary_keys = necessary_keys

        self.k4a = None
        self.undistort_img = undistort_img
        self.fps = fps
        self.skip_old_atol_ms = skip_old_atol_ms
        self.get_color_timestamp = get_color_timestamp
        self.parallel_bt = parallel_bt
        self.log_level = log_level
        self.gpu_id = gpu_id
        self.init_k4a()
        if self.undistort_img:
            map1, map2 = get_undistort_maps(self.fisheye_param)
            self.undist_map1 = map1
            self.undist_map2 = map2
        if k4abt_optimizer is None:
            self.k4abt_optimizer = k4abt_optimizer
        else:
            self.k4abt_optimizer = build_optimizer(k4abt_optimizer)

    def init_k4a(self):
        if self.k4a is None:
            if self.fps == 30:
                fps_config = pyk4a.FPS.FPS_30
            elif self.fps == 15:
                fps_config = pyk4a.FPS.FPS_15
            else:
                raise Exception(f'fps {self.fps} not found')
            k4a = pyk4a.PyK4A(
                pyk4a.Config(
                    color_resolution=pyk4a.ColorResolution.RES_1536P,
                    depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                    camera_fps=fps_config,
                    gpu_id=self.gpu_id))
            k4a.connect(lut=True)
            self.fisheye_param = get_xrprimer_fisheye_param(
                k4a.get_cam_params())
            self.k4a = k4a

    def get_data(self, **kwargs):
        k4a_data = self.k4a.get_capture2(
            verbose=self.fps if self.log_level >= 2 else 0,
            skip_old_atol_ms=self.skip_old_atol_ms,
            get_color_timestamp=self.get_color_timestamp,
            get_color=True,
            undistort_color=False,
            get_depth=False,
            get_bt=True,
            undistort_bt=False,
            parallel_bt=self.parallel_bt)
        for necessary_key in self.necessary_keys:
            if necessary_key not in k4a_data:
                return None
        img_arr = k4a_data['color'][..., :3]

        if len(k4a_data['pose']) < 1:
            return None

        if self.undistort_img:
            img_arr = cv2.remap(
                img_arr,
                self.undist_map1,
                self.undist_map2,
                interpolation=cv2.INTER_NEAREST)
        if self.k4abt_optimizer is not None:
            k4a_pose = k4a_data['pose']
            kps3d = (k4a_pose[0, :, 2:5] / 1000).astype(np.float32)
            kps3d = self.k4abt_optimizer.forward(kps3d)['kps3d']
            k4a_data['pose'][0, :, 2:5] = kps3d * 1000

        time_stamp = time.time()
        data_dict = {
            'img_arr': img_arr,
            'timestamp': time_stamp,
            'k4a_pose': k4a_data['pose']
        }
        if self.video_writer_cfg is not None \
                and self.video_writer is None:
            self.video_writer_cfg['logger'] = self.logger
            self.video_path = self.video_writer_cfg['output_path']
            self.video_writer = build_video_writer(self.video_writer_cfg)
            self.video_writer_resolution = [
                self.video_writer_cfg['resolution'][1],
                self.video_writer_cfg['resolution'][0]
            ]
            self.video_writer_cfg = None
        # write images to video and destroy it at last
        if self.video_writer is not None and data_dict is not None:
            canvas = data_dict['img_arr'].copy()
            canvas = cv2.resize(canvas, self.video_writer_resolution)
            write_success = try_to_write_frame(
                self.video_writer, img_arr=canvas)
            if not write_success:
                self.video_writer = None
                self.logger.info('Video has been written to ' +
                                 f'{self.video_path}.')
        return data_dict

    def __del__(self):
        if self.k4a is not None:
            self.k4a.disconnect()
