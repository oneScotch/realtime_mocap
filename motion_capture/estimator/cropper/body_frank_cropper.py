# yapf: disable
import cv2
import numpy as np
from typing import Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.ops.projection.builder import OpencvProjector
from xrprimer.utils.log_utils import get_logger

from realtime_mocap.utils.ffmpeg_utils import (
    build_video_writer, try_to_write_frame,
)
from .hand_intag_cropper import get_square_bbox

# yapf: enable


class BodyFrankCropper:

    def __init__(self,
                 target_size: float = 224,
                 square_scale: float = 1.0,
                 cam_param_path: Union[None, str] = None,
                 video_writer: Union[dict, None] = None,
                 logger=None) -> None:
        self.logger = get_logger(logger)
        self.square_scale = square_scale

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

        self.target_size = target_size
        self.video_writer_cfg = video_writer
        self.video_writer = None

    def forward(self,
                img_arr: np.ndarray,
                k4a_pose: Union[None, np.ndarray] = None,
                **kwargs):
        img_arr = img_arr[..., :3].copy()
        w = img_arr.shape[1]
        h = img_arr.shape[0]
        kps3d = k4a_pose[0, :, 2:5] / 1000
        kps2d = self.projector.project(points=kps3d)[0]
        lu = np.min(kps2d, axis=0)
        rb = np.max(kps2d, axis=0)
        lu, rb = get_square_bbox(w, h, lu, rb, scale=self.square_scale)
        bbox_xywh = np.array([lu[0], lu[1], rb[0] - lu[0], rb[1] - lu[1]])
        center = bbox_xywh[:2] + 0.5 * bbox_xywh[2:]
        max_edge = max(bbox_xywh[2:])
        # adjust bounding box tightness
        # TODO: why 200 and 1.2
        scale = max_edge / 200.0 * 1.2
        cropped_img = _crop_bbox(img_arr, center, scale,
                                 (self.target_size, self.target_size))
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
            img_to_write = cv2.resize(cropped_img,
                                      self.video_writer_resolution)
            write_success = try_to_write_frame(
                self.video_writer, img_arr=img_to_write)
            if not write_success:
                self.video_writer = None
                self.logger.info('Video has been written to ' +
                                 f'{self.video_path}.')
        return cropped_img


def _get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def _transform(pt, center, scale, res, invert=0, rot=0):
    """Transform pixel location to different reference."""
    t = _get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def _crop_bbox(img, center, scale, res=(224, 224)):
    """Crop image according to the supplied bounding box."""
    # Upper left point
    ul = np.array(_transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(
        _transform([res[0] + 1, res[1] + 1], center, scale, res, invert=1)) - 1

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    # new_img = np.zeros(new_shape)
    if new_shape[0] < 1 or new_shape[1] < 1:
        return None
    new_img = np.zeros(new_shape, dtype=np.uint8)

    if new_img.shape[0] == 0:
        return None

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])

    if new_y[0] < 0 or new_y[1] < 0 or new_x[0] < 0 or new_x[1] < 0:
        return None

    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                        old_x[0]:old_x[1]]

    if new_img.shape[0] < 20 or new_img.shape[1] < 20:
        return None
    new_img = cv2.resize(new_img, res)

    return new_img
