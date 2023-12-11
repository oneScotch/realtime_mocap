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


class HandFrankCropper:
    K4ABT_HANDS = ((np.array((14, 15, 16, 17)), 'right'), (np.array(
        (7, 8, 9, 10)), 'left'))

    def __init__(self,
                 default_left_hand_path: str,
                 default_right_hand_path: str,
                 target_size: float = 224,
                 square_scale: float = 1.0,
                 cam_param_path: Union[None, str] = None,
                 use_mmdet: bool = True,
                 use_mediapipe: bool = True,
                 use_k4abt: bool = True,
                 video_writer: Union[dict, None] = None,
                 logger=None) -> None:
        self.logger = get_logger(logger)
        self.square_scale = square_scale

        self.k4abt_warned = False
        self.use_k4abt = use_k4abt
        self.projector = None
        if use_k4abt:
            try:
                cam_param = PinholeCameraParameter.fromfile(cam_param_path)
            except ValueError:
                self.logger.warning(
                    'Log error [File content is not correct.] ' +
                    'is for loading PinholeCameraParameter. ' +
                    'If you see [FisheyeCameraParameter ' +
                    'has been loaded.], ' + 'this msg can be ignored.')
                cam_param = FisheyeCameraParameter.fromfile(cam_param_path)
                self.logger.info('FisheyeCameraParameter has been loaded.')
            self.projector = OpencvProjector(
                camera_parameters=[
                    cam_param,
                ], logger=self.logger)

        self.mediapipe_warned = False
        self.use_mediapipe = use_mediapipe

        self.use_mmdet = use_mmdet

        self.target_size = target_size
        default_left_hand = cv2.imread(default_left_hand_path)
        default_left_hand = cv2.resize(default_left_hand,
                                       (self.target_size, self.target_size))
        default_right_hand = cv2.imread(default_right_hand_path)
        default_right_hand = cv2.resize(default_right_hand,
                                        (self.target_size, self.target_size))
        self.default_hands = dict(
            left=default_left_hand, right=default_right_hand)

        self.video_writer_cfg = video_writer
        self.video_writer = None

    def forward(
        self,
        img_arr: np.ndarray,
        k4a_pose: Union[None, np.ndarray] = None,
        mp_keypoints2d_dict: Union[None, dict] = None,
        hands_bboxes: Union[None, dict] = None,
    ):
        img_arr = img_arr[..., :3]
        w = img_arr.shape[1]
        h = img_arr.shape[0]
        bbox_xywh_dict = dict()
        if mp_keypoints2d_dict is not None:
            mp_keypoints2d_dict_new = dict()
            # mediapipe defines differently from k4abt
            mp_keypoints2d_dict_new['left'] = \
                mp_keypoints2d_dict['right']
            mp_keypoints2d_dict_new['right'] = \
                mp_keypoints2d_dict['left']
            mp_keypoints2d_dict = mp_keypoints2d_dict_new
        # get bbox_xywh
        # only mediapipe
        if self.use_mediapipe and \
                not self.use_k4abt and \
                not self.use_mmdet:
            for side in ('left', 'right'):
                keypoints = mp_keypoints2d_dict[side]
                if keypoints is None:
                    bbox_xywh_dict[side] = None
                else:
                    kps2d = keypoints['keypoints'][..., :2]
                    hand_kps2d = np.squeeze(kps2d, axis=(0, 1))
                    lu = np.min(hand_kps2d, axis=0)
                    rb = np.max(hand_kps2d, axis=0)
                    lu, rb = get_square_bbox(
                        w, h, lu, rb, scale=self.square_scale)
                    bbox_xywh = np.array(
                        [lu[0], lu[1], rb[0] - lu[0], rb[1] - lu[1]])
                    # too small, set to None
                    if bbox_xywh[-1] * bbox_xywh[-2] == 0:
                        bbox_xywh = None
                    bbox_xywh_dict[side] = bbox_xywh
        # only k4abt
        elif self.use_k4abt and \
                not self.use_mediapipe and \
                not self.use_mmdet:
            body_pose = k4a_pose[0, :, 2:5] / 1000
            # crop hands from image
            for hand_indexes, side in self.__class__.K4ABT_HANDS:
                hand_kps3d = body_pose[hand_indexes]
                # cube_kps2d.shape: (4, 3)
                hand_kps2d = self.projector.project(points=hand_kps3d)[0]
                lu = np.min(hand_kps2d, axis=0)
                rb = np.max(hand_kps2d, axis=0)
                lu, rb = get_square_bbox(w, h, lu, rb, scale=self.square_scale)
                bbox_xywh = np.array(
                    [lu[0], lu[1], rb[0] - lu[0], rb[1] - lu[1]])
                # too small, set to None
                if bbox_xywh[-1] * bbox_xywh[-2] == 0:
                    bbox_xywh = None
                bbox_xywh_dict[side] = bbox_xywh
        elif self.use_mmdet and \
                not self.use_k4abt and \
                not self.use_mediapipe:
            for side in ('together', 'left', 'right'):
                bbox_xyxy = hands_bboxes[side]
                if bbox_xyxy is None:
                    bbox_xywh = None
                else:
                    lu = bbox_xyxy[:2]
                    rb = bbox_xyxy[2:4]
                    aspect_ratio = max(
                        abs((bbox_xyxy[2] - bbox_xyxy[0]) /
                            (bbox_xyxy[3] - bbox_xyxy[1])),
                        abs((bbox_xyxy[3] - bbox_xyxy[1]) /
                            (bbox_xyxy[2] - bbox_xyxy[0])))
                    if aspect_ratio > 2.5:
                        lu, rb = get_square_bbox(w, h, lu, rb, scale=1)
                    else:
                        lu, rb = get_square_bbox(
                            w, h, lu, rb, scale=self.square_scale)
                    bbox_xywh = np.array(
                        [lu[0], lu[1], rb[0] - lu[0], rb[1] - lu[1]])
                if side == 'together':
                    if bbox_xyxy is not None:
                        bbox_xywh_dict['left'] = bbox_xywh
                        bbox_xywh_dict['right'] = bbox_xywh
                        break
                    else:
                        continue
                else:
                    bbox_xywh_dict[side] = bbox_xywh
        else:
            self.logger.error('Not a valid crop strategy!')
            raise RuntimeError
        # prepare single hand image
        hand_cropped_imgs = dict()
        for side, bbox_xywh in bbox_xywh_dict.items():
            if bbox_xywh is None:
                shand_img = self.default_hands[side].copy()
            else:
                x, y, w, h = bbox_xywh
                shand_img = img_arr[y:y + h, x:x + w, :3]
                shand_img = cv2.resize(shand_img,
                                       (self.target_size, self.target_size))
            hand_cropped_imgs[side] = shand_img
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
            half_wid = int(self.video_writer_resolution[0] * 0.5)
            img = np.ones(
                shape=(self.video_writer_resolution[1],
                       self.video_writer_resolution[0], 3),
                dtype=np.uint8)
            right_hand = cv2.resize(hand_cropped_imgs['right'], (
                half_wid,
                self.video_writer_resolution[1],
            ))
            left_hand = cv2.resize(hand_cropped_imgs['left'], (
                half_wid,
                self.video_writer_resolution[1],
            ))
            img[:, :half_wid, :] = right_hand
            img[:, half_wid:, :] = left_hand
            write_success = try_to_write_frame(self.video_writer, img_arr=img)
            if not write_success:
                self.video_writer = None
                self.logger.info('Video has been written to ' +
                                 f'{self.video_path}.')
        return hand_cropped_imgs, bbox_xywh_dict
