# yapf: disable
import cv2
import numpy as np
from typing import Union
from xrmocap.utils.geometry import compute_iou
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.ops.projection.builder import OpencvProjector
from xrprimer.utils.log_utils import get_logger

from realtime_mocap.utils.ffmpeg_utils import (
    build_video_writer, try_to_write_frame,
)

# yapf: enable


class HandIntagCropper:
    K4ABT_HANDS = ((np.array((14, 15, 16, 17)), 'right'), (np.array(
        (7, 8, 9, 10)), 'left'))

    def __init__(self,
                 default_left_hand_path: str,
                 default_right_hand_path: str,
                 target_size: float = 256,
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
        default_left_hand = cv2.resize(
            default_left_hand,
            (int(self.target_size / 2), int(self.target_size / 2)))
        default_right_hand = cv2.imread(default_right_hand_path)
        default_right_hand = cv2.resize(
            default_right_hand,
            (int(self.target_size / 2), int(self.target_size / 2)))
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
            hand_cropped_imgs[side] = shand_img
        cropped_img = crop_intaghand_input_img(
            hand_bbox_xywh=bbox_xywh_dict,
            hand_cropped_imgs=hand_cropped_imgs,
            src_img=img_arr)
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
            img = cv2.resize(cropped_img, self.video_writer_resolution)
            write_success = try_to_write_frame(self.video_writer, img_arr=img)
            if not write_success:
                self.video_writer = None
                self.logger.info('Video has been written to ' +
                                 f'{self.video_path}.')
        return cropped_img, bbox_xywh_dict


def get_square_bbox(img_w, img_h, lu, rb, scale=1.0):

    max_half_edge = max(abs(rb[0] - lu[0]), abs(rb[1] - lu[1])) / 2.0 * scale
    middle = [(lu[0] + rb[0]) / 2, (lu[1] + rb[1]) / 2]
    lu = [middle[0] - max_half_edge, middle[1] - max_half_edge]
    rb = [middle[0] + max_half_edge, middle[1] + max_half_edge]
    lu = np.clip(lu, a_min=(0, 0), a_max=[img_w, img_h]).astype(int)
    rb = np.clip(rb, a_min=(0, 0), a_max=[img_w, img_h]).astype(int)
    return lu, rb


def crop_intaghand_input_img(hand_bbox_xywh: dict, hand_cropped_imgs: dict,
                             src_img: np.ndarray):
    if hand_bbox_xywh['left'] is not None and\
            hand_bbox_xywh['right'] is not None:
        iou = compute_iou(
            hand_bbox_xywh['right'],
            hand_bbox_xywh['left'],
            bbox_convention='xywh')
    else:
        iou = 0
    w = src_img.shape[1]
    h = src_img.shape[0]
    height_left = hand_cropped_imgs['left'].shape[0]
    height_right = hand_cropped_imgs['right'].shape[0]
    width_left = hand_cropped_imgs['left'].shape[1]
    width_right = hand_cropped_imgs['right'].shape[1]
    if iou > 0:
        lu = (min(hand_bbox_xywh['right'][0], hand_bbox_xywh['left'][0]),
              min(hand_bbox_xywh['right'][1], hand_bbox_xywh['left'][1]))
        rb = (max(hand_bbox_xywh['right'][0] + width_right,
                  hand_bbox_xywh['left'][0] + width_left),
              max(hand_bbox_xywh['right'][1] + height_right,
                  hand_bbox_xywh['left'][1] + height_left))
        lu, rb = get_square_bbox(w, h, lu, rb, scale=0.8)
        hand_bbox_xywh['all'] = np.array(
            [lu[0], lu[1], rb[0] - lu[0], rb[1] - lu[1]])
        cropped_img = src_img[lu[1]:rb[1], lu[0]:rb[0], :3]
    else:
        wide_sum = width_left + width_right
        bbox_edge = max(wide_sum, height_left, height_right)
        cropped_img = np.full((bbox_edge, bbox_edge, 3), 100, dtype=np.uint8)
        point_y = (bbox_edge - max(height_left, height_right)) // 2
        cropped_img[point_y:height_right + point_y,
                    0:width_right] = hand_cropped_imgs['right']
        cropped_img[point_y:height_left + point_y,
                    -width_left:] = hand_cropped_imgs['left']
    return cropped_img
