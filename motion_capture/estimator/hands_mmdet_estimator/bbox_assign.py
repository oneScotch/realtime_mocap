# yapf: disable
import numpy as np
from mmcv.utils import Registry
from typing import Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.ops.projection.builder import OpencvProjector
from xrprimer.utils.log_utils import get_logger

# yapf: enable

BBOX_ASSIGN = Registry('bbox_assign')


class CenterDistBboxAssign:
    K4ABT_HAND_INDEXES = dict(left=8, right=15)
    K4ABT_ARMS_INDEXES = np.array((5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17))
    K4ABT_BODY_INDEXES = np.array((3, 18, 22))
    K4ABT_UPPER_ARM_INDEXES = dict(
        left=np.array((6, 7)),
        right=np.array((13, 14)),
    )
    K4ABT_HANDS = ((np.array((14, 15, 16, 17)), 'right'), (np.array(
        (7, 8, 9, 10)), 'left'))

    def __init__(self,
                 cam_param_path: str,
                 body_height_threshold: Union[float, None] = 0.8,
                 arm_area_threshold: Union[float, None] = 1.28,
                 together_aspect_ratio_threshold: Union[float, None] = 3.0,
                 single_aspect_ratio_threshold: Union[float, None] = 2.5,
                 single_dot_product_threshold: Union[float, None] = 0.95,
                 logger=None) -> None:
        self.logger = get_logger(logger)
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
        self.body_height_threshold = body_height_threshold
        self.arm_area_threshold = arm_area_threshold
        self.together_aspect_ratio_threshold = together_aspect_ratio_threshold
        self.single_aspect_ratio_threshold = single_aspect_ratio_threshold
        self.single_dot_product_threshold = single_dot_product_threshold
        self.tracking_hand_centers = dict(right=None, left=None)

    def assign_bboxes(self, hands_bboxes, k4a_pose, img_width, img_height):
        hands_bboxes = self.filter_bboxes(hands_bboxes, k4a_pose)
        if len(hands_bboxes) <= 0:
            return dict(together=None, left=None, right=None)
        kps3d = (k4a_pose[0, :, 2:5] / 1000).astype(np.float32)
        kps2d = self.projector.project(points=kps3d)[0]
        hand_kps2d = dict()
        for side, index in self.__class__.K4ABT_HAND_INDEXES.items():
            hand_kps2d[side] = kps2d[index]
        ret_bbox_dict = dict()
        # if both hands are in the same bbox
        min_area = 9999999
        min_idx = -1
        together_bbox = None
        for idx, bbox_xyxy in enumerate(hands_bboxes):
            if _point_in_bbox_xyxy(
                    hand_kps2d['left'],
                    bbox_xyxy) and \
                _point_in_bbox_xyxy(
                    hand_kps2d['right'],
                    bbox_xyxy):
                area = (bbox_xyxy[2] - bbox_xyxy[0]) * \
                    (bbox_xyxy[3] - bbox_xyxy[1])
                if area < min_area:
                    min_idx = idx
                    min_area = area
        if min_idx >= 0:
            bbox_xyxy = hands_bboxes[min_idx]
            together_bbox = bbox_xyxy
            # if use together_aspect_ratio_threshold
            if together_bbox is not None and \
                    self.together_aspect_ratio_threshold is not None:
                aspect_ratio = _get_aspect_ratio(bbox_xyxy)
                if aspect_ratio > self.together_aspect_ratio_threshold:
                    together_bbox = None

        # there are 2 bboxes, and each contains only one hand
        candidates = (('left', 'right'), ('right', 'left'))
        min_cost = 9999.0
        min_cost_idx = 0
        for candidate_idx, candidate_assign in enumerate(candidates):
            cost = 0.0
            for side_idx, side in enumerate(candidate_assign):
                if side_idx < len(hands_bboxes):
                    kp2d = hand_kps2d[side]
                    bbox_xyxy = hands_bboxes[side_idx]
                    bbox_center = (bbox_xyxy[:2] + bbox_xyxy[2:4]) * 0.5
                    cost += np.linalg.norm(bbox_center - kp2d, ord=2)
            if cost < min_cost:
                min_cost = cost
                min_cost_idx = candidate_idx
        assign_result = candidates[min_cost_idx]
        for side_idx, side in enumerate(assign_result):
            if side_idx < len(hands_bboxes):
                # if use single_aspect_ratio
                bbox_xyxy = hands_bboxes[side_idx]
                if self.single_aspect_ratio_threshold is not None:
                    aspect_ratio = _get_aspect_ratio(bbox_xyxy)
                    bbox_vec = _get_vector_from_bbox(bbox_xyxy)
                    upper_arm_vec = self.get_upper_arm_vector(k4a_pose, side)
                    dot_prod = np.abs(np.dot(bbox_vec, upper_arm_vec))
                    if aspect_ratio > self.single_aspect_ratio_threshold\
                            and \
                            dot_prod > self.single_dot_product_threshold:
                        bbox_xyxy = None
            else:
                bbox_xyxy = None
            ret_bbox_dict[side] = bbox_xyxy

        # filter dramatic change of hands
        for side in ['right', 'left']:
            if ret_bbox_dict[side] is not None:
                hand_center = (ret_bbox_dict[side][:2] +
                               ret_bbox_dict[side][2:4]) * 0.5
                if self.tracking_hand_centers[side] is not None:
                    if np.linalg.norm(self.tracking_hand_centers[side] -
                                      hand_center) > 0.1 * img_width:
                        ret_bbox_dict[side] = None
                self.tracking_hand_centers[side] = hand_center
            else:
                self.tracking_hand_centers[side] = None

        ret_bbox_dict['together'] = None
        if ret_bbox_dict['right'] is not None and ret_bbox_dict[
                'left'] is not None:
            right_center = (ret_bbox_dict['right'][:2] +
                            ret_bbox_dict['right'][2:4]) * 0.5
            left_center = (ret_bbox_dict['left'][:2] +
                           ret_bbox_dict['left'][2:4]) * 0.5
            if together_bbox is not None and\
                    np.linalg.norm((right_center -
                                    left_center)) < 0.08 * img_width:
                ret_bbox_dict['together'] = together_bbox
                ret_bbox_dict['right'] = None
                ret_bbox_dict['left'] = None
            else:
                pass  # too close, need to be blended
        elif together_bbox is not None:
            ret_bbox_dict['together'] = together_bbox
            ret_bbox_dict['right'] = None
            ret_bbox_dict['left'] = None
        return ret_bbox_dict

    def get_arms_area(self, k4a_pose):
        hand_indexes = self.__class__.K4ABT_ARMS_INDEXES
        kps3d = (k4a_pose[0, :, 2:5] / 1000).astype(np.float32)[hand_indexes]
        kps2d = self.projector.project(points=kps3d)[0]
        # this bbox may go out of boundary
        bbox_xyxy = np.array([
            np.min(kps2d[:, 0]),
            np.min(kps2d[:, 1]),
            np.max(kps2d[:, 0]),
            np.max(kps2d[:, 1]),
        ])
        bbox_width = (bbox_xyxy[2] - bbox_xyxy[0])
        bbox_height = (bbox_xyxy[3] - bbox_xyxy[1])
        edge = max(bbox_width, bbox_height)
        arms_area = edge * edge
        return arms_area

    def get_upper_body_height(self, k4a_pose):
        body_indexes = self.__class__.K4ABT_BODY_INDEXES
        kps3d = (k4a_pose[0, :, 2:5] / 1000).astype(np.float32)[body_indexes]
        kps2d = self.projector.project(points=kps3d)[0]
        # this bbox may go out of boundary
        bbox_xyxy = np.array([
            np.min(kps2d[:, 0]),
            np.min(kps2d[:, 1]),
            np.max(kps2d[:, 0]),
            np.max(kps2d[:, 1]),
        ])
        bbox_height = (bbox_xyxy[3] - bbox_xyxy[1])
        return bbox_height

    def filter_bboxes(self, hands_bboxes, k4a_pose):
        ret_list = []
        arms_area = self.get_arms_area(k4a_pose)
        upper_body_height = self.get_upper_body_height(k4a_pose)
        for bbox_xyxy in hands_bboxes:
            bbox_qualified = True
            bbox_height = bbox_xyxy[3] - bbox_xyxy[1]
            bbox_area = (bbox_xyxy[2] - bbox_xyxy[0]) * \
                (bbox_xyxy[3] - bbox_xyxy[1])
            # if use height_threshold
            if self.body_height_threshold is not None:
                if bbox_height / upper_body_height > \
                        self.body_height_threshold:
                    bbox_qualified = False
            # if use arm_area_threshold
            if bbox_qualified and \
                    self.arm_area_threshold is not None:
                if bbox_area / arms_area > self.arm_area_threshold:
                    bbox_qualified = False
            if bbox_qualified:
                ret_list.append(bbox_xyxy)
        return ret_list

    def get_upper_arm_vector(self, k4a_pose, side):
        arm_indexes = self.__class__.K4ABT_UPPER_ARM_INDEXES[side]
        kps3d = (k4a_pose[0, :, 2:5] / 1000).astype(np.float32)[arm_indexes]
        kps2d = self.projector.project(points=kps3d)[0]
        unnormed_vec = kps2d[1] - kps2d[0]
        norm = np.linalg.norm(unnormed_vec, ord=2)
        normed_vec = unnormed_vec / norm
        return normed_vec


def _get_aspect_ratio(bbox_xyxy):
    aspect_ratio = max(
        abs((bbox_xyxy[2] - bbox_xyxy[0]) / (bbox_xyxy[3] - bbox_xyxy[1])),
        abs((bbox_xyxy[3] - bbox_xyxy[1]) / (bbox_xyxy[2] - bbox_xyxy[0])))
    return aspect_ratio


def _get_vector_from_bbox(bbox_xyxy):
    width = bbox_xyxy[2] - bbox_xyxy[0]
    height = bbox_xyxy[3] - bbox_xyxy[1]
    if width > height:
        return np.array((1, 0))
    else:
        return np.array((0, 1))


BBOX_ASSIGN.register_module(
    name='CenterDistBboxAssign', module=CenterDistBboxAssign)


def build_bbox_assign(cfg) -> CenterDistBboxAssign:
    """Build bbox_assign."""
    return BBOX_ASSIGN.build(cfg)


def _point_in_bbox_xyxy(point2d, bbox_xyxy):
    top_left_vec = point2d - bbox_xyxy[:2]
    bottom_right_vec = bbox_xyxy[2:4] - point2d
    flat_values = np.array((top_left_vec, bottom_right_vec)).reshape(-1)
    for v in flat_values:
        if v < 0:
            return False
    return True


def get_square_bbox(img_w, img_h, lu, rb, scale=1.0):

    max_half_edge = max(abs(rb[0] - lu[0]), abs(rb[1] - lu[1])) / 2.0 * scale
    middle = [(lu[0] + rb[0]) / 2, (lu[1] + rb[1]) / 2]
    lu = [middle[0] - max_half_edge, middle[1] - max_half_edge]
    rb = [middle[0] + max_half_edge, middle[1] + max_half_edge]
    lu = np.clip(lu, a_min=(0, 0), a_max=[img_w, img_h]).astype(int)
    rb = np.clip(rb, a_min=(0, 0), a_max=[img_w, img_h]).astype(int)
    return lu, rb
