# yapf: disable
import mmcv
import numpy as np
import time
import torch
from xrmocap.data_structure.body_model import SMPLXData
from xrprimer.data_structure.camera.fisheye_camera import (
    PinholeCameraParameter,
)
from xrprimer.transform.image.color import bgr2rgb

from .base_optimizer import BaseOptimizer

try:
    import mediapipe as mp
    from mmhuman3d.core.conventions.cameras.convert_convention import (
        convert_camera_matrix,
    )
    from mmhuman3d.core.conventions.keypoints_mapping import convert_kps
    from mmhuman3d.data.data_structures.human_data import HumanData
    from mmhuman3d.models.registrants.builder import build_registrant
    has_smplifyx = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_smplifyx = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception

# yapf: enable


class SMPLifyxOptimizer(BaseOptimizer):

    def __init__(self,
                 pinhole_param_path,
                 smplify_config_path,
                 use_tracking: bool = True,
                 profile_period: float = 10,
                 verbose: bool = True,
                 device: str = 'cuda',
                 logger=None) -> None:
        BaseOptimizer.__init__(self, logger=logger)
        if not has_smplifyx:
            self.logger.error(import_exception)
            raise ImportError
        self.mediapipe_estimator = MediaPipeEstimator()
        self.smplify_processor = SmplifyProcessor(
            pinhole_param_path=pinhole_param_path,
            smplify_config_path=smplify_config_path,
            device=device)

        self.profile_period = profile_period
        self.verbose = verbose
        self.forward_count = 0
        self.last_profile_time = time.time()
        self.mediapipe_time_sum = 0.0
        self.smplify_time_sum = 0.0
        self.last_smplify_output = None
        self.use_tracking = use_tracking

    def forward(self, img_arr, smplx_data=None, return_kps2d=False, **kwargs):
        with torch.no_grad():
            mediapipe_start_time = time.time()
            human_data_2d = self.mediapipe_estimator.process(img_arr)
            mediapipe_end_time = time.time()
            self.mediapipe_time_sum += \
                mediapipe_end_time - mediapipe_start_time
            if self.last_smplify_output is None:
                kwargs = dict(initial_data=smplx_data, anchor_pose=None)
            elif self.use_tracking:
                kwargs = dict(
                    initial_data=self.last_smplify_output,
                    anchor_pose=smplx_data)
            else:
                kwargs = dict(initial_data=smplx_data)
            smplify_output = self.smplify_processor.fit(
                human_data_2d, **kwargs)
            self.last_smplify_output = smplify_output
            smplify_end_time = time.time()
            self.smplify_time_sum += smplify_end_time - mediapipe_end_time
        ret_smplx_data = SMPLXData()
        ret_smplx_data.from_param_dict(smplify_output)
        ret_dict = dict(img_arr=img_arr, smplx_data=ret_smplx_data)
        time_diff = time.time() - self.last_profile_time
        self.forward_count += 1
        if time_diff >= self.profile_period:
            if self.verbose:
                self.logger.info(
                    '\n' + 'mediapipe_time:' +
                    f' {self.mediapipe_time_sum/self.forward_count}\n' +
                    'smplify_time:' +
                    f' {self.smplify_time_sum/self.forward_count}\n' + '\n')
            self.forward_count = 0
            self.last_profile_time = time.time()
            self.mediapipe_time_sum = 0.0
            self.smplify_time_sum = 0.0
        if return_kps2d:
            return ret_dict, human_data_2d
        else:
            return ret_dict


class MediaPipeEstimator():

    def __init__(self,
                 hand_model_complexity=1,
                 hand_max_num_hands=2,
                 hand_min_detection_confidence=0.5,
                 hand_min_tracking_confidence=0.5,
                 body_model_complexity=1,
                 body_min_detection_confidence=0.7,
                 body_min_tracking_confidence=0.5):
        self.hands_mp = mp.solutions.hands.Hands(
            model_complexity=hand_model_complexity,
            max_num_hands=hand_max_num_hands,
            min_detection_confidence=hand_min_detection_confidence,
            min_tracking_confidence=hand_min_tracking_confidence,
        )
        self.body_mp = mp.solutions.pose.Pose(
            # upper_body_only=upper_body_only,
            model_complexity=body_model_complexity,
            enable_segmentation=False,
            min_detection_confidence=body_min_detection_confidence,
            min_tracking_confidence=body_min_tracking_confidence,
        )

    def process(self, img):
        img = bgr2rgb(img.copy())
        hands_result_mp = self.hands_mp.process(img)
        body_result_mp = self.body_mp.process(img)

        body_pose = None
        if body_result_mp.pose_landmarks:
            kps_list = [[landmark.x, landmark.y, landmark.visibility]
                        for landmark in body_result_mp.pose_landmarks.landmark]
            body_pose = np.expand_dims(np.array(kps_list), 0)
        else:
            body_pose = np.zeros((1, 33, 3))

        two_hands = {'right': None, 'left': None}
        hands_flag = []
        if hands_result_mp.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(
                    hands_result_mp.multi_hand_landmarks):
                kps_list = [[landmark.x, landmark.y, 1]
                            for landmark in hand_landmarks.landmark]
                left_distance = np.linalg.norm(
                    np.array(kps_list[0][:2]) - body_pose[0, 15, :2])
                right_distance = np.linalg.norm(
                    np.array(kps_list[0][:2]) - body_pose[0, 16, :2])
                handedness = 'right' if left_distance \
                             < right_distance else 'left'
                hands_flag.append(handedness)
                two_hands[handedness] = np.expand_dims(np.array(kps_list), 0)
        if 'right' not in hands_flag:
            two_hands['right'] = np.zeros((1, 21, 3))
        if 'left' not in hands_flag:
            two_hands['left'] = np.zeros((1, 21, 3))

        # remove hand and wrist from body
        body2whole_idx = [i for i in range(33) if i not in range(15, 23)]
        keypoints_src = np.concatenate((body_pose[:, body2whole_idx, :],
                                        two_hands['right'], two_hands['left']),
                                       axis=1)
        keypoints_src_mask = np.ones(shape=(67, ), dtype=np.uint8)
        keypoints, mask = convert_kps(
            keypoints_src,
            mask=keypoints_src_mask,
            src='mediapipe_whole_body',
            dst='smplx_wo_contour')

        human_data = HumanData()
        human_data['keypoints2d_mask'] = mask
        human_data['keypoints2d'] = keypoints
        human_data['keypoints2d_convention'] = 'smplx_wo_contour'

        return human_data


class SmplifyProcessor():

    def __init__(self,
                 pinhole_param_path,
                 smplify_config_path,
                 device='cuda') -> None:
        self.smplify_config = mmcv.Config.fromfile(smplify_config_path)
        self.device = torch.device(device)

        self.pinhole_param = PinholeCameraParameter.fromfile(
            pinhole_param_path)
        if self.pinhole_param.world2cam:
            self.pinhole_param.inverse_extrinsic()
        K = np.array(self.pinhole_param.get_intrinsic(3))
        R = np.array(self.pinhole_param.get_extrinsic_r())
        T = np.array(self.pinhole_param.get_extrinsic_t())
        R = torch.Tensor(R).view(-1, 3, 3)
        T = torch.Tensor(T).view(-1, 3)
        K = torch.Tensor(K).view(-1, K.shape[-2], K.shape[-1])
        is_perspective = True
        in_ndc = False
        convention = 'opencv'
        projection = 'perspective'
        self.render_resolution = (self.pinhole_param.height,
                                  self.pinhole_param.width)
        K, R, T = convert_camera_matrix(
            convention_dst='pytorch3d',
            K=K,
            R=R,
            T=T,
            is_perspective=is_perspective,
            convention_src=convention,
            resolution_src=self.render_resolution,
            in_ndc_src=in_ndc,
            in_ndc_dst=in_ndc)

        cameras_kinect_config = dict(
            type=projection,
            in_ndc=in_ndc,
            device=self.device,
            K=K,
            R=R,
            T=T,
            resolution=self.render_resolution)

        self.smplify_config['camera'] = cameras_kinect_config
        self.smplify_config['img_res'] = torch.tensor(
            self.render_resolution, device=self.device)

        # import pdb; pdb.set_trace()
        self.smplify = build_registrant(dict(self.smplify_config))

    def fit(self, human_data, initial_data=None, anchor_pose=None):
        mask = human_data['keypoints2d_mask']
        keypoints = human_data['keypoints2d']

        keypoints_conf = np.repeat(mask[None], keypoints.shape[0], axis=0)
        # mark invisible keypoints
        keypoints_conf = keypoints_conf * (keypoints[..., 2] > 0.3
                                           )  # 67 - 8 = 59
        keypoints_conf[:, 66:76] = keypoints_conf[:, 66:76] * 3  # 手指权重
        keypoints = keypoints[..., :2] * (keypoints[..., 2:] > 0.3)
        resolution = np.array(self.render_resolution)
        keypoints = keypoints * (resolution[::-1] - 1)

        keypoints = torch.tensor(
            keypoints, dtype=torch.float32, device=self.device)
        keypoints_conf = torch.tensor(
            keypoints_conf, dtype=torch.float32, device=self.device)

        # run SMPLify(X)
        if initial_data is not None:
            if isinstance(initial_data, SMPLXData):
                pose_dict = initial_data.to_tensor_dict(device=self.device)
            else:
                pose_dict = initial_data
            if anchor_pose is not None:
                anchor_pose = anchor_pose.to_tensor_dict(device=self.device)
            smplify_output = self.smplify(
                keypoints2d=keypoints,
                keypoints2d_conf=keypoints_conf,
                init_global_orient=pose_dict['global_orient'],
                init_transl=pose_dict['transl'],
                init_body_pose=pose_dict['body_pose'],
                init_betas=pose_dict['betas'],
                init_left_hand_pose=pose_dict['left_hand_pose'].reshape(1, -1),
                init_right_hand_pose=pose_dict['right_hand_pose'].reshape(
                    1, -1),
                anchor_pose=anchor_pose)
        else:
            smplify_output = self.smplify(
                keypoints2d=keypoints,
                keypoints2d_conf=keypoints_conf,
            )

        return smplify_output
