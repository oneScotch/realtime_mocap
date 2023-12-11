# yapf: disable
import numpy as np
import time
import torch
from manopth import manolayer
from mmhuman3d.core.conventions.keypoints_mapping.mano import MANO_REORDER_MAP
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.ops.projection.builder import OpencvProjector

from realtime_mocap.motion_capture.estimator.base_estimator import (
    BaseEstimator,
)
from realtime_mocap.motion_capture.optimizer.builder import build_optimizer
from .model.model import load_model  # noqa: E501
from .preprocess import (
    crop_intaghand_input_img, get_square_bbox, transform_img_intaghand,
)

# yapf: enable


class ManoJrWrapper():

    def __init__(self, J_regressor, device='cuda'):
        self.device = device
        self.process_J_regressor(J_regressor)

    def process_J_regressor(self, J_regressor):
        J_regressor = J_regressor.clone().detach()
        tip_regressor = torch.zeros_like(J_regressor[:5])
        tip_regressor[0, 745] = 1.0  # thumb_tip
        tip_regressor[1, 317] = 1.0  # index_tip
        tip_regressor[2, 444] = 1.0  # middle_tip
        tip_regressor[3, 556] = 1.0  # ring_tip
        tip_regressor[4, 673] = 1.0  # pinky_tip
        J_regressor = torch.cat([J_regressor, tip_regressor], dim=0)

        self.J_regressor = J_regressor[MANO_REORDER_MAP].contiguous()
        self.J_regressor = J_regressor.to(self.device)

    def __call__(self, v):
        return torch.matmul(self.J_regressor, v)


class HandsIntagEstimator(BaseEstimator):

    def __init__(self,
                 path,
                 model,
                 train,
                 cam_param_path,
                 profile_period: float = 10.0,
                 n_test: int = 1,
                 optimizer=None,
                 input_size=256,
                 mano_model_path='data/body_models/mano/',
                 device='cuda',
                 ret_crop: bool = False,
                 verbose: bool = True,
                 logger=None):
        super().__init__(logger=logger)
        self.verbose = verbose
        self.profile_period = profile_period
        self.n_test = n_test
        self.input_size = input_size
        self.device = device
        self.logger = logger
        self.left_optimizer = None
        self.right_optimizer = None
        self.ret_crop = ret_crop
        try:
            self.cam_param = PinholeCameraParameter.fromfile(cam_param_path)
        except ValueError:
            self.logger.warning(
                'Log msg [File content is not correct.] ' +
                'is for loading PinholeCameraParameter. ' +
                'If the program goes on, this msg can be ignored.')
            self.cam_param = FisheyeCameraParameter.fromfile(cam_param_path)

        self.projector = OpencvProjector(
            camera_parameters=[self.cam_param], logger=self.logger)

        self.left_optimizer = None
        self.right_optimizer = None
        if isinstance(optimizer, dict):
            optimizer['logger'] = self.logger
            if optimizer['type'] == 'AIKOptimizer':
                optimizer['mano_config']['side'] = 'right'
                self.right_optimizer = build_optimizer(optimizer)
                optimizer['mano_config']['side'] = 'left'
                self.left_optimizer = build_optimizer(optimizer)
            else:
                self.right_optimizer = build_optimizer(optimizer)

        self.model = load_model(model, train, path)
        state = torch.load(path['checkpoint_path'], map_location='cpu')
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(device)

        # Get J_regressor
        mano_layer = {
            'left':
            manolayer.ManoLayer(
                mano_root=mano_model_path,
                side='left',
                center_idx=None,
                flat_hand_mean=True,
                use_pca=False).to(device),
            'right':
            manolayer.ManoLayer(
                mano_root=mano_model_path,
                side='right',
                center_idx=None,
                flat_hand_mean=True,
                use_pca=False).to(device)
        }
        self.J_regressor = {
            'left': ManoJrWrapper(mano_layer['left'].th_J_regressor),
            'right': ManoJrWrapper(mano_layer['right'].th_J_regressor)
        }

    def estimate_vertice(self, img_tensor):
        """Estimate vertices from input images."""
        result = self.model(img_tensor)

        params = {}
        params['v3d_left'] = result['left']
        params['v3d_right'] = result['right']
        params['j3d_left'] = self.verts2joints(params['v3d_left'], 'left')
        params['j3d_right'] = self.verts2joints(params['v3d_right'], 'right')
        return params

    @torch.no_grad()
    def verts2joints(self, verts, side):
        """Get hand joints from predicted vertices."""
        joints_pred = self.J_regressor[side](verts)
        joints_pred = joints_pred[:, MANO_REORDER_MAP]

        return joints_pred

    @torch.no_grad()
    def joints2rotmat_iknet(self, hand_dict):
        results = dict()
        for side, kps3d in hand_dict.items():
            if side == 'left':
                wrist_j = kps3d[:, 0, 0]
                kps3d[:, :, 0] = -kps3d[:, :, 0]
                offset_j = wrist_j - kps3d[:, 0, 0]
                kps3d[:, :, 0] += offset_j
            rotmat = self.right_optimizer(kps3d, side)
            results[side] = rotmat
        return results

    @torch.no_grad()
    def joints2rotmat_aik(self, hand_dict):
        results = dict()
        for side, kps3d in hand_dict.items():
            for _ in range(self.n_test):
                smplx_data = dict()
                if side == 'left':
                    ret = self.left_optimizer(
                        smplx_data,
                        kps3d[0].cpu().numpy(),
                        do_eval=False,
                        output_tensor=False,
                        return_joints=False)
                else:
                    ret = self.right_optimizer(
                        smplx_data,
                        kps3d[0].cpu().numpy(),
                        do_eval=False,
                        output_tensor=False,
                        return_joints=False)
                hand_pose, hand_shape = ret[:2]
            results[side] = hand_pose
        return results

    def forward(self, img_arr, k4a_pose, **kwargs):
        result = dict()
        img = img_arr[..., :3]
        w = img.shape[1]
        h = img.shape[0]
        body_pose = k4a_pose[0, :, 2:5] / 1000
        if not hasattr(self, 'crop_time_sum'):
            self.crop_time_sum = 0.0
        if not hasattr(self, 'count'):
            self.count = 0
        if not hasattr(self, 'estimate_time_sum'):
            self.estimate_time_sum = 0.0
        if not hasattr(self, 'calc_rotmax_time_sum'):
            self.calc_rotmax_time_sum = 0.0
        start_time = time.time()
        ###
        hand_bbox_xywh = dict()
        hand_cropped_imgs = dict()
        # crop hands from image
        for hand_index, side in [
            [[14, 15, 16, 17], 'right'],
            [[7, 8, 9, 10], 'left'],
        ]:
            hand_kp3d = body_pose[hand_index]
            # cube_kps3d.shape: (4, 3)
            cube_kps2d = self.projector.project(points=hand_kp3d)[0]
            lu = np.min(cube_kps2d, axis=0)
            rb = np.max(cube_kps2d, axis=0)
            lu, rb = get_square_bbox(w, h, lu, rb, size=2)
            hand_bbox_xywh[side] = np.array(
                [lu[0], lu[1], rb[0] - lu[0], rb[1] - lu[1]])
            cropped_img = img[lu[1]:rb[1], lu[0]:rb[0], :3]
            hand_cropped_imgs[side] = cropped_img
        cropped_img = crop_intaghand_input_img(
            hand_bbox_xywh=hand_bbox_xywh,
            hand_cropped_imgs=hand_cropped_imgs,
            src_img=img_arr)
        if self.ret_crop:
            result['cropped_img'] = cropped_img.copy()
        input_np = transform_img_intaghand(cropped_img)
        input_tensor = torch.tensor(
            data=input_np, dtype=torch.float32,
            device=self.device).unsqueeze(0)
        crop_time = time.time()
        self.crop_time_sum += crop_time - start_time
        with torch.no_grad():
            params = self.estimate_vertice(input_tensor)
        estimate_time = time.time()
        self.estimate_time_sum += estimate_time - crop_time
        hand_dict = {'left': params['j3d_left'], 'right': params['j3d_right']}
        if self.left_optimizer is None and self.right_optimizer is not None:
            handpose = self.joints2rotmat_iknet(hand_dict)
        elif self.left_optimizer is not None \
                and self.right_optimizer is not None:
            handpose = self.joints2rotmat_aik(hand_dict)
        else:
            raise NotImplementedError
        calc_rotmax_time = time.time()
        self.calc_rotmax_time_sum += calc_rotmax_time - estimate_time
        time_diff = self.crop_time_sum + \
            self.estimate_time_sum + self.calc_rotmax_time_sum
        self.count += 1
        if time_diff >= self.profile_period:
            if self.verbose:
                self.logger.info(
                    f'\ncrop_time: {self.crop_time_sum/self.count}\n' +
                    f'estimate_time: {self.estimate_time_sum/self.count}\n' +
                    f'calc_rotmax_time: {self.calc_rotmax_time_sum/self.count}'
                    + '\n')
            self.count = 0
            self.crop_time_sum = 0.0
            self.estimate_time_sum = 0.0
            self.calc_rotmax_time_sum = 0.0
        return handpose
