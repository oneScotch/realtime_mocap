# yapf: disable
import numpy as np
import time
import torch

from realtime_mocap.motion_capture.estimator.base_estimator import (
    BaseEstimator,
)
from .nn_module import BodyRgbdRotmatModule
from .pre_post_processor import RgbdBodyposePrePoseProcessor


# yapf: enable
class BodyRgbdPytorchEstimator(BaseEstimator):

    def __init__(self,
                 rnn_checkpoint_path,
                 body_model_dir,
                 pykinect_path,
                 betas_path,
                 gender='male',
                 device='cuda',
                 profile_period: float = 10.0,
                 verbose: bool = True,
                 logger=None):
        super().__init__(logger=logger)
        self.device = device
        self.bodypose_module = BodyRgbdRotmatModule(
            rnn_checkpoint_path, device=device)
        self.processor = RgbdBodyposePrePoseProcessor(
            body_model_dir=body_model_dir,
            pykinect_path=pykinect_path,
            betas_path=betas_path,
            gender=gender,
            device=device)
        self.hidden = None

        self.profile_period = profile_period
        self.verbose = verbose

        self.forward_count = 0
        self.last_profile_time = time.time()

        self.pre_time_sum = 0.0
        self.rnn_time_sum = 0.0
        self.post_time_sum = 0.0

    def forward(self,
                img_arr,
                k4a_pose,
                timestamp=None,
                frame_idx=None,
                **kwargs):
        start_time = time.time()
        with torch.no_grad():
            input_kinect_joints = (k4a_pose[0, :, 2:5] / 1000).astype(
                np.float32)
            module_input_dict = self.processor.pre_process(input_kinect_joints)
            pre_time = time.time()
            self.pre_time_sum += pre_time - start_time
            body_pose_rotmat, self.hidden = self.bodypose_module.forward(
                kinect_kp=module_input_dict['kinect_kp'].to(
                    self.bodypose_module.device),
                pose_init=module_input_dict['pose_init'].to(
                    self.bodypose_module.device),
                twists=module_input_dict['twists'].to(
                    self.bodypose_module.device),
                hidden=self.hidden)
            self.hidden = self.hidden.detach()
            rnn_time = time.time()
            self.rnn_time_sum += rnn_time - pre_time
            smplx_result = self.processor.post_process(
                body_pose_rotmat.to(self.processor.device),
                input_kinect_joints)
            post_time = time.time()
            self.post_time_sum += post_time - rnn_time
        smplx_result.update(
            dict(timestamp=timestamp, frame_idx=frame_idx, img_arr=img_arr))
        time_diff = time.time() - self.last_profile_time
        self.forward_count += 1
        if time_diff >= self.profile_period:
            if self.verbose:
                self.logger.info(
                    '\n' + 'pre_time:' +
                    f' {self.pre_time_sum/self.forward_count}\n' +
                    'rnn_time:' +
                    f' {self.rnn_time_sum/self.forward_count}\n' +
                    'post_time:' +
                    f' {self.post_time_sum/self.forward_count}\n' + '\n')
            self.forward_count = 0
            self.last_profile_time = time.time()
            self.pre_time_sum = 0.0
            self.rnn_time_sum = 0.0
            self.post_time_sum = 0.0
        return smplx_result
