# yapf: disable
import torch

from realtime_mocap.motion_capture.estimator.base_estimator import (
    BaseEstimator,
)
from realtime_mocap.motion_capture.estimator.cropper.builder import (
    build_cropper,
)
from .body_modules.body_frank_rot6d import BodyFrankRot6d
from .body_modules.hmr import Bottleneck
from .postprocess import DEFAULT_TRANSL, transform_rotation_frankbody
from .preprocess import normalize_img_frankbody

# yapf: able


class BodyFrankEstimator(BaseEstimator):

    def __init__(self,
                 cropper,
                 regressor_checkpoint,
                 smpl_mean_params,
                 device='cpu',
                 logger=None):
        BaseEstimator.__init__(self, logger)
        self.device = torch.device(device)

        # Load pre-trained neural network
        self.model = BodyFrankRot6d(Bottleneck, [3, 4, 6, 3], smpl_mean_params)
        checkpoint = torch.load(regressor_checkpoint)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.eval()
        self.model.to(self.device)

        self.cropper = build_cropper(cropper)

    def forward(self,
                img_arr,
                k4a_pose,
                timestamp,
                **kwargs):
        ret_dict = dict(img_arr=img_arr, timestamp=timestamp,
                        body_pose=None, global_orient=None)
        ret_dict.update(kwargs)
        cropped_img = self.cropper.forward(
            img_arr=img_arr,
            k4a_pose=k4a_pose)
        input_batch_np = normalize_img_frankbody(cropped_img)
        with torch.no_grad():
            input_batch_torch = torch.from_numpy(
                input_batch_np).to(self.device, dtype=torch.float32)
            body_rot6d = self.model(input_batch_torch)
            rotaa_dict = transform_rotation_frankbody(body_rot6d)
        ret_dict.update(rotaa_dict)
        ret_dict['transl'] = DEFAULT_TRANSL
        return ret_dict
