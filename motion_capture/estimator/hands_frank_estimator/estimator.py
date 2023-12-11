# yapf: disable
import numpy as np
import pickle
import torch

from realtime_mocap.motion_capture.estimator.base_estimator import (
    BaseEstimator,
)
from ..cropper.builder import build_cropper
from .hand_modules.h3dw_networks import H3DWEncoder
from .hand_modules.hands_frank_rotmat_module import HandsFrankRotmatModule
from .preprocess import get_input_batch_frank


# yapf: enable
def load_pkl(pkl_file):
    assert pkl_file.endswith('.pkl')
    with open(pkl_file, 'rb') as in_f:
        try:
            data = pickle.load(in_f)
        except UnicodeDecodeError:
            in_f.seek(0)
            data = pickle.load(in_f, encoding='latin1')
    return data


def load_params(mean_param_file, total_params_dim, batch_size=1):
    # load mean params first
    mean_vals = load_pkl(mean_param_file)
    mean_params = np.zeros((1, total_params_dim))

    # set camera model first
    mean_params[0, 0] = 5.0

    # set pose (might be problematic)
    mean_pose = mean_vals['mean_pose'][3:]
    # set hand global rotation
    mean_pose = np.concatenate((np.zeros((3, )), mean_pose))
    mean_pose = mean_pose[None, :]

    # set shape
    mean_shape = np.zeros((1, 10))
    mean_params[0, 3:] = np.hstack((mean_pose, mean_shape))
    # concat them together
    mean_params = np.repeat(mean_params, batch_size, axis=0)
    mean_params = torch.from_numpy(mean_params).float()
    mean_params.requires_grad = False
    return mean_params


class HandsFrankEstimator(BaseEstimator):

    def __init__(self,
                 cropper,
                 mean_hand_dir,
                 hand_config,
                 device='cpu',
                 logger=None):
        BaseEstimator.__init__(self, logger)
        self.device = device
        self.cropper = build_cropper(cropper)

        self.cam_params_dim = hand_config['cam_params_dim']
        self.pose_params_dim = hand_config['pose_params_dim']
        assert (hand_config['total_params_dim'] == self.cam_params_dim +
                self.pose_params_dim + hand_config['shape_params_dim'])

        # load mean params, the mean params are from HMR
        mean_params = load_params(hand_config['mean_param_file'],
                                  hand_config['total_params_dim'],
                                  hand_config['batchSize'])

        # set encoder and optimizer
        encoder = H3DWEncoder(hand_config, mean_params, self.device)
        saved_weights = torch.load(hand_config['checkpoint_path'])
        encoder.load_state_dict(saved_weights)
        encoder.eval()
        self.model = HandsFrankRotmatModule(
            encoder, mean_hand_dir, device=self.device)

    def forward(self,
                img_arr,
                k4a_pose,
                hands_keypoints2d=None,
                hands_bboxes=None,
                **kwargs):
        ret_dict = dict(
            img_arr=img_arr,
            k4a_pose=k4a_pose,
            hands_keypoints2d=hands_keypoints2d,
            hands_bboxes=hands_bboxes)
        ret_dict.update(kwargs)
        cropped_imgs, bbox_dict = self.cropper.forward(
            img_arr=img_arr,
            k4a_pose=k4a_pose,
            mp_keypoints2d_dict=hands_keypoints2d,
            hands_bboxes=hands_bboxes)
        normed_img_np = get_input_batch_frank(cropped_imgs, bbox_dict)

        with torch.no_grad():
            dhands_rotmat = self.model(
                torch.from_numpy(normed_img_np).to(
                    self.device, dtype=torch.float32))
        if bbox_dict['left'] is None:
            ret_dict['left_hand_rotmat'] = None
        else:
            ret_dict['left_hand_rotmat'] = dhands_rotmat[0:1].detach().cpu(
            ).numpy()

        if bbox_dict['right'] is None:
            ret_dict['right_hand_rotmat'] = None
        else:
            ret_dict['right_hand_rotmat'] = dhands_rotmat[1:2].detach().cpu(
            ).numpy()
        return ret_dict
