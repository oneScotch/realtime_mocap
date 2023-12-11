import numpy as np
import torch
from manopth import manolayer

from realtime_mocap.utils.ik_utils import normalize_with_target
from .aik import PSO, OneEuroFilter, adaptive_IK, calculate_length
from .base_optimizer import BaseOptimizer


class AIKOptimizer(BaseOptimizer):
    """Adaptive IK for optimizing rotations from given keypoints.

    Currently it is only tested on left and right MANO models.
    """

    def __init__(self,
                 mano_config: dict,
                 ik_device: str = 'cpu',
                 mano_device: str = 'cpu',
                 enable_shape_opt: bool = False,
                 enable_wrist_only: bool = False,
                 logger=None) -> None:
        """Initialization of AIKOptimizer.

        Args:
            mano_config (dict): Config of the oprimizer.
            logger: Logger of the optimizer. Defaults to None.
        """
        super().__init__(logger=logger)
        self.ik_device = torch.device(ik_device)
        self.mano_device = torch.device(mano_device)

        self.side = mano_config['side']
        mano_config['flat_hand_mean'] = True
        self.pose_key = f'{self.side}_hand_pose'
        self.shape_key = f'{self.side}_hand_shape'
        self.mano_layer = manolayer.ManoLayer(**mano_config).to(
            self.mano_device)

        self.enable_wrist_only = enable_wrist_only

        self.enable_shape_opt = enable_shape_opt
        if self.enable_shape_opt:
            # create filter similar to Minimal-Hand
            # self.point_fliter = OneEuroFilter(4.0, 0.0)
            NGEN = 100
            popsize = 100
            low = np.zeros((1, 10)) - 3.0
            up = np.zeros((1, 10)) + 3.0
            self.pso_parameters = [NGEN, popsize, low, up]
            self.shape_fliter = OneEuroFilter(4.0, 0.0)
            self.default_shape = None
        else:
            self.default_shape = torch.zeros(
                size=(1, 10), dtype=torch.float32, device=self.mano_device)

    def aik(self,
            kps3d_tgt,
            pose_init=None,
            do_eval=False,
            output_tensor=False,
            return_joints=False):
        """Adaptive Inverse Kinematics.

        Note that it only supports batch_size == 1 currently.

        Args:
            kps3d_tgt (np.ndarray): positions with shape (1, 21, 3).
            pose_init (torch.Tensor): rotation matrix with shape (1, 16, 3, 3).
            do_eval (bool): whether do evaluation to compute MPJPE between
                input keypoints and keypoints obtained from estimated pose.
            output_tensor (bool): If True, the outputs are torch.Tensor,
                otherwise they are np.ndarray.
            return_joints (bool): If True, return joints with shape (1, 21, 3),
                otherwise only hand_pose and hand_shape will be returned.
        """
        # optimize shape
        if self.enable_shape_opt:
            pre_useful_bone_len = calculate_length(kps3d_tgt, label='useful')
            pso = PSO(
                self.pso_parameters,
                pre_useful_bone_len.reshape((1, 15)),
                mano_layer=self.mano_layer)
            pso.main()
            opt_shape = pso.ng_best
            shape_np = self.shape_fliter.process(opt_shape)
            shape_tensor = torch.tensor(
                shape_np, dtype=torch.float, device=self.mano_device)
        else:
            shape_tensor = self.default_shape

        # optimize pose
        if pose_init is None:
            pose_init = torch.eye(3).repeat(1, 16, 1, 1).to(
                self.mano_device, dtype=torch.float32)
        elif isinstance(pose_init, np.ndarray):
            pose_init = torch.tensor(
                pose_init, dtype=torch.float32, device=self.mano_device)
        elif isinstance(pose_init, torch.Tensor):
            pose_init = pose_init.to(self.mano_device)
        _, kps3d_init = self.mano_layer(pose_init, shape_tensor)
        kps3d_init = kps3d_init.squeeze(0) / 1000.0
        kps3d_init = kps3d_init.to(self.ik_device)
        if isinstance(kps3d_tgt, np.ndarray):
            kps3d_tgt = torch.tensor(
                kps3d_tgt, dtype=torch.float32, device=self.ik_device)
        kps3d_tgt_norm = normalize_with_target(kps3d_tgt, kps3d_init)
        pose_est = adaptive_IK(
            kps3d_init,
            kps3d_tgt_norm,
            th_parallel=1e-5,
            device=self.ik_device,
            enable_wrist_only=self.enable_wrist_only)

        # The final pose_est applies pose_init first, then applying pose_est
        # According to https://github.com/hassony2/manopth/blob/master/manopth/manolayer.py#L188, # noqa: E501
        # `torch.matmul(self.th_posedirs, th_pose_map.transpose(0, 1))` is
        # left multiplication, and thus we put pose_init on the right.
        # TODO: check whether to handle R0 separately
        pose_est_tensor = torch.matmul(pose_est, pose_init.to(self.ik_device))

        if do_eval:
            error = self.evaluate(pose_est_tensor, shape_tensor,
                                  kps3d_tgt_norm)
            self.logger.info(f'AIKOptimizer error: {error:.2f} mm')

        if output_tensor:
            if return_joints:
                _, kps3d_est_tensor = self.mano_layer(pose_est_tensor,
                                                      shape_tensor)
                return pose_est_tensor, shape_tensor, kps3d_est_tensor
            else:
                return pose_est_tensor, shape_tensor
        else:
            pose_np = pose_est_tensor.cpu().numpy()
            shape_np = shape_tensor.cpu().numpy()
            if return_joints:
                _, kps3d_est_tensor = self.mano_layer(pose_est_tensor,
                                                      shape_tensor)
                kps3d_est = kps3d_est_tensor.cpu().numpy()
                return pose_np, shape_np, kps3d_est
            else:
                return pose_np, shape_np

    def evaluate(self, pose, shape, kps3d_tgt):
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).float()
        if isinstance(shape, np.ndarray):
            shape = torch.from_numpy(shape).float()
        _, kps3d_est = self.mano_layer(pose, shape)
        kps3d_est = kps3d_est.squeeze(0)
        kps3d_tgt *= 1000
        err = torch.linalg.norm(kps3d_tgt - kps3d_est, ord=2, axis=-1)
        err = err.mean()
        return err

    def forward(self,
                smplx_data,
                kps3d,
                do_eval=False,
                output_tensor=False,
                return_joints=False,
                **kwargs):
        """Forward function of AIKOptimizer.

        Args:
            smplx_data (Dict): smplx dict used as input for optimization.
                If the hand_pose key does not exist,
                it will be initialized to zero tensor with shape (1, 16, 3, 3).
            keypoints (np.ndarray): (b, 21, 3) positions.
            pose_init (torch.Tensor): (b, 16, 3, 3) rotation matrix.
            do_eval (bool): whether do evaluation to compute MPJPE between
                input keypoints and keypoints obtained from estimated pose.
            output_tensor (bool): If True, the outputs are torch.Tensor,
                otherwise they are np.ndarray.
        """
        pose_init = smplx_data.get(self.pose_key, None)
        ret = self.aik(
            kps3d_tgt=kps3d,
            pose_init=pose_init,
            do_eval=do_eval,
            output_tensor=output_tensor,
            return_joints=return_joints)
        hand_pose, hand_shape = ret[:2]
        smplx_data[self.pose_key] = hand_pose
        smplx_data[self.shape_key] = hand_shape
        if return_joints:
            hand_joints = ret[2]
            return hand_pose, hand_shape, hand_joints
        else:
            return hand_pose, hand_shape

    def __call__(self,
                 smplx_data,
                 kps3d,
                 do_eval=False,
                 output_tensor=False,
                 return_joints=False,
                 **kwargs):
        return self.forward(smplx_data, kps3d, do_eval, output_tensor,
                            return_joints, **kwargs)
