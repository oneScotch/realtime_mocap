# yapf: disable
import numpy as np
import os
import torch
import torch.nn.functional as F
from mmhuman3d.utils.transforms import rotmat_to_aa
from scipy.spatial.transform import Rotation as scipy_Rotation
from scipy.special import softmax
from xrprimer.utils.log_utils import get_logger

from realtime_mocap.utils.geometry_utils import batch_rodrigues
from .numpy_utils import get_smplx_init as get_smplx_init_numpy
from .numpy_utils import orthoprocrustes
from .torch_utils import get_smplx_init as get_smplx_init_torch
from .torch_utils import has_smplx_kinect

get_smplx_init_candidates = dict(
    torch=get_smplx_init_torch,
    numpy=get_smplx_init_numpy
)

try:
    from realtime_mocap.extern.smplx_optimization.smplx_optimization.pykinect.smplx_model import (  # noqa: E501
        ExpBodyModel,
    )
    has_smplx_kinect = True and has_smplx_kinect
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_smplx_kinect = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception

# yapf: enable


class RgbdBodyposePrePoseProcessor:

    def __init__(self,
                 body_model_dir,
                 pykinect_path,
                 betas_path,
                 gender: str = 'male',
                 device: str = 'cuda',
                 backend: str = 'numpy',
                 logger=None):
        self.logger = get_logger(logger)
        self.device = device
        self.gender = gender
        exp_bm, s2k, J, kintree_table = load_exp_bm(body_model_dir,
                                                    pykinect_path, gender, 12,
                                                    device)
        self.backend = backend
        self.pykinect_data = {
            'exp_bm': exp_bm.to(device=self.device, dtype=torch.float32),
            's2k': s2k.astype(np.float32),
            'J': J.astype(np.float32),
            'kintree_table': kintree_table.astype(np.int32)
        }
        self.smplx_parents = torch.tensor(
            self.pykinect_data['kintree_table'][0], device=self.device).long()
        self.betas = np.load(betas_path).reshape(1, -1)
        # default shape
        torch_betas = torch.tensor(
            self.betas, dtype=torch.float32, device=self.device)
        expression = torch.zeros_like(torch_betas)
        shape_components = torch.cat(
            [torch_betas, self.pykinect_data['exp_bm'].fe_scale * expression],
            dim=-1)
        shapedirs = torch.cat([
            self.pykinect_data['exp_bm'].shapedirs,
            self.pykinect_data['exp_bm'].exprdirs
        ],
                              dim=-1)
        v_shaped = self.pykinect_data['exp_bm'].v_template + blend_shapes(
            shape_components, shapedirs)
        self.v_shaped = v_shaped.detach().cpu().numpy()
        J = torch.einsum('bik,ji->bjk',
                         [v_shaped, self.pykinect_data['exp_bm'].J_regressor])
        self.J = J.detach().cpu().numpy()
        self.T = np.eye(4, dtype=np.float32)
        # default jaw, eye, hand pose
        self.default_poses = [
            np.zeros((1, 3), dtype=np.float32),  # jaw
            np.zeros((1, 6), dtype=np.float32),  # eyes
            self.pykinect_data['exp_bm'].left_hand_mean.reshape(
                1, -1).clone().detach().cpu().numpy().astype(np.float32),
            self.pykinect_data['exp_bm'].right_hand_mean.reshape(
                1, -1).clone().detach().cpu().numpy().astype(np.float32)
        ]
        self.bodyparts = np.array([
            0, 3, 6, 9, 13, 16, 18, 20, 20, 20, 20, 14, 17, 19, 21, 21, 21, 21,
            1, 4, 7, 10, 2, 5, 8, 11, 12, 15, 15, 15, 15, 15
        ])
        kintree_path = os.path.join(pykinect_path, 'kintree_kinect')
        self.kintree = np.loadtxt(kintree_path, dtype=int)
        bones = []
        for child, parent in enumerate(self.kintree):
            bones.append([parent, child + 1])
        self.bones = np.array(bones)

        if backend == 'torch':
            self.pykinect_data = {
                'exp_bm':
                exp_bm.to(device=self.device, dtype=torch.float32),
                's2k':
                torch.tensor(s2k, device=self.device, dtype=torch.float32),
                'J':
                torch.tensor(J, device=self.device, dtype=torch.float32),
                'kintree_table':
                torch.tensor(
                    kintree_table, device=self.device, dtype=torch.int32)
            }
            self.betas = torch_betas
            self.T = torch.eye(4, dtype=torch.float32, device=self.device)
        # attr below always uses torch
        for idx, arr in enumerate(self.default_poses):
            self.default_poses[idx] = torch.tensor(
                arr, dtype=torch.float32, device=self.device)
        self.v_shaped = v_shaped
        self.J = J
        self.bodyparts = torch.tensor(self.bodyparts, device=self.device)
        self.bones = torch.tensor(self.bones, device=self.device)
        self.init_global_positioner()
        self.hidden = None

    def init_global_positioner(self):

        self.filtered_kinect_joints = np.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22,
            23, 24, 25, 26, 27, 29, 31
        ])

    def fit_global_position(self, target_points, fit_points, center):
        """fit_points of shape (n, 3) target_points of shape (n, 3) center of
        shape (3,)"""

        R, t = orthoprocrustes(
            fit_points[self.filtered_kinect_joints] - center.reshape(1, -1),
            target_points[self.filtered_kinect_joints])
        rvec = scipy_Rotation.from_matrix(R).as_rotvec()
        tvec = t - center
        return rvec, tvec

    def pre_process(self, input_kinect_joints, **kwargs):
        if self.backend == 'numpy':
            kinect_confs = 2 * np.ones((input_kinect_joints.shape[0], 1),
                                       dtype=input_kinect_joints.dtype)
        elif self.backend == 'torch':
            input_kinect_joints = torch.tensor(
                input_kinect_joints, dtype=torch.float32, device=self.device)
            kinect_confs = 2 * torch.ones(
                size=(len(input_kinect_joints), 1),
                dtype=torch.float32,
                device=input_kinect_joints.device)
        # kp_processor starts
        aligned_kinect_joints = input_kinect_joints - input_kinect_joints[[0]]
        # kp_processor ends

        # get pose_init
        init_body_pose, init_global_rot, init_global_trans = \
            get_smplx_init_candidates[self.backend](
                kinect_joints=aligned_kinect_joints,
                kinect_confs=kinect_confs,
                betas=self.betas,
                kintree_table=self.pykinect_data['kintree_table'],
                T=self.T,
                s2k=self.pykinect_data['s2k'],
                J=self.pykinect_data['J'])
        aligned_kinect_joints = torch.tensor(
            aligned_kinect_joints, device=self.device)
        module_input_dict = dict(kinect_kp=aligned_kinect_joints)

        if self.backend == 'numpy':
            init_body_pose = torch.tensor(
                init_body_pose, dtype=torch.float32, device=self.device)
            init_global_rot = torch.tensor(
                init_global_rot, dtype=torch.float32, device=self.device)
            init_global_trans = torch.tensor(
                init_global_trans, dtype=torch.float32, device=self.device)
        elif self.backend == 'torch':
            init_body_pose = init_body_pose.to(self.device)
            init_global_rot = init_global_rot.to(self.device)
            init_global_trans = init_global_trans.to(self.device)

        module_input_dict['pose_init'] = init_body_pose

        # get twists
        # self.kj_bm.inf starts

        verts, _, A = self.get_smplx_forward_result(init_global_rot,
                                                    init_body_pose,
                                                    init_global_trans)
        # self.kj_bm.inf ends
        # kinect_bm_out2kinect_joints starts
        kinect_joints = verts[0, -32:]
        kinect_joints -= kinect_joints[[0], :]

        # kinect_bm_out2kinect_joints ends
        # self.exp_bm_wrapper.get_twists_v2 starts
        init_A = A[0, :, :3, :3]
        init_A_select = torch.index_select(init_A, 0, self.bodyparts)
        init_A_select_inv = torch.transpose(init_A_select, -2, -1)

        init_dirs = kinect_joints[self.bones[:,
                                             1]] - kinect_joints[self.bones[:,
                                                                            0]]
        init_dirs = init_dirs.to(dtype=torch.float32)
        init_dirs_A_inv = torch.bmm(init_A_select_inv[self.bones[:, 0]],
                                    init_dirs.unsqueeze(-1))[:, :, 0]

        target_dirs = aligned_kinect_joints[
            self.bones[:, 1]] - aligned_kinect_joints[self.bones[:, 0]]
        target_dirs = target_dirs.to(dtype=torch.float32)
        target_dirs_A_inv = torch.bmm(init_A_select_inv[self.bones[:, 0]],
                                      target_dirs.unsqueeze(-1))[:, :, 0]

        twists = rotate_a_b_axis_angle_torch_batched(init_dirs_A_inv,
                                                     target_dirs_A_inv)
        module_input_dict['twists'] = twists
        return module_input_dict

    def infer(self, input_tensor):
        with torch.no_grad():
            net_out_dict, self.hidden = self.rnn_model(input_tensor,
                                                       self.hidden)
        return net_out_dict

    def post_process(self, body_pose_rotmat, input_kinect_joints):
        pred = body_pose_rotmat
        body_pose = rotmat_to_aa(pred)

        verts, joints, _ = self.get_smplx_forward_result(
            body_pose=body_pose,
            global_orient=torch.zeros_like(body_pose[:1, :3]),
            transl=torch.zeros_like(body_pose[:1, :3]))
        result_kinect_joints = verts[0, -32:]
        smplx_pelvis = joints[0, 0]
        global_orient, transl = self.fit_global_position(
            input_kinect_joints,
            result_kinect_joints.detach().cpu().numpy(),
            smplx_pelvis.detach().cpu().numpy())
        return dict(
            body_pose=body_pose.detach().cpu().numpy(),
            global_orient=global_orient,
            transl=transl)

    def get_smplx_forward_result(self, global_orient, body_pose, transl):
        full_pose = torch.cat(
            [global_orient.reshape(1, -1),
             body_pose.reshape(1, -1)] + self.default_poses,
            dim=1)
        verts, joints, A = lbs(
            device=self.device,
            J=self.J,
            pose=full_pose,
            parents=self.smplx_parents,
            v_shaped=self.v_shaped,
            lbs_weights=self.pykinect_data['exp_bm'].weights,
            dtype=self.pykinect_data['exp_bm'].dtype)

        verts = verts + transl.reshape(1, 1, 3)
        joints = joints + transl.reshape(1, 1, 3)
        return verts, joints, A


def load_exp_bm(body_model_dir, pykinect_data_dp, gender, n_pca, device):
    if gender == 'male':
        bm_path = os.path.join(body_model_dir, 'SMPLX_MALE.npz')
        s2k_path = os.path.join(pykinect_data_dp, 'rob75_val/s2k_m.npy')
    elif gender == 'female':
        bm_path = os.path.join(body_model_dir, 'SMPLX_FEMALE.npz')
        s2k_path = os.path.join(pykinect_data_dp, 'rob75_val/s2k_f.npy')
    else:
        raise Exception(f'gender {gender} unknown')

    smpl_dict = np.load(bm_path, allow_pickle=True)
    kintree_table = smpl_dict['kintree_table']

    s2k = np.load(s2k_path)

    kinect_vert_weights_path = os.path.join(pykinect_data_dp,
                                            'rob75_val/weights.npy')
    w_add = np.load(kinect_vert_weights_path)
    w_add = softmax(w_add, axis=1)

    exp_bm = ExpBodyModel(
        bm_path,
        is_hand_pca=True,
        num_hand_pca=n_pca,
        fe_scale=10000,
        s2v=s2k,
        w_add=w_add,
        comp_device=device)

    J_path = os.path.join(pykinect_data_dp, 'rob75_val/J.npy')
    J = np.load(J_path)

    return exp_bm, s2k, J, kintree_table


def lbs(device, J, pose, parents, v_shaped, lbs_weights, dtype=torch.float32):
    """adapted from
    https://github.com/vchoutas/smplx/blob/master/smplx/lbs.py."""

    batch_size = 1

    rot_mats = batch_rodrigues(
        pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

    # skip pose blend shapes
    # v_posed = pose_offsets + v_shaped
    v_posed = v_shaped

    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    W = lbs_weights[-32:].unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = parents.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype,
                               device=device)

    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)[:, -32:]
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """Applies a batch of rigid transformations to the joints.

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    # print(transforms_mat[0][0])
    # print(transforms_mat[0][1])

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:,
                                                                            i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # print(transforms[0][1])

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)],
                     dim=2)


def blend_shapes(betas, shape_disps):
    """Calculates the per vertex displacement due to the blend shapes.

    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def rotate_a_b_axis_angle_torch_batched(a, b):
    a = a / torch.norm(a, dim=1, keepdim=True)
    b = b / torch.norm(b, dim=1, keepdim=True)
    rot_axis = torch.cross(a, b)

    a_proj = b * torch.sum(a * b, dim=1, keepdim=True)
    a_ort = a - a_proj
    theta = torch.atan2(
        torch.norm(a_ort, dim=1, keepdim=True),
        torch.norm(a_proj, dim=1, keepdim=True))

    theta[torch.sum(a *
                    b, dim=1) < 0] = np.pi - theta[torch.sum(a * b, dim=1) < 0]

    aa = rot_axis / torch.norm(rot_axis, dim=1, keepdim=True) * theta

    return aa
