#!/usr/bin/env python3
# -*- coding=utf-8 -*-
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as spRotation

from xrmort.constants import (
    NUM_SMPLX_BODYJOINTS,
    SMPL_IDX_TO_JOINTS,
    SMPL_PARENT_IDX,
    SMPLX_HAND_POSES,
    SMPLX_IDX_TO_JOINTS,
    SMPLX_JOINT_NAMES,
    SMPLX_PARENT_IDX,
)

from .motion import Converter, Motion

ConverterType = Callable[[np.ndarray], np.ndarray]


class SMPLMotion(Motion):
    SMPL_IDX_TO_NAME: Dict[int, str] = OrderedDict(SMPL_IDX_TO_JOINTS)
    NAME_TO_SMPL_IDX = OrderedDict(
        [(v, k) for k, v in SMPL_IDX_TO_NAME.items() if v]
    )
    NAMES = [x for x in SMPL_IDX_TO_NAME.values() if x]
    PARENTS = list(SMPL_PARENT_IDX)
    BONE_NAMES = SMPLX_JOINT_NAMES[:NUM_SMPLX_BODYJOINTS]
    BONE_NAME_TO_IDX: Dict[str, int] = {
        bone_name: idx for idx, bone_name in enumerate(BONE_NAMES)
    }

    # In order to make the smpl head up to +z
    GLOBAL_ORIENT_ADJUSTMENT = spRotation.from_euler(
        'xyz', np.deg2rad([180, 0, 0])
    )

    def __init__(
        self,
        transl: np.ndarray,
        body_poses: np.ndarray,
        n_frames: Optional[int] = None,
        fps: float = 30.0,
    ) -> None:
        super().__init__(transl, body_poses, n_frames=n_frames, fps=fps)
        self.smpl_data: Dict[str, np.ndarray]

    @classmethod
    def from_smpl_data(
        cls,
        smpl_data: Dict[str, np.ndarray],
        fps: float = 30.0,
        insert_rest_pose: bool = False,
        global_orient_adj: Optional[spRotation] = GLOBAL_ORIENT_ADJUSTMENT,
        vector_convertor: Optional[
            ConverterType
        ] = Converter.vec_humandata2smplx,
    ) -> 'SMPLMotion':
        """Create SMPLMotion instance from smpl_data.

        `smpl_data` should be a dict like object,
        with required keys: ["body_pose", "global_orient"], and optional key ["transl"].

        Args:
            smpl_data: dict with require keys ["body_pose", "global_orient"]
                and optional key ["transl"]
            insert_rest_pose (bool): whether to insert a rest pose at the 0th-frame.

        Returns:
            SMPLMotion: _description_
        """
        smpl_data = dict(smpl_data)
        _get_smpl = partial(
            _get_from_smpl_x, smpl_x_data=smpl_data, dtype=np.float32
        )

        n_frames = smpl_data['body_pose'].shape[0]
        betas = _get_smpl('betas', shape=[10])
        transl = _get_smpl('transl', shape=[n_frames, 3], required=False)
        global_orient = _get_smpl('global_orient', shape=[n_frames, 3])
        body_pose = _get_smpl('body_pose', shape=[n_frames, -1])
        if body_pose.shape[1] == 63:
            body_pose = np.concatenate(
                [body_pose, np.zeros([n_frames, 6])], axis=1
            )
        assert body_pose.shape[1] == 69, f"body_pose.shape={body_pose.shape}"
        # Insert the 0 frame as a T-Pose
        smpl_data = {
            "betas": betas,
            "transl": transl,
            "global_orient": global_orient,
            "body_pose": body_pose,
        }
        if insert_rest_pose:
            for key, arr in smpl_data.items():
                if key != "betas":
                    arr = np.insert(arr, 0, 0, axis=0)
                    smpl_data[key] = arr

        # Create instance
        transl_bl = smpl_data['transl']
        n_frames = transl_bl.shape[0]
        body_poses_bl = np.concatenate(
            [smpl_data[key] for key in ('global_orient', 'body_pose')],
            axis=1,
            dtype=np.float32,
        ).reshape([n_frames, -1, 3])
        # - Adjust in order to make the smpl head up to +z
        if global_orient_adj is not None:
            body_poses_bl[:, 0, :] = (
                global_orient_adj
                * spRotation.from_rotvec(body_poses_bl[:, 0, :])
            ).as_rotvec()
        # - Convert from humandata to smplx pelvis local space in blender
        if vector_convertor is not None:
            transl_bl = vector_convertor(transl_bl)
        instance = SMPLMotion(
            transl=transl_bl, body_poses=body_poses_bl, fps=fps
        )
        instance.smpl_data = smpl_data
        return instance

    def get_bone_rotvec(self, bone_name, frame=0) -> np.ndarray:
        idx = self._bone2idx(bone_name)
        if idx == 0:
            return self.get_global_orient(frame)
        elif idx:
            return self.body_poses[frame, idx, :3]
        else:
            return np.zeros([3], dtype=np.float32)

    def get_parent_bone_name(self, bone_name) -> Optional[str]:
        idx = self._bone2idx(bone_name)
        if idx is None:
            raise ValueError(f'bone.name="{bone_name}" not in smpl skeleton.')
        else:
            parent_idx = self.PARENTS[idx]
            if parent_idx == -1:
                return None
            else:
                return self.BONE_NAMES[parent_idx]


class SMPLXMotion(Motion):
    SMPLX_IDX_TO_NAME: Dict[int, str] = OrderedDict(SMPLX_IDX_TO_JOINTS)
    NAME_TO_SMPL_IDX = OrderedDict(
        [(v, k) for k, v in SMPLX_IDX_TO_NAME.items() if v]
    )
    NAMES = [x for x in SMPLX_IDX_TO_NAME.values() if x]
    PARENTS = list(SMPLX_PARENT_IDX)
    BONE_NAMES = SMPLX_JOINT_NAMES
    BONE_NAME_TO_IDX: Dict[str, int] = {
        bone_name: idx for idx, bone_name in enumerate(BONE_NAMES)
    }

    # In order to make the smpl head up to +z
    GLOBAL_ORIENT_ADJUSTMENT = spRotation.from_euler(
        'xyz', np.deg2rad([180, 0, 0])
    )

    def __init__(
        self,
        transl: np.ndarray,
        body_poses: np.ndarray,
        n_frames: Optional[int] = None,
        fps: float = 30.0,
    ) -> None:
        super().__init__(transl, body_poses, n_frames=n_frames, fps=fps)
        self.smplx_data: Dict[str, np.ndarray]

    @classmethod
    def from_smplx_data(
        cls,
        smplx_data: Dict[str, np.ndarray],
        fps: float = 30.0,
        insert_rest_pose: bool = False,
        flat_hand_mean: bool = False,
        global_orient_adj: Optional[spRotation] = GLOBAL_ORIENT_ADJUSTMENT,
        vector_convertor: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = Converter.vec_humandata2smplx,
    ) -> 'SMPLXMotion':
        """Create SMPLXMotion instance from smpl_data.

        `smplx_data` should be a dict like object,
        with required keys: ["body_pose", "global_orient"], and optional key ["transl"].

        Args:
            smplx_data: require keys ["body_pose", "global_orient"]
                and optional key ["transl"]
            fps (float): the motion's FPS. Defaults to 30.0.
            insert_rest_pose (bool): whether to insert a rest pose at the 0th-frame.
                Defaults to False.
            flat_hand_mean (bool): whether the hands with zero rotations are flat hands.
                Defaults to False.
            global_orient_adj (spRotation, None):
            vector_convertor: a function applies to smplx_data's translation.

        Returns:
            SMPLXMotion: _description_
        """
        smplx_data = dict(smplx_data)
        _get_smplx = partial(
            _get_from_smpl_x, smpl_x_data=smplx_data, dtype=np.float32
        )
        n_frames = smplx_data['body_pose'].shape[0]
        betas = _get_smplx('betas', shape=[10])
        transl = _get_smplx('transl', shape=[n_frames, 3], required=False)
        global_orient = _get_smplx('global_orient', shape=[n_frames, 3])
        body_pose = _get_smplx('body_pose', shape=[n_frames, 63])
        jaw_pose = _get_smplx('jaw_pose', shape=[n_frames, 3], required=False)
        leye_pose = _get_smplx('leye_pose', shape=[n_frames, 3], required=False)
        reye_pose = _get_smplx('reye_pose', shape=[n_frames, 3], required=False)
        left_hand_pose = _get_smplx('left_hand_pose', shape=[n_frames, 45], required=False)
        right_hand_pose = _get_smplx('right_hand_pose', shape=[n_frames, 45], required=False)
        expression = _get_smplx('expression', shape=[n_frames, 10], required=False)

        # Insert the 0 frame as a T-Pose
        smplx_data = {
            "betas": betas,
            "transl": transl,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "jaw_pose": jaw_pose,
            "leye_pose": leye_pose,
            "reye_pose": reye_pose,
            "expression": expression,
        }
        if insert_rest_pose:
            for key, arr in smplx_data.items():
                if key != 'betas':
                    arr = np.insert(arr, 0, 0, axis=0)
                    smplx_data[key] = arr

        # Create instance
        transl_bl = smplx_data['transl']
        # hand relax pose
        if not flat_hand_mean:
            left_hand_relax_pose = np.array(
                SMPLX_HAND_POSES['relaxed'][0]
            ).reshape(45)
            right_hand_relax_pose = np.array(
                SMPLX_HAND_POSES['relaxed'][1]
            ).reshape(45)
            smplx_data['left_hand_pose'] += left_hand_relax_pose
            smplx_data['right_hand_pose'] += right_hand_relax_pose
        # - Adjust in order to make the smpl head up to +z
        if global_orient_adj is not None:
            smplx_data['global_orient'] = (
                global_orient_adj * spRotation.from_rotvec(smplx_data['global_orient'])  # type: ignore
            ).as_rotvec()
            if insert_rest_pose:
                smplx_data['global_orient'][0] = 0
        # - Convert from humandata to smplx pelvis local space in blender
        if vector_convertor is not None:
            transl_bl = vector_convertor(transl_bl)
            smplx_data['transl'] = transl_bl

        # Concatenate all the poses
        body_pose_keys = (
            'global_orient',
            'body_pose',
            'jaw_pose',
            'leye_pose',
            'reye_pose',
            'left_hand_pose',
            'right_hand_pose',
        )
        body_poses_bl = [smplx_data[key] for key in body_pose_keys]
        n_frames = transl_bl.shape[0]
        body_poses_bl = np.concatenate(
            body_poses_bl, axis=1, dtype=np.float32
        ).reshape([n_frames, -1, 3])

        instance = SMPLXMotion(
            transl=transl_bl, body_poses=body_poses_bl, fps=fps
        )
        instance.smplx_data = smplx_data
        return instance

    @classmethod
    def from_amass_data(cls, amass_data, insert_rest_pose: bool):
        assert (
            amass_data['surface_model_type'] == 'smplx'
        ), f"surface_model_type={amass_data['surface_model_type']}"
        fps = amass_data['mocap_frame_rate']

        betas = amass_data['betas'][:10]
        transl = amass_data['trans']
        global_orient = amass_data['root_orient']
        body_pose = amass_data['pose_body']
        left_hand_pose = amass_data['pose_hand'][:, :45]
        right_hand_pose = amass_data['pose_hand'][:, 45:]
        jaw_pose = amass_data['pose_jaw']
        leye_pose = amass_data['pose_eye'][:, :3]
        reye_pose = amass_data['pose_eye'][:, 3:]
        n_frames = global_orient.shape[0]
        expression = np.zeros([n_frames, 10], dtype=np.float32)

        # motions in AMASS dataset are -y up, rotate it to +y up
        amass2humandata_adj = spRotation.from_euler('xyz', np.deg2rad([90, 180, 0]))  # type: ignore
        global_orient = (amass2humandata_adj * spRotation.from_rotvec(global_orient)).as_rotvec()  # type: ignore
        # transl_0 = transl[0, :]
        # transl = amass2humandata_adj.apply(transl - transl_0) + transl_0
        transl = Converter.vec_amass2humandata(transl)
        # TODO: all axis offset
        height_offset = transl[0, 1]

        smplx_data = {
            "betas": betas,
            "transl": transl,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "jaw_pose": jaw_pose,
            "leye_pose": leye_pose,
            "reye_pose": reye_pose,
            "expression": expression,
        }
        if insert_rest_pose:
            for key, arr in smplx_data.items():
                arr = arr.astype(np.float32)
                if key != "betas":
                    arr = np.insert(arr, 0, 0, axis=0)
                    if key == "global_orient":
                        # make 0-th frame has the same orient with humandata
                        arr[0, :] = [np.pi, 0, 0]
                    elif key == "transl":
                        arr[1:, 1] -= height_offset
                        # TODO: handle pelvis height, get pelvis_height, and set frame-0 as T-pose
                        # arr[0, 1] = pelvis_height
                smplx_data[key] = arr

        return cls.from_smplx_data(
            smplx_data,
            insert_rest_pose=False,
            fps=fps,
            flat_hand_mean=True,
        )

    def get_parent_bone_name(self, bone_name) -> Optional[str]:
        idx = self._bone2idx(bone_name)
        if idx is None:
            raise ValueError(f'bone.name="{bone_name}" not in smplx skeleton.')
        else:
            parent_idx = self.PARENTS[idx]
            if parent_idx == -1:
                return None
            else:
                return self.BONE_NAMES[parent_idx]


def _get_from_smpl_x(
    key, shape, *, smpl_x_data, dtype=np.float32, required=True
) -> np.ndarray:
    if required or key in smpl_x_data:
        return smpl_x_data[key].astype(dtype).reshape(shape)
    else:
        return np.zeros(shape, dtype=dtype)
