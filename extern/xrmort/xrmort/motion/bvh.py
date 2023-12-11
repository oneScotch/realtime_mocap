#!/usr/bin/env python3
# -*- coding=utf-8 -*-
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation as spRotation

from xrmort.io.bvh import BVHImporter
from .motion import Converter, Motion

ConverterType = Callable[[np.ndarray], np.ndarray]


class BVHMotion(Motion):
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
    def from_bvh_file(
        cls,
        bvh_file_path: Union[str, Path],
    ) -> 'BVHMotion':
        importer = BVHImporter()
        with open(bvh_file_path, "r") as f:
            bvh_data = f.read()
        importer.parse(bvh_data)
        importer.joint_names




    @classmethod
    def from_bvh_data(
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
