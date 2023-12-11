#!/usr/bin/env python3
# -*- coding=utf-8 -*-
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as spRotation


class Converter:
    @classmethod
    def vec_humandata2smplx(cls, vector: np.ndarray) -> np.ndarray:
        """From humandata transl (in **OpenCV space**) to SMPLX armature's
        **pelvis local space** in Blender.
        (The pelvis local space is designed to be the same with **SMPL space**.)

        [right, front, up]: (-x, -z, -y) ==> (-x, z, y)

        Args:
            vector (np.ndarray): of shape (N, 3) or (3,)

        Returns:
            np.ndarray: of shape (N, 3) or (3,)
        """
        if vector.shape == (3,):
            vector = np.array(
                [vector[0], -vector[1], -vector[2]], dtype=vector.dtype
            )
        elif len(vector.shape) == 2 and vector.shape[1] == 3:
            vector = np.array([vector[:, 0], -vector[:, 1], -vector[:, 2]]).T
        else:
            raise ValueError(f"vector.shape={vector.shape}")
        return vector

    @classmethod
    def vec_smplx2humandata(cls, vector: np.ndarray) -> np.ndarray:
        # vice versa
        return cls.vec_humandata2smplx(vector)

    @classmethod
    def vec_amass2humandata(cls, vector: np.ndarray) -> np.ndarray:
        """From amass transl (pelvis's local space) to humandata transl
        (in **OpenCV space**)

        [right, front, up]: (x, y, z) ==> (-x, -z, -y)

        (CAUTION: we can see amass animation actors face back
            in blender via the smplx add-on)

        Args:
            vector (np.ndarray): of shape (N, 3) or (3,)

        Returns:
            np.ndarray: of shape (N, 3) or (3,)
        """
        if vector.shape == (3,):
            vector = np.array(
                [-vector[0], -vector[2], -vector[1]], dtype=vector.dtype
            )
        elif len(vector.shape) == 2 and vector.shape[1] == 3:
            vector = np.array([-vector[:, 0], -vector[:, 2], -vector[:, 1]]).T
        else:
            raise ValueError(f"vector.shape={vector.shape}")
        return vector

    @classmethod
    def vec_ybot2humandata(cls, vector: np.ndarray, scaling=1.0) -> np.ndarray:
        """From ybot/ffbiped transl (pelvis's local space) to humandata transl
        (in **OpenCV space**)

        [right, front, up]: (-x, z, y) ==> (-x, -z, -y)

        Args:
            vector (np.ndarray): of shape (N, 3) or (3,)

        Returns:
            np.ndarray: of shape (N, 3) or (3,)
        """
        if vector.shape == (3,):
            vector = (
                np.array(
                    [vector[0], -vector[1], -vector[2]], dtype=vector.dtype
                )
                / scaling
            )
        elif len(vector.shape) == 2 and vector.shape[1] == 3:
            vector = (
                np.array([vector[:, 0], -vector[:, 1], -vector[:, 2]]).T
                / scaling
            )
        else:
            raise ValueError(f"vector.shape={vector.shape}")
        return vector


class Motion:
    """Wrap motion data. Provide methods to get transform info for 3D
    calculations.

    The motion data will be used along with `Skeleton` instance in retargeting,
    and the local spaces of bones are all defined in such skeletons.
    """

    BONE_NAMES: List[str]
    BONE_NAME_TO_IDX: Dict[str, int]
    PARENTS: List[int]

    def __init__(
        self,
        transl: np.ndarray,
        body_poses: np.ndarray,
        n_frames: Optional[int] = None,
        fps: float = 30.0,
    ) -> None:
        """transl & body_poses are in the space of
        corresponding `Skeleton` instance.
        """
        transl = transl.reshape([-1, 3])
        body_poses = body_poses.reshape([body_poses.shape[0], -1, 3])
        if n_frames is None:
            n_frames = min(transl.shape[0], body_poses.shape[0])
        self.transl: np.ndarray = transl[:n_frames, :]
        self.body_poses: np.ndarray = body_poses[:n_frames, :, :]
        self.global_orient: np.ndarray = self.body_poses[:, 0, :]

        assert n_frames > 0, f"n_frames={n_frames}"
        self.n_frames = n_frames
        assert fps > 0, f"fps={fps}"
        self.fps = fps

    def _bone2idx(self, bone_name) -> Optional[int]:
        return self.BONE_NAME_TO_IDX.get(bone_name)

    def get_transl(self, frame=0) -> np.ndarray:
        return self.transl[frame, :3]

    def get_global_orient(self, frame=0) -> np.ndarray:
        return self.global_orient[frame, :3]

    def get_bone_rotvec(self, bone_name, frame=0) -> np.ndarray:
        idx = self._bone2idx(bone_name)
        if idx == 0:
            return self.get_global_orient(frame)
        elif idx:
            return self.body_poses[frame, idx, :3]
        else:
            return np.zeros([3], dtype=np.float32)

    def get_bone_rotation(self, bone_name: str, frame=0) -> spRotation:
        rotvec = self.get_bone_rotvec(bone_name, frame)
        return spRotation.from_rotvec(rotvec)  # type: ignore

    def get_bone_matrix_basis(self, bone_name: str, frame=0) -> np.ndarray:
        """pose2rest: relative to the bone space at rest pose.

        Result:
            np.ndarray: transform matrix like
                [
                    [R, T],
                    [0, 1]
                ]
        """
        idx = self._bone2idx(bone_name)
        if idx == 0:
            transl = self.get_transl(frame)
        else:
            transl = np.zeros(3)
        rot = self.get_bone_rotation(bone_name, frame)
        matrix_basis = rot.as_matrix()
        matrix_basis = np.pad(matrix_basis, (0, 1))
        matrix_basis[:3, 3] = transl
        matrix_basis[3, 3] = 1
        return matrix_basis

    def get_parent_bone_name(self, bone_name: str) -> Optional[str]:
        ...

    def convert_fps_smplx_data(
        self, smplx_data: Dict[str, np.ndarray], scaling: int
    ) -> Dict[str, np.ndarray]:
        for key, value in smplx_data.items():
            if key in ['betas']:
                continue
            smplx_data[key] = value[::scaling, :]
        return smplx_data

    def convert_fps(self, fps):
        if fps == self.fps:
            return

        scaling = self.fps / fps
        if scaling - int(scaling) <= 1e-7:
            scaling = int(scaling)
            self.transl = self.transl[::scaling, :]
            self.body_poses = self.body_poses[::scaling, :, :]
            self.global_orient: np.ndarray = self.global_orient[::scaling, :]
            self.n_frames = self.body_poses.shape[0]
            if hasattr(self, "smpl_data"):
                self.smpl_data = self.convert_fps_smplx_data(
                    self.smpl_data, scaling
                )
            if hasattr(self, "smplx_data"):
                self.smplx_data = self.convert_fps_smplx_data(
                    self.smplx_data, scaling
                )
            self.fps = fps
        elif fps > self.fps:
            # TODO: motion interpolation
            raise NotImplementedError(
                f"Not support up sampling from {self.fps}fps to {fps}fps"
            )
        else:
            # TODO: motion interpolation
            raise NotImplementedError(
                f"Not support down sampling from {self.fps}fps to {fps}fps"
            )
