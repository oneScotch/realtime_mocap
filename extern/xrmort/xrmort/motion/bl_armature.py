#!/usr/bin/env python3
# -*- coding=utf-8 -*-
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as spRotation

from .motion import Motion


class ArmatureMotion(Motion):
    """Get Motion from blender's armature structure.
    """

    @classmethod
    def from_armature(
        cls,
        armature_name: str,
        pelvis_name: str,
    ):
        from xrmort.bl_utils.armature import Armature
        from xrmort.bl_utils.scene import SceneWrapper

        armature = Armature(
            armature_name=armature_name,
            hip_bone_name=pelvis_name,
        )
        if pelvis_name not in armature.bones:
            raise ValueError(
                f"pelvis_name={pelvis_name} not in armature.bone_names."
            )
        keys = armature.get_keys()
        n_frames = len(keys)
        scene = SceneWrapper()
        fps = scene.get_fps()

        bone_names = []
        bone_name_to_idx = {}
        parents = []
        found_pelvis = False
        for bone in armature.bones:
            bone_name = bone.name
            if bone_name == pelvis_name:
                found_pelvis = True
            if not found_pelvis:
                # skip bones before pelvis. Those bones should not have animations.
                continue
            bone_name_to_idx[bone_name] = len(bone_name)
            bone_names.append(bone_name)

            parent = bone.parent
            if parent and parent.name in bone_name_to_idx:
                parents.append(bone_name_to_idx[parent.name])
            else:
                parents.append(-1)
        assert bone_names[0] == pelvis_name

        # Get transform
        transl_bl = []
        body_poses_bl = []
        n_frames = max(keys)
        for f in range(n_frames):
            scene.set_frame_current(f)
            frame_quats = []
            for name in bone_names:
                bone = armature.bones[name]
                if name == pelvis_name:
                    transl_bl.append(list(bone.location))
                quat = bone.rotation_quaternion
                frame_quats.append((quat.x, quat.y, quat.z, quat.w))
            frame_body_poses_bl = spRotation.from_quat(frame_quats).as_rotvec()
            body_poses_bl.append(frame_body_poses_bl)

        # Create instance
        transl = np.array(transl_bl, dtype=np.float32).reshape(n_frames, 3)
        body_poses = np.array(body_poses_bl, dtype=np.float32).reshape(n_frames, -1, 3)
        instance = ArmatureMotion(transl=transl, body_poses=body_poses, fps=fps)
        instance.BONE_NAMES = bone_names
        instance.BONE_NAME_TO_IDX = {x: i for i, x in enumerate(bone_names)}
        instance.PARENTS = parents
        return instance

    def get_parent_bone_name(self, bone_name) -> Optional[str]:
        idx = self._bone2idx(bone_name)
        if idx is None:
            raise ValueError(f'bone.name="{bone_name}" not in skeleton.')
        else:
            parent_idx = self.PARENTS[idx]
            if parent_idx == -1:
                return None
            else:
                return self.BONE_NAMES[parent_idx]
