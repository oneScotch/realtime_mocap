#!/usr/bin/env python3
# -*- coding=utf-8 -*-
"""Classes help to parse a skeleton's static definition (in json file).
"""
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from mathutils import Matrix, Quaternion, Vector


class BoneInfo:
    """Static bone info at rest pose,
    including bone's name, parent bone, spatial location & rotation.
    """

    __slots__ = (
        '_pose_bone_info',
        'bone_name',
        'matrix',
        'matrix_basis',
        'matrix_local',
        'location',
        'rotation',
        'parent_name',
        'parent',
        'is_pelvis',
        'is_pose_bone'
    )

    def __init__(self, pose_bone_info: Dict[str, Any], is_pelvis: bool = False):
        self._pose_bone_info = pose_bone_info
        self.bone_name: str = self._pose_bone_info["bone_name"].lstrip("_")
        self.matrix = Matrix(self._pose_bone_info["matrix"])
        self.matrix_basis = Matrix(self._pose_bone_info["matrix_basis"])
        # self._matrix_local = Matrix(self.pose_bone_info["matrix_local"])
        self.matrix_local = Matrix(self._pose_bone_info["matrix_local"])
        self.location = Vector(self._pose_bone_info["location"])
        self.rotation = Quaternion(self._pose_bone_info["rotation"])
        self.parent_name: str = self._pose_bone_info["parent_name"] or ''
        self.parent: Optional['BoneInfo'] = None        # Set parent later

        self.is_pelvis: bool = is_pelvis
        # If not pose bone, the bone is not intended to have animation
        self.is_pose_bone: bool = True

    def __str__(self) -> str:
        return f'BoneInfo<bone_name="{self.bone_name}">'

    def __repr__(self) -> str:
        return self.__str__()


class SkeletonInfo:
    """A skeleton's info, including bones' spatial transform and relationship
    """
    _instances = {}

    @classmethod
    def from_json(cls, json_path: Path) -> 'SkeletonInfo':
        """Get a skeleton singleton from json."""
        json_path = json_path.resolve()
        if json_path not in cls._instances:
            cls._instances[str(json_path)] = SkeletonInfo(json_path)
        instance = cls._instances[str(json_path)]
        return instance

    def __init__(self, json_path: Path):
        self._json_path = Path(json_path)
        with open(self._json_path, 'r') as f:
            self._raw_data = json.load(f)
        self._raw_pose_bones_info: Dict[str, Dict[str, Any]] = \
            self._raw_data["pose_bones"]
        self.armature_matrix = Matrix(self._raw_data["armature_matrix"])
        self.pelvis_name: str = self._raw_data["pelvis_name"]
        self.pelvis_height: float = (
            self.armature_matrix
            @ Matrix(self._raw_pose_bones_info[self.pelvis_name]['matrix'])
        ).decompose()[0].z
        # Setup bones and their relationship
        self.bones: OrderedDict[str, BoneInfo] = OrderedDict()
        for name in self._raw_pose_bones_info.keys():
            self._update_bone_info(name)
        self.bone_names: List[str] = list(self.bones.keys())

    def _update_bone_info(self, bone_name: Optional[str]) -> Optional[BoneInfo]:
        """Update self.bones"""
        if not bone_name:
            return None
        elif bone_name in self.bones:
            bone_info = self.bones[bone_name]
        else:
            is_pelvis = bone_name == self.pelvis_name
            bone_info = BoneInfo(
                self._raw_pose_bones_info[bone_name],
                is_pelvis=is_pelvis
            )
            bone_info.parent = self._update_bone_info(bone_info.parent_name)
            self.bones[bone_name] = bone_info
        return bone_info

    def __str__(self) -> str:
        return f'<SkeletonInfo("{self._json_path}")>'

    def __repr__(self) -> str:
        return self.__str__()
