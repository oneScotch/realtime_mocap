#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import sys
from pathlib import Path
from typing import Dict, Optional, Union

ACTOR_MODELS_DIR_PATH = Path(__file__).resolve().parent / 'models'
ACTOR_SKELETON_INFO_DIR_PATH = Path(__file__).resolve().parent / 'skeletons'


def get_actor_path(filename: str) -> Path:
    actor_path = ACTOR_MODELS_DIR_PATH / filename
    if not actor_path.suffix == ".fbx":
        raise ValueError(
            f"{actor_path.suffix} files are currently not support."
            f" (filename={filename})"
        )
    return actor_path


def get_skeleton_path(filename: str) -> Path:
    actor_path = ACTOR_SKELETON_INFO_DIR_PATH / filename
    return actor_path


class BaseActor:
    actor_path: Optional[Path] = None       # FBX (prefer rigged only)
    actor_meshed_path: Optional[Path] = None    # FBX (with mesh)
    skeleton_path: Optional[Path] = None    # json

    name_to_bone_mapping = {}
    bone_to_name_mapping = {
        v: k for k, v in name_to_bone_mapping.items() if k and v
    }

    def __init__(
        self,
        actor_path: Optional[Union[str, Path]] = None,
        actor_meshed_path: Optional[Union[str, Path]] = None,
        skeleton_path: Optional[Union[str, Path]] = None,
        name_to_bone_mapping: Optional[Dict[str, str]] = None,
    ):
        # meshed actor
        if actor_meshed_path:
            self.actor_meshed_path = Path(actor_meshed_path)
        elif self.__class__.actor_meshed_path:
            self.actor_meshed_path = Path(self.__class__.actor_meshed_path)
        else:
            self.actor_meshed_path = None

        # actor (prefer rigged only)
        if actor_path:
            self.actor_path = Path(actor_path)
        elif self.__class__.actor_path:
            self.actor_path = Path(self.__class__.actor_path)
        elif self.actor_meshed_path:
            # Use a meshed skeleton instead of the rigged one
            self.actor_path = self.actor_meshed_path
        else:
            self.actor_path = None

        if skeleton_path:
            self.skeleton_path = Path(skeleton_path)
        elif self.__class__.skeleton_path:
            self.skeleton_path = Path(self.__class__.skeleton_path)
        else:
            self.skeleton_path = None

        if name_to_bone_mapping:
            self.name_to_bone_mapping = name_to_bone_mapping
        elif self.__class__.name_to_bone_mapping:
            self.name_to_bone_mapping = self.__class__.name_to_bone_mapping
        else:
            self.name_to_bone_mapping = {}

        self.bone_to_name_mapping = {
            v: k for k, v in self.name_to_bone_mapping.items() if k and v
        }
        self.armature_name: str = ""

    def __str__(self) -> str:
        return (
            f'{self.__class__.__name__}'
            f'(actor_path="{self.actor_path}",'
            f' actor_meshed_path="{self.actor_meshed_path}")'
        )

    def import_actor(self, remove_namespace=True, mesh=True):
        """(Only functional in Blender) Import actor into scene."""
        import bpy

        if not self.actor_path:
            raise ValueError("No actor_path given. This actor is not loadable!")
        if not self.actor_path.exists():
            raise ValueError(f"actor_path does not exist: `{self.actor_path}`")

        actor_path = self.actor_path
        if mesh:
            if self.actor_meshed_path is None:
                print(f"Warning: no mesh available for {self}", file=sys.stderr)
            elif not self.actor_meshed_path.exists():
                raise ValueError(
                    "actor_meshed_path does not exist:"
                    f" `{self.actor_meshed_path}`"
                )
            else:
                actor_path = self.actor_meshed_path

        obj_before = bpy.context.object
        try:
            bpy.ops.import_scene.fbx(filepath=str(actor_path))
        except Exception:
            pass
        armature = bpy.context.object
        if not armature or armature is obj_before:
            raise ValueError(
                f"Failed to import fbx model from {self.actor_path}"
            )
        armature = bpy.context.object
        if armature.name not in bpy.data.armatures:
            for x in armature.children:
                if x.name in bpy.data.armatures:
                    armature = x
                    break
            else:
                raise ValueError("Unable to find a armature.")
        self.armature_name = armature.name
        # remove namespace
        if remove_namespace:
            for b in self.armature.pose.bones:
                b.name = b.name.rpartition(":")[-1]

    @property
    def armature(self):
        import bpy

        arm_ = bpy.data.objects.get(self.armature_name)
        if not arm_:
            raise ValueError(
                f"Failed to get armature `{self.armature_name}` in the scene."
            )
        else:
            return arm_

    def name2bone(self, name: str) -> Optional[str]:
        bone = self.name_to_bone_mapping.get(name) if name else None
        return bone or None

    def bone2name(self, bone: str) -> Optional[str]:
        name = self.bone_to_name_mapping.get(bone) if bone else None
        return name or None

    def to_actor_bone(
        self, other_actor: "BaseActor", bone: str
    ) -> Optional[str]:
        if bone:
            name = self.bone2name(bone)
            if name:
                bone_b = other_actor.name2bone(name)
                if bone_b:
                    return bone_b
        return None

    def rename_namespace(self, new_namespace: str = ""):
        for bone in self.armature.pose.bones:
            name = bone.name.rpartition(":")[-1]
            new_name = name if not new_namespace else f"{new_namespace}:{name}"
            print(f"\t{name} -> {new_name}")
            bone.name = new_name
