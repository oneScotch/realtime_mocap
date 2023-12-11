#!/usr/bin/env python3
# -*- coding=utf-8 -*-
import math
import re
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import bpy
import mathutils
import numpy as np


class Armature:
    def __init__(
        self,
        armature_name: str,
        hip_bone_name: Optional[str] = None,
        log: bool = False,
    ) -> None:
        self.armature_name = armature_name
        self.hip_bone_name = hip_bone_name
        self.log = log

    @property
    def _armature(self):
        arm = bpy.data.objects.get(self.armature_name)
        if not arm:
            raise ValueError(f"Armature not found: {self.armature_name}")
        return arm

    @property
    def bones(self):
        return self._armature.pose.bones

    @property
    def bone_names(self) -> List[str]:
        return [b.name for b in self.bones]

    @property
    def hip_bone(self):
        if self.hip_bone_name:
            root = self.bones[self.hip_bone_name]
        else:
            root = self.bones[0]
            while True:
                if not root.children:
                    raise ValueError(
                        "Failed to find a hip bone! Please check the skeleton."
                    )
                elif len(root.children) == 1:
                    # root bone should have several children like left&right hips and a spine.
                    root = root.children[0]
                    continue
                else:
                    # Found hip
                    break
        return root

    @property
    def _fcurves(self):
        return self._armature.animation_data.action.fcurves

    def set_key(self) -> None:
        for bone in self.bones:
            bone.keyframe_insert(data_path='location')
            bone.keyframe_insert(data_path='rotation_quaternion')

    def get_keys(self) -> List[int]:
        keys = set()
        for fcu in self._fcurves:
            parts = fcu.data_path.split('"')
            if len(parts) > 2 and parts[1] == self.hip_bone.name:
                for keyframe in fcu.keyframe_points:
                    keys.add(keyframe.co[0])
        assert keys, f"No keyframes found of bone_name == {self.hip_bone.name}."
        keys_list = [int(round(x)) for x in keys]
        keys_list.sort()
        return keys_list

    def get_keyframes(self) -> Tuple[Dict[str, np.ndarray], List[float]]:
        """Read keyframes data from target objects.

        Returns:
            Dict[str, np.array]: Keys are strings in format of "{bone_name}.{attribute}".
                Attributes are in {"translation", "rotation_quaternion", "scale"}.
                The values are 2d np.array, whose 1st column is a list of keys,
                    and the other columns are attribute values at the corresponding keys.
                If the attribute is "rotation_quaternion", the values will be in the shape of (T, 5),
                    stands for `t` & quaternion(`w`, `x`, `y`, `z`);
                Otherwise the values will be in the shape of (T, 4), stands for `t` & `x`, `y`, `z`

                e.g.
                {
                    "pelvis.location": np.array([[1, 10, 0, 5],
                                                 [2, 20, 0, 1],
                                                 [3, 15, 0, 0],
                                                 [4, 10, 0, 1],
                                                 ...
                                                 ]),
                    "pelvis.rotation_quaternion": np.array([[1, 1, 0, 0, 0],
                                                            [2, 1, 0, 0, 0],
                                                            [3, 1, 0, 0, 0],
                                                            [4, 1, 0, 0, 0],
                                                            ...
                                                            ]),
                }
            List[float]: all keys of those keyframes
        """
        armature = self._armature
        keyframes_raw = OrderedDict()
        bone_to_data_path = {}

        pattern = re.compile(r'^pose\.bones\["(.+?)"\]\.(.+?)$')
        anim = armature.animation_data
        all_keys = set()
        if anim is not None and anim.action is not None:
            for fcu in anim.action.fcurves:
                # parse name and channel
                matched = re.match(pattern, fcu.data_path)
                if not matched:
                    continue
                bone_name = matched.group(1)
                # attribute in ('location', 'rotation_quaternion')
                attribute = matched.group(2)
                bone_attribute = (bone_name, attribute)
                channel = fcu.array_index
                bone_to_data_path[bone_attribute] = fcu.data_path

                # get keyframe points
                keys = []
                values = []
                for keyframe in fcu.keyframe_points:
                    k, v = keyframe.co
                    keys.append(k)
                    values.append(v)
                    all_keys.add(k)

                # reformat keyframes data
                if bone_attribute not in keyframes_raw:
                    # the 0th column is keys, columns 1~n are channel data.
                    n = 4 if attribute.endswith("quaternion") else 3
                    arr = np.empty((len(keys), n + 1), dtype=np.float64)
                    arr.fill(np.nan)
                    arr[:, 0] = keys
                    keyframes_raw[bone_attribute] = arr
                else:
                    # Check if keys of all channels are the same.
                    arr = keyframes_raw[bone_attribute]
                    assert np.allclose(arr[:, 0], keys), (
                        f"Error: keyframes for {fcu.data_path}"
                        " has different keytimes."
                    )
                keyframes_raw[bone_attribute][:, channel + 1] = values

        # Check NaN
        for bone_attribute, arr in keyframes_raw.items():
            assert not np.any(np.isnan(arr)), (
                "Error: lack of channels for"
                f" {bone_to_data_path[bone_attribute]}"
            )

        # Reformat: in bone's order
        bone_names_keyframes = {}
        for bone_name, attribute in keyframes_raw.keys():
            bone_names_keyframes.setdefault(bone_name, []).append(attribute)
        keyframes = OrderedDict()
        for bone_name in self.bone_names:
            for attribute in bone_names_keyframes.get(bone_name, []):
                arr = keyframes_raw[(bone_name, attribute)]
                bone_attribute_str = f"{bone_name}.{attribute}"
                keyframes[bone_attribute_str] = arr
        all_keys = sorted(list(all_keys))
        return keyframes, all_keys

    def rename_bones(
        self,
        mapping: Callable[[str], Optional[str]],
        discard_namespace: bool = True,
    ) -> None:
        for pb in self.bones:
            origin_name = pb.name
            namespace, _, name = origin_name.rpartition(":")
            tgt_name = mapping(name)
            if not tgt_name:
                continue
            if not discard_namespace:
                tgt_name = f"{namespace}:{tgt_name}"
            pb.name = tgt_name
            if self.log:
                print(f"\t{origin_name:30s}\t-> {pb.name:>15s}")

    def set_to_rest_pose(self, frame: int = 0, insert_key: bool = True):
        bpy.data.scenes["Scene"].frame_set(frame)

        bpy.ops.object.mode_set(mode='OBJECT')
        for obj in bpy.data.objects:
            obj.select_set(False)
        bpy.context.view_layer.objects.active = self._armature
        self._armature.select_set(True)

        rot_identity = mathutils.Quaternion()
        for bone in self._armature.pose.bones:
            bone.rotation_quaternion = rot_identity
            if insert_key:
                if bone.rotation_mode == "QUATERNION":
                    bone.keyframe_insert(data_path='rotation_quaternion')
                else:
                    bone.keyframe_insert(data_path='rotation_euler')

        hips_bone = self._armature.pose.bones[0]
        trans_identity = mathutils.Vector()
        hips_bone.location = trans_identity
        if insert_key:
            hips_bone.keyframe_insert(data_path='location')

    def adjust_rest_location(self, target_hip_world_location: Tuple) -> None:
        """Adjust the armature's location and then set the origin at (0, 0, 0).

        Args:
             target_hip_world_location (Tuple): the global location of the hips at a standard rest pose.
                Defaults to (0, 0, 0).
        """
        armature = self._armature

        bpy.ops.object.mode_set(mode='OBJECT')
        for obj in bpy.data.objects:
            obj.select_set(False)
        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)
        hip_pose_bone = self.bones[0]

        # Adjust the location of the armature
        bpy.data.scenes["Scene"].frame_set(0)
        # [1] Adjust armature position so that the rest pose can stand on the world origin point.
        #     (The pose has no transform at the 0th frame and the hip_pose_bone has no parent.)
        hip_world_loc_before = (
            armature.matrix_world @ hip_pose_bone.matrix.to_translation()
        )
        diff_world_loc = (
            mathutils.Vector(target_hip_world_location) - hip_world_loc_before
        )
        diff_loc = armature.matrix_world.inverted() @ diff_world_loc
        # print(f"diff_world_loc      = {diff_world_loc}")
        # print(f"diff_loc            = {diff_loc}")
        armature.data.transform(mathutils.Matrix.Translation(diff_loc))

        # [2] Compensate the vertical movement of the armature.
        # --- Get keys.
        keytime = set()
        for fcu in armature.animation_data.action.fcurves:
            if fcu.data_path.split('"')[1] == hip_pose_bone.name:
                for keyframe in fcu.keyframe_points:
                    keytime.add(keyframe.co[0])
        assert (
            keytime
        ), f"No keytime found of bone_name == {hip_pose_bone.name}."
        keytime = list(keytime)
        keytime.sort()
        # --- Modify hip_pose_bone's vertical locations at each frame
        #     (except the pose of the 0th frame whose feet are already on the floor).
        vert_comp_world_loc = mathutils.Vector([0, 0, -diff_world_loc.z])
        vert_comp_arm_loc = (
            armature.matrix_world.inverted() @ vert_comp_world_loc
        )
        vert_comp_arm_mat = mathutils.Matrix.Translation(vert_comp_arm_loc)
        # print(f"vert_comp_world_loc = {vert_comp_world_loc}")
        # print(f"vert_comp_arm_loc   = {vert_comp_arm_loc}")
        for t in keytime:
            bpy.data.scenes["Scene"].frame_set(t)
            if t <= 0:
                hip_pose_bone.location = mathutils.Vector()
            else:
                hip_pose_bone.matrix = vert_comp_arm_mat @ hip_pose_bone.matrix
            hip_pose_bone.keyframe_insert(data_path='location')

        # Move the origin cursor to (0, 0, 0)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.scene.cursor.location = mathutils.Vector()
        bpy.context.scene.cursor.rotation_euler = mathutils.Euler()
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
        return

    @property
    def matrix_world(self) -> mathutils.Matrix:
        return self._armature.matrix_world

    def set_object_mode(self, mode_name: str) -> None:
        bpy.ops.object.mode_set(mode=mode_name)
        return


def adjust_armature_orientation(armature_name: str) -> None:
    armature = bpy.data.objects[armature_name]
    armature.rotation_euler = mathutils.Euler((math.radians(-90), 0, 0))

    bpy.ops.object.mode_set(mode='OBJECT')
    for obj in bpy.data.objects:
        obj.select_set(False)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    # apply rotations
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    armature.rotation_euler = mathutils.Euler((math.radians(90), 0, 0))
    return
