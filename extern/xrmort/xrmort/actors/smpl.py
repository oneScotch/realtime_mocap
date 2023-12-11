#!/usr/bin/env python3
# -*- coding=utf-8 -*-
from .base_actor import BaseActor, get_actor_path, get_skeleton_path


class SMPL(BaseActor):
    actor_path = get_actor_path("SMPLX_male.fbx")
    skeleton_path = get_skeleton_path("smplx_male_T.json")

    name_to_bone_mapping = {
        "Ankle_L": "left_ankle",
        "Ankle_R": "right_ankle",
        "Chest": "spine3",
        "Elbow_L": "left_elbow",
        "Elbow_R": "right_elbow",
        "Head": "head",
        "Hip_L": "left_hip",
        "Hip_R": "right_hip",
        "Knee_L": "left_knee",
        "Knee_R": "right_knee",
        "Neck": "neck",
        "Pelvis": "pelvis",
        "Scapula_L": "left_collar",
        "Scapula_R": "right_collar",
        "Shoulder_L": "left_shoulder",
        "Shoulder_R": "right_shoulder",
        "Spine1": "spine1",
        "Spine2": "spine2",
        "SpinePattern": "",
        "Toes_L": "left_foot",
        "Toes_R": "right_foot",
        "Wrist_L": "left_wrist",
        "Wrist_R": "right_wrist",
        "index_A_L": "left_index1",
        "index_A_R": "right_index1",
        "index_B_L": "left_index2",
        "index_B_R": "right_index2",
        "index_C_L": "left_index3",
        "index_C_R": "right_index3",
        "middle_A_L": "left_middle1",
        "middle_A_R": "right_middle1",
        "middle_B_L": "left_middle2",
        "middle_B_R": "right_middle2",
        "middle_C_L": "left_middle3",
        "middle_C_R": "right_middle3",
        "pinky_A_L": "left_pinky1",
        "pinky_A_R": "right_pinky1",
        "pinky_B_L": "left_pinky2",
        "pinky_B_R": "right_pinky2",
        "pinky_C_L": "left_pinky3",
        "pinky_C_R": "right_pinky3",
        "ring_A_L": "left_ring1",
        "ring_A_R": "right_ring1",
        "ring_B_L": "left_ring2",
        "ring_B_R": "right_ring2",
        "ring_C_L": "left_ring3",
        "ring_C_R": "right_ring3",
        "thumb_A_L": "left_thumb1",
        "thumb_A_R": "right_thumb1",
        "thumb_B_L": "left_thumb2",
        "thumb_B_R": "right_thumb2",
        "thumb_C_L": "left_thumb3",
        "thumb_C_R": "right_thumb3"
    }

    bone_to_name_mapping = {v: k for k, v in name_to_bone_mapping.items() if k and v}
