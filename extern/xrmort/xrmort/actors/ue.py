#!/usr/bin/env python3
# -*- coding=utf-8 -*-
from .base_actor import BaseActor, get_actor_path, get_skeleton_path


class Mannequin(BaseActor):
    actor_path = get_actor_path("SK_Mannequin.fbx")
    skeleton_path = get_skeleton_path("Mannequin_T.json")

    name_to_bone_mapping = {
        "Ankle_L": "foot_l",
        "Ankle_R": "foot_r",
        "Chest": "spine_03",
        "Elbow_L": "lowerarm_l",
        "Elbow_R": "lowerarm_r",
        "Head": "head",
        "Hip_L": "thigh_l",
        "Hip_R": "thigh_r",
        "Knee_L": "calf_l",
        "Knee_R": "calf_r",
        "Neck": "neck_01",
        "Pelvis": "pelvis",
        "Scapula_L": "clavicle_l",
        "Scapula_R": "clavicle_r",
        "Shoulder_L": "upperarm_l",
        "Shoulder_R": "upperarm_r",
        "Spine1": "spine_01",
        "Spine2": "spine_02",
        "SpinePattern": "",
        "Toes_L": "ball_l",
        "Toes_R": "ball_r",
        "Wrist_L": "hand_l",
        "Wrist_R": "hand_r",
        "index_A_L": "index_01_l",
        "index_A_R": "index_01_r",
        "index_B_L": "index_02_l",
        "index_B_R": "index_02_r",
        "index_C_L": "index_03_l",
        "index_C_R": "index_03_r",
        "middle_A_L": "middle_01_l",
        "middle_A_R": "middle_01_r",
        "middle_B_L": "middle_02_l",
        "middle_B_R": "middle_02_r",
        "middle_C_L": "middle_03_l",
        "middle_C_R": "middle_03_r",
        "pinky_A_L": "pinky_01_l",
        "pinky_A_R": "pinky_01_r",
        "pinky_B_L": "pinky_02_l",
        "pinky_B_R": "pinky_02_r",
        "pinky_C_L": "pinky_03_l",
        "pinky_C_R": "pinky_03_r",
        "ring_A_L": "ring_01_l",
        "ring_A_R": "ring_01_r",
        "ring_B_L": "ring_02_l",
        "ring_B_R": "ring_02_r",
        "ring_C_L": "ring_03_l",
        "ring_C_R": "ring_03_r",
        "thumb_A_L": "thumb_01_l",
        "thumb_A_R": "thumb_01_r",
        "thumb_B_L": "thumb_02_l",
        "thumb_B_R": "thumb_02_r",
        "thumb_C_L": "thumb_03_l",
        "thumb_C_R": "thumb_03_r"
    }
    bone_to_name_mapping = {v: k for k, v in name_to_bone_mapping.items() if k and v}


class MannequinYUp(Mannequin):
    name_to_bone_mapping = {
        "Ankle_L": "foot_l",
        "Ankle_R": "foot_r",
        "Chest": "spine_03",
        "Elbow_L": "lowerarm_l",
        "Elbow_R": "lowerarm_r",
        "Head": "head",
        "Hip_L": "upperleg_l",#thigh_l
        "Hip_R": "upperleg_r",
        "Knee_L": "lowerleg_l",#calf_l
        "Knee_R": "lowerleg_r",
        "Neck": "neck",#neck_01
        "Pelvis": "hip",# pelvis
        "Scapula_L": "shoulder_l",#clavicle_l
        "Scapula_R": "shoulder_r",
        "Shoulder_L": "upperarm_l",
        "Shoulder_R": "upperarm_r",
        "Spine1": "spine_01",
        "Spine2": "spine_02",
        "SpinePattern": "",
        "Toes_L": "ball_l",
        "Toes_R": "ball_r",
        "Wrist_L": "hand_l",
        "Wrist_R": "hand_r",
        "index_A_L": "index_01_l",
        "index_A_R": "index_01_r",
        "index_B_L": "index_02_l",
        "index_B_R": "index_02_r",
        "index_C_L": "index_03_l",
        "index_C_R": "index_03_r",
        "middle_A_L": "middle_01_l",
        "middle_A_R": "middle_01_r",
        "middle_B_L": "middle_02_l",
        "middle_B_R": "middle_02_r",
        "middle_C_L": "middle_03_l",
        "middle_C_R": "middle_03_r",
        "pinky_A_L": "pinky_01_l",
        "pinky_A_R": "pinky_01_r",
        "pinky_B_L": "pinky_02_l",
        "pinky_B_R": "pinky_02_r",
        "pinky_C_L": "pinky_03_l",
        "pinky_C_R": "pinky_03_r",
        "ring_A_L": "ring_01_l",
        "ring_A_R": "ring_01_r",
        "ring_B_L": "ring_02_l",
        "ring_B_R": "ring_02_r",
        "ring_C_L": "ring_03_l",
        "ring_C_R": "ring_03_r",
        "thumb_A_L": "thumb_01_l",
        "thumb_A_R": "thumb_01_r",
        "thumb_B_L": "thumb_02_l",
        "thumb_B_R": "thumb_02_r",
        "thumb_C_L": "thumb_03_l",
        "thumb_C_R": "thumb_03_r"
    }
    bone_to_name_mapping = {v: k for k, v in name_to_bone_mapping.items() if k and v}
