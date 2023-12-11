#!/usr/bin/env python3
# -*- coding=utf-8 -*-
from .base_actor import BaseActor, get_actor_path, get_skeleton_path


class YBot(BaseActor):
    actor_path = get_actor_path("ybot.fbx")
    skeleton_path = get_skeleton_path("ybot_T.json")

    name_to_bone_mapping = {
        "Ankle_L": "LeftFoot",
        "Ankle_R": "RightFoot",
        "Chest": "Spine2",
        "Elbow_L": "LeftForeArm",
        "Elbow_R": "RightForeArm",
        "Head": "Head",
        "Hip_L": "LeftUpLeg",
        "Hip_R": "RightUpLeg",
        "Knee_L": "LeftLeg",
        "Knee_R": "RightLeg",
        "Neck": "Neck",
        "Pelvis": "Hips",
        "Scapula_L": "LeftShoulder",
        "Scapula_R": "RightShoulder",
        "Shoulder_L": "LeftArm",
        "Shoulder_R": "RightArm",
        "Spine1": "Spine",
        "Spine2": "Spine1",
        "Toes_L": "LeftToeBase",
        "Toes_R": "RightToeBase",
        "Wrist_L": "LeftHand",
        "Wrist_R": "RightHand",
        "index_A_L": "LeftHandIndex1",
        "index_A_R": "RightHandIndex1",
        "index_B_L": "LeftHandIndex2",
        "index_B_R": "RightHandIndex2",
        "index_C_L": "LeftHandIndex3",
        "index_C_R": "RightHandIndex3",
        "middle_A_L": "LeftHandMiddle1",
        "middle_A_R": "RightHandMiddle1",
        "middle_B_L": "LeftHandMiddle2",
        "middle_B_R": "RightHandMiddle2",
        "middle_C_L": "LeftHandMiddle3",
        "middle_C_R": "RightHandMiddle3",
        "pinky_A_L": "LeftHandPinky1",
        "pinky_A_R": "RightHandPinky1",
        "pinky_B_L": "LeftHandPinky2",
        "pinky_B_R": "RightHandPinky2",
        "pinky_C_L": "LeftHandPinky3",
        "pinky_C_R": "RightHandPinky3",
        "ring_A_L": "LeftHandRing1",
        "ring_A_R": "RightHandRing1",
        "ring_B_L": "LeftHandRing2",
        "ring_B_R": "RightHandRing2",
        "ring_C_L": "LeftHandRing3",
        "ring_C_R": "RightHandRing3",
        "thumb_A_L": "LeftHandThumb1",
        "thumb_A_R": "RightHandThumb1",
        "thumb_B_L": "LeftHandThumb2",
        "thumb_B_R": "RightHandThumb2",
        "thumb_C_L": "LeftHandThumb3",
        "thumb_C_R": "RightHandThumb3"
    }
    bone_to_name_mapping = {v: k for k, v in name_to_bone_mapping.items() if k and v}
