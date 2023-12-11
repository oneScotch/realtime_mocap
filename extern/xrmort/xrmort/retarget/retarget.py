#!/usr/bin/env python3
# -*- coding=utf-8 -*-

from typing import Dict, List, Optional, Tuple

import numpy as np
from mathutils import Matrix
from tqdm import tqdm

from xrmort.actors import get_actor
from xrmort.motion import Motion
from xrmort.skeleton import BoneInfo, SkeletonInfo


def calculate_one_bone_retargeting(
    # skeleton
    src_matrix_world: Matrix,
    tgt_matrix_world: Matrix,
    # rest pose
    mat_0_src: Matrix,
    mat_0_tgt: Matrix,
    # pose
    mat_src: Matrix,
    src2tgt_hip_align_scaling: Optional[Tuple[float, float, float]] = None,
) -> Matrix:
    """Do one bone retargeting."""
    if src2tgt_hip_align_scaling is not None:
        # consider scaling alignment
        src2tgt_scaling_mat = Matrix(
            np.eye(3) * src2tgt_hip_align_scaling
        ).to_4x4()
        src_matrix_world_inv = (
            src_matrix_world @ src2tgt_scaling_mat
        ).inverted()
    else:
        src_matrix_world_inv = src_matrix_world.inverted()
    tgt_matrix_world_inv = tgt_matrix_world.inverted()

    # * in world space, src & tgt pelvis have the save transl and rot
    #   tgt_matrix_world @ mat_tgt @ (tgt_matrix_world @ mat_0_tgt).inverted() \
    #   == \
    #   src_matrix_world @ mat_src @ (src_matrix_world @ mat_0_src).inverted()
    mat_tgt = (
        tgt_matrix_world_inv
        @ src_matrix_world
        @ mat_src
        @ mat_0_src.inverted()
        @ src_matrix_world_inv
        @ tgt_matrix_world
        @ mat_0_tgt
    )
    return mat_tgt


def calculate_bone_matrix(
    matrix_record: Dict[str, Matrix],
    matrix_basis_record: Dict[str, Matrix],
    bone_info: BoneInfo,
) -> Matrix:
    """Calculate the pose_bone's matrix (bone2arm).
    Update matrix_record.
    """
    bone_name = bone_info.bone_name
    if bone_name in matrix_record:
        return matrix_record[bone_name]

    local = bone_info.matrix_local
    basis = matrix_basis_record[bone_name]
    # if bone_info.is_pose_bone:
    #     basis = matrix_basis_record[bone_name]
    # else:
    #     # If it has no pose, the matrix basis won't change
    #     basis = bone_info.matrix_basis
    parent = bone_info.parent
    if parent is None:
        mat = local @ basis
    else:
        parent_local = parent.matrix_local
        mat = (
            calculate_bone_matrix(matrix_record, matrix_basis_record, parent)
            @ parent_local.inverted()
            @ local
            @ basis
        )
    return mat


def do_retargeting(
    src_motion: Motion,
    src_skeleton: SkeletonInfo,
    tgt_skeleton: SkeletonInfo,
    src2tgt_bone_names: Dict[str, str],
    src2tgt_scaling: float = 1,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
) -> Tuple[List[Dict[str, Dict[str, List[float]]]], int, int]:
    """Do skeleton retargeting

    Returns:
        (List[Dict[str, Dict[str, List[float]]]], int, int):
            compose of (motion_data, frame_start, frame_end).

            motion_data: e.g.
                [
                    {
                        'pelvis': {
                            'rotation': (1, 0, 0, 0),
                            'location': (0, 0, 0),
                        },
                        'hip_l': {
                            'rotation': (1, 0, 0, 0),
                        }
                    }
                ]
            frame_start: the first frame (included)
            frame_end: the last frame (included)
    """
    src_matrix_world = src_skeleton.armature_matrix
    tgt_matrix_world = tgt_skeleton.armature_matrix
    n_frames = src_motion.n_frames

    # Initialize bone matrix_basis
    matrix_basis_src_record: Dict[str, Matrix] = {
        name: bone.matrix_basis for name, bone in src_skeleton.bones.items()
    }
    matrix_basis_tgt_record: Dict[str, Matrix] = {
        name: bone.matrix_basis for name, bone in tgt_skeleton.bones.items()
    }
    motion_data: List[Dict[str, Dict[str, List[float]]]] = []

    # Get frame scope
    frame_range = [
        f
        for f in range(frame_start, n_frames)
        if frame_end is None or f < frame_end
    ]
    if 0 not in frame_range:
        frame_range.insert(0, 0)
    frame_end = len(frame_range) + frame_start

    # Do retargeting
    frames_iter = tqdm(frame_range, desc="retarget", miniters=60)
    for frame in frames_iter:
        matrix_src_record: Dict[str, Matrix] = {}
        matrix_tgt_record: Dict[str, Matrix] = {}

        frame_motion_data = {}
        for src_bone_name in src_skeleton.bone_names:
            # ! bone_names' order make sure parents are visited before children
            tgt_bone_name = src2tgt_bone_names.get(src_bone_name)
            if not tgt_bone_name:
                if frame < 1:
                    print(f"[SKIPPED] {src_bone_name}\t")
                continue
            else:
                if frame < 1:
                    assert tgt_bone_name in tgt_skeleton.bones, (
                        f"{tgt_bone_name} not in bones "
                        f"{list(tgt_skeleton.bones.keys())}"
                    )
                    # print(f"{src_bone_name}\t->\t{tgt_bone_name}")

            src_bone = src_skeleton.bones[src_bone_name]
            tgt_bone = tgt_skeleton.bones[tgt_bone_name]
            is_pelvis = src_bone.is_pelvis
            # * pose2rest
            if src_bone_name in src_motion.BONE_NAMES:
                mat_basis_src = Matrix(
                    src_motion.get_bone_matrix_basis(src_bone_name, frame)
                )
            else:
                mat_basis_src = src_bone.matrix_basis
            matrix_basis_src_record[src_bone_name] = mat_basis_src
            # * bone2armature with pose
            mat_src = calculate_bone_matrix(
                matrix_src_record, matrix_basis_src_record, src_bone
            )
            # * restBone2posedParent
            if tgt_bone.parent is None:
                tgt_local_mat = tgt_bone.matrix_local
            else:
                # bone2armature of the parent
                tgt_parent_matrix = calculate_bone_matrix(
                    matrix_tgt_record, matrix_basis_tgt_record, tgt_bone.parent
                )
                tgt_local_mat = (
                    tgt_parent_matrix
                    @ tgt_bone.parent.matrix_local.inverted()
                    @ tgt_bone.matrix_local
                )
            # * bone2armature at the 0-th frame
            mat_0_src = src_bone.matrix
            mat_0_tgt = tgt_bone.matrix

            # [*] Retargeting
            mat_tgt = calculate_one_bone_retargeting(
                src_matrix_world=src_matrix_world,
                tgt_matrix_world=tgt_matrix_world,
                mat_0_src=mat_0_src,
                mat_0_tgt=mat_0_tgt,
                mat_src=mat_src,
            )
            mat_basis_tgt = tgt_local_mat.inverted() @ mat_tgt
            matrix_basis_tgt_record[tgt_bone_name] = mat_basis_tgt

            # [*] Record the result
            loc_, quat_, _ = mat_basis_tgt.decompose()
            transform = frame_motion_data.setdefault(tgt_bone_name, {})
            transform["rotation"] = list(quat_)
            if is_pelvis:
                transform["location"] = list(loc_ * src2tgt_scaling)
        motion_data.append(frame_motion_data)
    return motion_data, frame_start, frame_end


def retarget_motion_between_actors(
    src_motion: Motion,
    src_actor_name: str,
    src_actor_conf: str = '',
    tgt_actor_name: str = 'ybot',
    tgt_actor_conf: str = '',
    frame_start: int = 0,
    frame_end: Optional[int] = None,
) -> Tuple[List[Dict[str, Dict[str, List[float]]]], int, int]:
    """Do skeleton retargeting.

    Parameters
    ----------
    src_motion : Motion
        Source motion data.
    src_actor_name : str
        Source actor's name.
    src_actor_conf : str, optional
        Path to the source actor's `actor.json` config file. Required only
        when src_actor_name == "dynamic". By default ''
    tgt_actor_name : str, optional
        Target actor's name, by default 'ybot'
    tgt_actor_conf : str, optional
        Path to the target actor's `actor.json` config file. Required only
        when tgt_actor_name == "dynamic". By default ''
    frame_start : int, optional
        The first frame to do retargeting, by default 0
    frame_end : Optional[int], optional
        The last frame to do retargeting (not included). If `None`, frames
        to the end will be included. By default None.

    Returns
    -------
    Tuple[List[Dict[str, Dict[str, List[float]]]], int, int]
        tuple of the following contents:
        1. motion_data: e.g.
            [
                {
                    'pelvis': {
                        'rotation': (1, 0, 0, 0),
                        'location': (0, 0, 0),
                    },
                    'hip_l': {
                        'rotation': (1, 0, 0, 0),
                    }
                }
            ]
        2. frame_start: the first frame (included)
        3. frame_end: the last frame (included)
    """
    src_actor = get_actor(src_actor_name, src_actor_conf)
    assert src_actor.skeleton_path is not None, f"{src_actor_name}"
    tgt_actor = get_actor(tgt_actor_name, tgt_actor_conf)
    assert tgt_actor.skeleton_path is not None, f"{tgt_actor_name}"

    src_skeleton = SkeletonInfo.from_json(src_actor.skeleton_path)
    tgt_skeleton = SkeletonInfo.from_json(tgt_actor.skeleton_path)
    src2tgt_scaling = tgt_skeleton.pelvis_height / src_skeleton.pelvis_height
    src2tgt_bone_names = {
        bone: src_actor.to_actor_bone(tgt_actor, bone)
        for bone in src_actor.bone_to_name_mapping.keys()
    }

    motion_data, frame_start, frame_end = do_retargeting(
        src_motion,
        src_skeleton,
        tgt_skeleton,
        src2tgt_bone_names=src2tgt_bone_names,
        src2tgt_scaling=src2tgt_scaling,
        frame_start=frame_start,
        frame_end=frame_end,
    )
    return motion_data, frame_start, frame_end


if __name__ == '__main__':
    import argparse
    import time

    from xrmort.motion import SMPLMotion

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-smpl", type=str, help="Input smpl file")
    args = parser.parse_args()

    smpl_data = np.load(args.input_smpl, allow_pickle=True)['smpl'].item()
    src_motion = SMPLMotion.from_smpl_data(smpl_data, insert_rest_pose=True)

    ctime = time.time()
    motion_data = retarget_motion_between_actors(
        src_motion=src_motion,
        src_actor_name="SMPL",
        tgt_actor_name="ybot",
    )
    print("cost time:", time.time() - ctime)
    print(motion_data)
