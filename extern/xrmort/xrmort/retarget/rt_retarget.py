"""Retarget one motion frame.

Optimized for realtime pipeline.

## Usage:

(1.1) Call `retarget_one_frame` for realtime retargeting.

Example:

```python
smplx_data = {
    'body_pose': ...,       # np.ndarray of shape (1, 63)
    'global_orient': ...,   # np.ndarray of shape (1, 3)
    'betas': ...,           # np.ndarray of shape (1, 10)
    'left_hand_pose': ...,  # np.ndarray of shape (1, 15)
    'right_hand_pose': ..., # np.ndarray of shape (1, 15)
    # other keys are optional
    ...
}

motion_data = retarget_one_frame(
    src_smpl_x_data=smplx_data,
    tgt_actor_name=actor_name,
    tgt_actor_conf=tgt_actor_conf,
)
```

(1.2) Try this script via:

```bash
# retarget from smplx to a preset actor
python rt_retarget.py --smplx xxx.npz --actor-name ybot

# retarget from smplx to a dynamic acto
python rt_retarget.py --smplx xxx.npz --actor-conf /path/to/actor.json
```

"""
import json
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mathutils import Matrix, Quaternion, Vector
from scipy.spatial.transform import Rotation as spRotation

from xrmort.actors import BaseActor, get_actor
from xrmort.actors.constants import FINGERS
from xrmort.motion import Motion, SMPLMotion, SMPLXMotion
from xrmort.skeleton import SkeletonInfo

from .retarget import calculate_bone_matrix, calculate_one_bone_retargeting

DEFAULT_BONE_ORDER_CONF = (
    Path(__file__).absolute().parent / "bone_order-example.json"
)


def retarget_one_frame(
    src_smpl_x_data: Dict[str, np.ndarray],
    frame: int = 0,
    tgt_name2bone: Optional[Dict[str, str]] = None,
    src_actor_name: str = 'SMPL',
    src_actor_conf: str = '',
    src_pelvis_height: Optional[float] = None,
    tgt_actor_name: str = 'ybot',
    tgt_actor_conf: str = '',
    tgt_pelvis_height: Optional[float] = None,
    tgt_bone_order_conf: Optional[Union[str, Path]] = DEFAULT_BONE_ORDER_CONF,
    has_transl: bool = False,
) -> Dict[str, Union[Tuple[float, float, float], str]]:
    """Do skeleton retargeting.

    Args:
        src_smpl_x_data (Dict[str, np.ndarray]):
        source motion data in SMPL/SMPLX.
            e.g. for SMPL:
            {
                'body_pose': body_pose,
                'global_orient': global_orient,
                'betas': smpl_beta,
                # other keys are optional
            }
            For SMPLX:
            {
                'body_pose': body_pose,
                'global_orient': global_orient,
                'betas': smpl_beta,
                'left_hand_pose': left_hand_pose,
                'right_hand_pose': right_hand_pose,
                # other keys are optional
            }
        frame (int):
            the frame of source motion to retarget, defaults to 0.
        src_actor_name (str):
            source actor's name. Defaults to "SMPL"
        src_actor_conf (optional, Path):
            path to the source actor's `actor.json` config file. Required only
            when src_actor_name == "dynamic". Defaults to "".
        src_pelvis_height (Optional[float], optional): provide the height of
            the source skeleton's pelvis joint to overwrite the pre-defined
            value. Defaults to None.
        tgt_actor_name (str):
            target actor's name. Defaults to "ybot"
        tgt_actor_conf (optional, Path):
            path to the target actor's `actor.json` config file. Required only
            when tgt_actor_name == "dynamic". Defaults to "".
        tgt_bone_order_conf: Union[str, Path] = path to a json file
            containing a list of target bone names. Bones will following that
            order in the output. If empty, the output will follows the same
            order that the source skeleton's bones appear. If None,
            the bones order will not be fixed.
            Defaults to DEFAULT_BONE_ORDER_CONF.
        tgt_pelvis_height (Optional[float], optional): overwrite the height of
            the target skeleton's pelvis joint to overwrite the pre-defined
            value. Defaults to None.
        has_transl (bool, optional): whether to consider translations,
            default to False.

    Returns:
        Dict[str, Dict[str, Union[Tuple[float, float, float], str]]]:
            target motion data in UE space
            e.g.
            {
                "MXJ_walk": {
                    "Ankle_L": [0.0, 0.0, 0.0],
                    "Ankle_R": [0.0, 0.0, 0.0],
                    "Chest_M": [0.0, 0.0, 0.0],
                    # # "joint's name": [euler1, euler2, euler3],
                    ...

                    # # ignore:
                    # "_root_name": "_tex_Char_MXJHM_Rig_publish:Root_M"
                    # "_root_translate": [0.0, 0.0, 0.0]
                }
            }
    """
    if "left_hand_pose" in src_smpl_x_data:
        src_motion = SMPLXMotion.from_smplx_data(
            src_smpl_x_data, insert_rest_pose=False, flat_hand_mean=True
        )
    else:
        src_motion = SMPLMotion.from_smpl_data(
            src_smpl_x_data, insert_rest_pose=False
        )

    src_actor, src_skeleton = _get_actor_skeleton(
        src_actor_name,
        src_actor_conf,
        src_pelvis_height,
    )
    tgt_actor, tgt_skeleton = _get_actor_skeleton(
        tgt_actor_name,
        tgt_actor_conf,
        tgt_pelvis_height,
    )
    src2tgt_hip_align_scaling = (
        tgt_skeleton.pelvis_height / src_skeleton.pelvis_height
    )
    src2tgt = _src2tgt_bone_names(src_actor, tgt_actor)

    optimize_thumbs = __thumb_bones(tgt_actor)
    optimize_scapulas = __scapula_bones(tgt_actor)
    optimize_fingers_limit = __finger_bones_except_thumbs(tgt_actor)
    tgt_bone_order_conf = (
        str(tgt_bone_order_conf) if tgt_bone_order_conf else None
    )
    smplx_motion_data = _do_retargeting_one_for_ue(
        src_motion=src_motion,
        frame=frame,
        src_skeleton=src_skeleton,
        tgt_skeleton=tgt_skeleton,
        src2tgt_bone_names=src2tgt,
        src2tgt_scaling=src2tgt_hip_align_scaling,
        optimize_thumbs=optimize_thumbs,
        optimize_scapulas=optimize_scapulas,
        optimize_finger_limit=optimize_fingers_limit,
        tgt_bone_order_conf=tgt_bone_order_conf,
        has_transl=has_transl,
    )
    return smplx_motion_data


# =========================
# Based on `motion_utils.retarget.retarget.do_retarget`,
#  and optimized for realtime running.
# =========================
def _do_retargeting_one_for_ue(
    src_motion: Motion,
    frame: int,
    src_skeleton: SkeletonInfo,
    tgt_skeleton: SkeletonInfo,
    src2tgt_bone_names: Dict[str, str],
    src2tgt_scaling: float = 1,
    # frame_start: int = 0,
    # frame_end: Optional[int] = None,
    extra_options: Optional[Dict] = None,
    optimize_thumbs: Optional[List[str]] = None,
    optimize_scapulas: Optional[List[str]] = None,
    optimize_finger_limit: Optional[Dict[str, int]] = None,
    tgt_bone_order_conf: Optional[str] = None,
    has_transl: bool = False,
) -> Dict[str, Union[Tuple[float, float, float], str]]:
    """Do skeleton retargeting

    Returns:
        Dict[str, Union[Tuple[float, float, float], str]]: motion_data, e.g.
            {
                "Ankle_L": [0.0, 0.0, 0.0],
                "Ankle_R": [0.0, 0.0, 0.0],
                "Chest_M": [0.0, 0.0, 0.0],
                # # "joint's name": [euler1, euler2, euler3],
                ...

                # # ignored if has_transl = False
                # "_root_name": "_tex_Char_MXJHM_Rig_publish:Root_M"
                # "_root_translate": [0.0, 0.0, 0.0]
            }
        # ! the target motion data is in UE space
    """
    extra_options = extra_options or {}
    src2tgt_hip_align_scaling = (
        src2tgt_scaling,
        src2tgt_scaling,
        src2tgt_scaling,
    )
    src_matrix_world = src_skeleton.armature_matrix
    tgt_matrix_world = tgt_skeleton.armature_matrix
    # n_frames = src_motion.n_frames

    # Initialize bone matrix_basis
    matrix_basis_src_record: Dict[str, Matrix] = {
        name: bone.matrix_basis for name, bone in src_skeleton.bones.items()
    }
    matrix_basis_tgt_record: Dict[str, Matrix] = {
        name: bone.matrix_basis for name, bone in tgt_skeleton.bones.items()
    }
    # motion_data: List[Dict[str, Dict[str, List[float]]]] = []
    motion_data: Dict[str, Union[Tuple[float, float, float], str]] = {}

    # # Get frame scope
    # frame_range = [
    #     f for f in range(frame_start, n_frames)
    #     if frame_end is None or f < frame_end
    # ]
    # if 0 not in frame_range:
    #     frame_range.insert(0, 0)
    # frame_end = len(frame_range) + frame_start

    # Do retargeting
    # for frame in tqdm(frame_range, desc="retarget", miniters=60):
    for f_ in (frame,):
        matrix_src_record: Dict[str, Matrix] = {}
        matrix_tgt_record: Dict[str, Matrix] = {}

        frame_motion_data = _default_motion_data(
            tgt_bone_order_conf,
            has_transl=has_transl,
            root_name=tgt_skeleton.pelvis_name,
        ).copy()
        for src_bone_name in src_skeleton.bone_names:
            # ! bone_names' order make sure parents are visited before children
            tgt_bone_name = src2tgt_bone_names.get(src_bone_name)
            if not tgt_bone_name:
                continue
            else:
                assert tgt_bone_name in tgt_skeleton.bones, (
                    f"{tgt_bone_name} not in bones"
                    f" {list(tgt_skeleton.bones.keys())}"
                )

            src_bone = src_skeleton.bones[src_bone_name]
            tgt_bone = tgt_skeleton.bones[tgt_bone_name]
            # is_pelvis = src_bone.is_pelvis
            # * pose2rest
            if src_bone_name in src_motion.BONE_NAMES:
                mat_basis_src = Matrix(
                    src_motion.get_bone_matrix_basis(src_bone_name, f_)
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
                src2tgt_hip_align_scaling=src2tgt_hip_align_scaling,
            )
            mat_basis_tgt = tgt_local_mat.inverted() @ mat_tgt
            # Optimize steps:
            if optimize_thumbs and tgt_bone_name in optimize_thumbs:
                # mitigate the rotation of the first thumb joint
                # to produce more natural finger poses
                mat_basis_tgt = mat_basis_mitigate_rotation(
                    mat_basis_tgt, ratio=0.3
                )
            elif optimize_scapulas and tgt_bone_name in optimize_scapulas:
                # improve shoulder poses
                mat_basis_tgt = mat_basis_mitigate_rotation(
                    mat_basis_tgt, ratio=0.15
                )
            matrix_basis_tgt_record[tgt_bone_name] = mat_basis_tgt

            # [*] Record the result
            if tgt_bone_order_conf and tgt_bone_name not in frame_motion_data:
                print(
                    f'[!] Warning: tgt_bone_name="{tgt_bone_name}"'
                    ' not in tgt_bone_order_conf'
                )
            loc, quat, _scale = mat_basis_tgt.decompose()
            if optimize_finger_limit and tgt_bone_name in optimize_finger_limit:
                fingers_post_process(
                    quat, optimize_finger_limit[tgt_bone_name], ratio=1
                )
            # transform = frame_motion_data.setdefault(tgt_bone_name, {})
            # transform["rotation"] = list(Q)
            # if is_pelvis:
            #     transform["location"] = list(T)
            frame_motion_data[tgt_bone_name] = _bl_quat_to_ue(quat)
            if has_transl and tgt_bone_name == tgt_skeleton.pelvis_name:
                frame_motion_data["_root_translate"] = _bl_vector_to_ue(loc)

        # motion_data.append(frame_motion_data)
        motion_data = frame_motion_data
    return motion_data


def mat_basis_mitigate_rotation(
    mat_basis_tgt,
    ratio=0.3,
):
    """Calculate a new matrix whose rotation angle is multiplied with ratio."""
    mat_basis_tgt_ndarray = np.asarray(mat_basis_tgt)
    mat_basis_tgt_ndarray[:3, :3] = spRotation.from_rotvec(
        spRotation.from_matrix(mat_basis_tgt_ndarray[:3, :3]).as_rotvec()
        * ratio
    ).as_matrix()
    mat_basis_tgt = Matrix(mat_basis_tgt_ndarray)
    return mat_basis_tgt


def fingers_post_process(
    quat: Quaternion, finger_idx: int, ratio=1.15
) -> Quaternion:
    """Optimize finger pose performance."""
    # euler in yxz: [pitch, yaw, roll]
    euler = spRotation.from_quat(
        [quat.x, quat.y, quat.z, quat.w]
    ).as_euler("yxz").tolist()
    # increase pitch rotation and only pitch rotation
    if finger_idx == 1:
        euler = [euler[0] * ratio, euler[1], euler[2]]
    elif finger_idx == 2:
        euler = [max(euler[0], 0) * ratio, 0, 0]
    elif finger_idx == 3:
        euler = [max(euler[0], 0), 0, 0]
    quat = spRotation.from_euler("yxz", [euler[0], 0, 0]).as_quat()
    return Quaternion([quat[3], quat[0], quat[1], quat[2]])


@lru_cache(maxsize=None)
def _get_actor_skeleton(
    actor_name: str, actor_conf: str = '', pelvis_height: Optional[float] = None
) -> Tuple[BaseActor, SkeletonInfo]:
    """actor name to SkeletonInfo instance.

    Args:
        actor_name (str): actor's name
        actor_conf (str): path to `actor.json` file.
            Required when actor_name == "dynamic". Defaults to "".
        pelvis_height (Optional[float], optional): overwrite the height of
            the pelvis joint. Defaults to None.

    Returns:
        BaseActor:
        SkeletonInfo:
    """
    actor = get_actor(actor_name, actor_conf=actor_conf)
    if not actor.skeleton_path:
        raise ValueError(f"actor_name={actor_name}")
    skeleton_info = SkeletonInfo.from_json(actor.skeleton_path)
    if pelvis_height:
        skeleton_info.pelvis_height = pelvis_height
    return actor, skeleton_info


def _bl_quat_to_ue(quat: Quaternion) -> Tuple[float, float, float]:
    """From blender to ue.
    Only suitable for matrix in world space or armature space.
    Very strange but UE4's euler angles around x/y axis follow
    the right-handed convention, and the z axis angles follow
    the left-handed convention.
    """
    rot = spRotation.from_quat([quat.x, quat.y, quat.z, quat.w])  # type: ignore
    euler = rot.as_euler('xyz', degrees=True)
    # bl to ue rotation conversion: z -> -z
    euler = (euler[0], -euler[1], -euler[2])
    return euler


def _bl_vector_to_ue(vec: Vector) -> Tuple[float, float, float]:
    """From blender to ue. (y -> -y)"""
    return (vec[0], -vec[1], vec[2])


##################
# Cached intermediate data functions
##################
@lru_cache(maxsize=None)
def _src2tgt_bone_names(
    src_actor: BaseActor, tgt_actor: BaseActor
) -> Dict[str, str]:
    return {
        bone: src_actor.to_actor_bone(tgt_actor, bone)
        for bone in src_actor.bone_to_name_mapping.keys()
    }


@lru_cache(maxsize=None)
def _default_motion_data(
    bone_order_json: Optional[str] = None,
    has_transl: bool = False,
    root_name: str = '',
) -> OrderedDict:
    if not bone_order_json:
        data = OrderedDict()
    else:
        data = OrderedDict(
            (b, (0, 0, 0)) for b in __load_tgt_bones_order(bone_order_json)
        )
    if has_transl:
        data["_root_name"] = root_name
        data["_root_translate"] = (0, 0, 0)
    return data


@lru_cache(maxsize=None)
def __load_tgt_bones_order(bone_order_json: str) -> tuple:
    with open(bone_order_json, "r") as f:
        bones = json.load(f)
    return tuple(bones)


@lru_cache(maxsize=None)
def __thumb_bones(tgt_actor: BaseActor) -> List[str]:
    res = [
        tgt_actor.bone2name("thumb_A_L"),
        tgt_actor.bone2name("thumb_A_R"),
    ]
    res = [x for x in res if x]
    return res


@lru_cache(maxsize=None)
def __finger_bones_except_thumbs(tgt_actor: BaseActor) -> Dict[str, int]:
    res = OrderedDict()
    for x in FINGERS:
        if x.startswith("thumb"):
            continue
        bone = tgt_actor.bone2name(x)
        if not bone:
            continue
        if "_A_" in x:
            idx = 1
        elif "_B_" in x:
            idx = 2
        elif "_C_" in x:
            idx = 3
        else:
            raise ValueError(f"Unknown finger: {x}")
        res[x] = idx
    return res


@lru_cache(maxsize=None)
def __scapula_bones(tgt_actor: BaseActor) -> List[str]:
    res = [
        tgt_actor.bone2name("Scapula_L"),
        tgt_actor.bone2name("Scapula_R"),
    ]
    res = [x for x in res if x]
    return res


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--smplx', type=str, help='SMPLX humandata file path.')
    parser.add_argument(
        '--actor-name',
        type=str,
        default='dynamic',
        help='name of the target actor.',
    )
    parser.add_argument(
        '--actor-conf',
        type=str,
        default='',
        help=(
            'Path of a directory which includes actor config (actor.json).'
            'Required if `--actor-name` is dynamic'
        ),
    )
    parser.add_argument(
        '--bone-order-conf',
        type=str,
        default='',
        help=(
            'Path of a json file defining a list of bone names. '
            'Bones in the result will follow that order.'
        ),
    )
    args = parser.parse_args()

    if args.smplx:
        humandata = np.load(args.smplx, allow_pickle=True)
        smplx_data = humandata['smplx'].item()

        if args.actor_name.lower() == 'dynamic' and not args.actor_conf:
            raise ValueError(
                "--actor-config if required when"
                f" --actor-name={args.actor_name}"
            )

        motion_data = retarget_one_frame(
            src_smpl_x_data=smplx_data,
            tgt_actor_name=args.actor_name,
            tgt_actor_conf=args.actor_conf,
            tgt_bone_order_conf=args.bone_order_config,
        )
        print(json.dumps(motion_data, indent=2))
