# yapf: disable
import numpy as np

from realtime_mocap.utils.ik_utils import (
    rotation_global2local, rotation_local2global,
)

# yapf: enable

MANO_PARENTS = [
    -1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15
][:16]
FLIP_MAT = np.eye(3)
FLIP_MAT[0, 0] = -1


def flip_rotmat(local_rotmat):
    global_rotmat = rotation_local2global(
        rot_mats=local_rotmat, parents=MANO_PARENTS)
    for joint_index in range(len(global_rotmat)):
        global_rotmat[
            joint_index] = FLIP_MAT @ global_rotmat[joint_index] @ FLIP_MAT
    local_rotmat = rotation_global2local(global_rotmat, parents=MANO_PARENTS)
    return local_rotmat
