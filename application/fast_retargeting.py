# yapf: disable
import json
import socket
from xrmocap.data_structure.body_model import SMPLXData

from realtime_mocap.extern.fast_retargeting.fast_retargeting_singleton import (
    SRC_SKELETON_JSON, TGT_SKELETON_JSON, XIAOTAO_NAME_TO_BONE,
    retarget_one_frame,
)
from .base_application import BaseApplication

# yapf: enable


class FastRetargeting(BaseApplication):

    def __init__(self, host_ip: str, host_port: int, logger=None) -> None:
        BaseApplication.__init__(self, logger=logger)
        self.host_ip = host_ip
        self.host_port = host_port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def forward(self, smplx_data: SMPLXData, **kwargs):
        param_dict = smplx_data.to_param_dict()
        retarget_src_dict = {
            'body_pose': param_dict['body_pose'],
            'global_orient': param_dict['global_orient'],
            'betas': param_dict['betas'],
            'left_hand_pose': param_dict['left_hand_pose'].reshape(1, 45),
            'right_hand_pose': param_dict['right_hand_pose'].reshape(1, 45),
        }
        motion_data = retarget_one_frame(
            retarget_src_dict,
            tgt_name2bone=XIAOTAO_NAME_TO_BONE,
            src_skeleton_json=SRC_SKELETON_JSON,
            tgt_skeleton_json=TGT_SKELETON_JSON,
        )
        # send to ue
        motion_data.pop('transl')
        motion_data = {'XiaoTao': motion_data}
        motion_data = json.dumps(motion_data).encode('UTF-16LE')
        self.client.sendto(motion_data, (self.host_ip, self.host_port))

    def __del__(self):
        pass
