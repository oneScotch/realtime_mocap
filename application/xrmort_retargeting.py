# yapf: disable
import os
import json
import socket
from typing import Union
from xrmocap.data_structure.body_model import SMPLXData

from .base_application import BaseApplication

try:
    from realtime_mocap.extern.xrmort.xrmort.retarget.rt_retarget import (  # noqa: E501
        retarget_one_frame,
    )
    has_xrmort = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_xrmort = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception

# yapf: enable


class XRMoRTRetargeting(BaseApplication):

    def __init__(self,
                 output_path: str,
                 target_actor_key: str,
                 target_actor_name: str,
                 host_ip: str,
                 host_port: int,
                 target_actor_cfg: Union[str, None] = None,
                 logger=None) -> None:
        BaseApplication.__init__(self, logger=logger)
        if not has_xrmort:
            self.logger.error(import_exception)
            raise ImportError
        self.output_path = output_path
        self.target_actor_key = target_actor_key
        self.target_actor_name = target_actor_name
        self.target_actor_cfg = target_actor_cfg
        self.host_ip = host_ip
        self.host_port = host_port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def forward(self, smplx_data: SMPLXData, **kwargs):
        param_dict = smplx_data.to_param_dict()
        retarget_src_dict = {
            'betas': param_dict['betas'],
            'global_orient': param_dict['global_orient'],
            'body_pose': param_dict['body_pose'],
            'left_hand_pose': param_dict['left_hand_pose'].reshape(1, 45),
            'right_hand_pose': param_dict['right_hand_pose'].reshape(1, 45),
            'translation': param_dict['transl'],
        }
        motion_data = retarget_one_frame(
            src_smpl_x_data=retarget_src_dict,
            tgt_name2bone=None,
            src_actor_name='SMPLX',
            tgt_actor_name=self.target_actor_name,
            tgt_actor_conf=self.target_actor_cfg)
        # send to ue
        os.makedirs(self.output_path, exist_ok= True)
        output_file = self.output_path + 'smplx.txt'
        log_file = open(output_file, "a+")
        log_file.write(str(retarget_src_dict))
        log_file.write("\n")
        log_file.close()
        motion_data = {self.target_actor_key: motion_data}
        motion_data = json.dumps(motion_data).encode('UTF-16LE')
        #self.client.sendto(motion_data, (self.host_ip, self.host_port))

    def __del__(self):
        pass
