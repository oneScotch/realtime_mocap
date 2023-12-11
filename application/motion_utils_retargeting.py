# yapf: disable
import json
import socket
from xrmocap.data_structure.body_model import SMPLXData

from .base_application import BaseApplication

try:
    from realtime_mocap.extern.motion_utils.scripts.retargeting.fast_retargeting import (  # noqa: E501
        retarget_one_frame,
    )
    has_motion_utils = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_motion_utils = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception

# yapf: enable


class MotionUtilsRetargeting(BaseApplication):

    def __init__(self,
                 target_actor: str,
                 host_ip: str,
                 host_port: int,
                 logger=None) -> None:
        BaseApplication.__init__(self, logger=logger)
        if not has_motion_utils:
            self.logger.error(import_exception)
            raise ImportError
        self.target_actor = target_actor
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
            src_smpl_x_data=retarget_src_dict,
            tgt_name2bone=None,
            src_actor_name='SMPLX',
            tgt_actor_name=self.target_actor)
        # send to ue
        motion_data = {self.target_actor: motion_data}
        # for k in list(motion_data.keys()):
        #     if 'Finger' in k:
        #         motion_data.pop(k)
        # motion_data = {'xiaoning': motion_data}
        motion_data = json.dumps(motion_data).encode('UTF-16LE')
        self.client.sendto(motion_data, (self.host_ip, self.host_port))

    def __del__(self):
        pass
