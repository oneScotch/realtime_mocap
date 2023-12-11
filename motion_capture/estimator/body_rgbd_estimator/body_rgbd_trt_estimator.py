# yapf: disable
import numpy as np
import time
import torch

from realtime_mocap.motion_capture.estimator.base_estimator import (
    BaseEstimator,
)
from realtime_mocap.utils.trt_utils import HostDeviceMem
from .pre_post_processor import RgbdBodyposePrePoseProcessor

try:
    import pycuda.driver as cuda
    import tensorrt as trt
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    has_trt = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_trt = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception


# yapf: enable


class BodyRgbdTRTEstimator(BaseEstimator):

    def __init__(self,
                 trt_engine_path,
                 hidden_init_path,
                 body_model_dir,
                 pykinect_path,
                 betas_path,
                 gender='male',
                 device='cuda',
                 profile_period: float = 10.0,
                 verbose: bool = True,
                 logger=None):
        super().__init__(logger=logger)
        self.device = device
        self.trt_engine_path = trt_engine_path
        self.processor = RgbdBodyposePrePoseProcessor(
            body_model_dir=body_model_dir,
            pykinect_path=pykinect_path,
            betas_path=betas_path,
            gender=gender,
            device=device)
        self.init_trt_engine()

        self.hidden = np.load(hidden_init_path)

        self.profile_period = profile_period
        self.verbose = verbose

        self.forward_count = 0
        self.last_profile_time = time.time()

        self.pre_time_sum = 0.0
        self.rnn_time_sum = 0.0
        self.post_time_sum = 0.0

    def init_trt_engine(self):
        self.pycuda_context = cuda.Device(0).make_context()
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, namespace='')
        with open(self.trt_engine_path,
                  'rb') as f_rb, trt.Runtime(trt_logger) as trt_runtime:
            self.trt_runtime_engine = trt_runtime.deserialize_cuda_engine(
                f_rb.read())
        self.stream = cuda.Stream()

        self.trt_inputs = dict()
        self.trt_outputs = dict()
        self.buffer_ptrs = []
        for binding in self.trt_runtime_engine:
            size = trt.volume(
                self.trt_runtime_engine.get_binding_shape(binding))
            # size: how many bytes for this binding
            dtype = trt.nptype(
                self.trt_runtime_engine.get_binding_dtype(binding))
            # dtype: fp32-np.float32
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype=dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            self.buffer_ptrs.append(int(device_mem))
            # Append to the appropriate list.
            if self.trt_runtime_engine.binding_is_input(binding):
                self.trt_inputs[binding] = HostDeviceMem(host_mem, device_mem)
            else:
                self.trt_outputs[binding] = HostDeviceMem(host_mem, device_mem)
        self.context = self.trt_runtime_engine.create_execution_context()

    def forward(self,
                img_arr,
                k4a_pose,
                timestamp=None,
                frame_idx=None,
                **kwargs):
        start_time = time.time()
        with torch.no_grad():
            input_kinect_joints = (k4a_pose[0, :, 2:5] / 1000).astype(
                np.float32)
            module_input_dict = self.processor.pre_process(input_kinect_joints)
            pre_time = time.time()
            self.pre_time_sum += pre_time - start_time
            # create context for inferring
            self.pycuda_context.push()
            for input_name, input_mem in self.trt_inputs.items():
                if input_name == 'hidden_in':
                    np.copyto(input_mem.host,
                              self.hidden.astype('>f4').reshape(-1).ravel())
                else:
                    input_np = module_input_dict[input_name].cpu().numpy()
                    np.copyto(input_mem.host,
                              input_np.astype('>f4').reshape(-1).ravel())
            self.context.execute_async_v2(
                bindings=self.buffer_ptrs, stream_handle=self.stream.handle)
            for output_mem in self.trt_outputs.values():
                cuda.memcpy_dtoh_async(output_mem.host, output_mem.device,
                                       self.stream)
            self.stream.synchronize()
            body_pose_rotmat = self.trt_outputs[
                'body_pose_rotmat'].host.reshape(21, 3, 3)
            self.hidden = self.trt_outputs['hidden_out'].host.reshape(
                2, 1, 1000)
            self.pycuda_context.pop()

            rnn_time = time.time()
            self.rnn_time_sum += rnn_time - pre_time
            smplx_body_result = self.processor.post_process(
                torch.tensor(body_pose_rotmat, device=self.processor.device),
                input_kinect_joints=input_kinect_joints)

            post_time = time.time()
            self.post_time_sum += post_time - rnn_time
        smplx_body_result.update(
            dict(timestamp=timestamp, frame_idx=frame_idx, img_arr=img_arr))
        time_diff = time.time() - self.last_profile_time
        self.forward_count += 1
        if time_diff >= self.profile_period:
            if self.verbose:
                self.logger.info(
                    '\n' + 'pre_time:' +
                    f' {self.pre_time_sum/self.forward_count}\n' +
                    'rnn_time:' +
                    f' {self.rnn_time_sum/self.forward_count}\n' +
                    'post_time:' +
                    f' {self.post_time_sum/self.forward_count}\n' + '\n')
            self.forward_count = 0
            self.last_profile_time = time.time()
            self.pre_time_sum = 0.0
            self.rnn_time_sum = 0.0
            self.post_time_sum = 0.0
        return smplx_body_result
