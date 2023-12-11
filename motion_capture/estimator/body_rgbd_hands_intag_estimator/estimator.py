# yapf: disable
import numpy as np
import time
import torch

from realtime_mocap.motion_capture.estimator.base_estimator import (
    BaseEstimator,
)
from realtime_mocap.utils.trt_utils import HostDeviceMem
from ..body_rgbd_estimator.pre_post_processor import (
    RgbdBodyposePrePoseProcessor,
)
from ..cropper.builder import build_cropper
from ..hands_intag_estimator.preprocess import (
    normalize_img_intaghand, pad_resize_img_intaghand,
)

# yapf: enable

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


class BodyRgbdHandsIntagEstimator(BaseEstimator):

    def __init__(self,
                 device,
                 body_model_dir,
                 pykinect_path,
                 betas_path,
                 trt_engine_path,
                 hidden_init_path,
                 cropper: dict,
                 profile_period: float = 10.0,
                 verbose: bool = True,
                 ret_crop: bool = False,
                 gender='male',
                 dtype: str = 'fp32',
                 logger=None):
        super().__init__(logger=logger)
        self.context = None
        self.pycuda_context = None
        if not has_trt:
            self.logger.error(import_exception)
            raise ImportError
        self.ret_crop = ret_crop
        self.trt_engine_path = trt_engine_path
        self.profile_period = profile_period
        self.verbose = verbose
        self.device = device
        self.dtype = dtype

        cropper['logger'] = self.logger
        self.hand_cropper = build_cropper(cropper)
        self.body_processor = RgbdBodyposePrePoseProcessor(
            body_model_dir=body_model_dir,
            pykinect_path=pykinect_path,
            betas_path=betas_path,
            gender=gender,
            device=device)
        self.init_trt_engine()
        self.hidden = np.load(hidden_init_path)

        self.forward_count = 0
        self.last_profile_time = time.time()

        self.hand_crop_time_sum = 0.0
        self.body_pre_time_sum = 0.0
        self.infer_time_sum = 0.0
        self.body_post_time_sum = 0.0

    def init_trt_engine(self):
        self.pycuda_context = cuda.Device(0).make_context()
        # do not import pycuda.autoinit, which causes conflicts with smplx_mpr
        # Infer TensorRT Engine
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
        assert len(self.trt_inputs) == 5
        self.context = self.trt_runtime_engine.create_execution_context()

    def forward(self,
                img_arr,
                k4a_pose,
                hands_keypoints2d=None,
                hands_bboxes=None,
                **kwargs):
        result = dict()
        start_time = time.time()
        # crop hands from image
        cropped_img, bbox_dict = self.hand_cropper.forward(
            img_arr=img_arr,
            k4a_pose=k4a_pose,
            mp_keypoints2d_dict=hands_keypoints2d,
            hands_bboxes=hands_bboxes)
        if self.ret_crop:
            result['cropped_img'] = cropped_img.copy()
        cropped_img_resized = pad_resize_img_intaghand(cropped_img)
        normed_img_np = normalize_img_intaghand(cropped_img_resized)
        crop_time = time.time()
        self.hand_crop_time_sum += crop_time - start_time
        # pre-process for body pose
        with torch.no_grad():
            input_kinect_joints = (k4a_pose[0, :, 2:5] / 1000).astype(
                np.float32)
            module_input_dict = self.body_processor.pre_process(
                input_kinect_joints)
        pre_time = time.time()
        self.body_pre_time_sum += pre_time - crop_time
        # create context for inferring
        self.pycuda_context.push()
        for input_name, input_mem in self.trt_inputs.items():
            if input_name == 'hidden_in':
                np.copyto(input_mem.host,
                          self.hidden.astype('>f4').reshape(-1).ravel())
            elif input_name in module_input_dict:
                input_np = module_input_dict[input_name].cpu().numpy()
                np.copyto(input_mem.host,
                          input_np.astype('>f4').reshape(-1).ravel())
            else:
                np.copyto(input_mem.host,
                          normed_img_np.astype('>f4').reshape(-1).ravel())
            cuda.memcpy_htod_async(input_mem.device, input_mem.host,
                                   self.stream)
        self.context.execute_async_v2(
            bindings=self.buffer_ptrs, stream_handle=self.stream.handle)
        for output_mem in self.trt_outputs.values():
            cuda.memcpy_dtoh_async(output_mem.host, output_mem.device,
                                   self.stream)
        self.stream.synchronize()
        left_hand_rotmat = self.trt_outputs['left_rotmat'].host.reshape(
            1, 16, 3, 3)
        right_hand_rotmat = self.trt_outputs['right_rotmat'].host.reshape(
            1, 16, 3, 3)
        body_pose_rotmat = self.trt_outputs['body_pose_rotmat'].host.reshape(
            21, 3, 3)
        self.hidden = self.trt_outputs['hidden_out'].host.reshape(2, 1, 1000)
        self.pycuda_context.pop()
        infer_time = time.time()
        self.infer_time_sum += infer_time - pre_time
        smplx_body_result = self.body_processor.post_process(
            torch.tensor(body_pose_rotmat, device=self.body_processor.device),
            input_kinect_joints=input_kinect_joints)
        # prepare result
        result['left_hand_rotmat'] = left_hand_rotmat
        result['right_hand_rotmat'] = right_hand_rotmat
        result.update(kwargs)
        result.update(smplx_body_result)
        result['body_pose'] = result['body_pose'].reshape(-1)
        result['global_orient'] = result['global_orient'].reshape(-1)
        result['transl'] = result['transl'].reshape(-1)
        result['img_arr'] = img_arr
        # if some hand is default,
        # reset it to None for aggregator
        if bbox_dict['left'] is None:
            result['left_hand_rotmat'] = None
        if bbox_dict['right'] is None:
            result['right_hand_rotmat'] = None
        post_time = time.time()
        self.body_post_time_sum += post_time - infer_time

        time_diff = time.time() - self.last_profile_time
        self.forward_count += 1
        if time_diff >= self.profile_period:
            if self.verbose:
                self.logger.info(
                    '\n' + 'hand_crop_time:' +
                    f' {self.hand_crop_time_sum/self.forward_count}\n' +
                    'body_pre_time:' +
                    f' {self.body_pre_time_sum/self.forward_count}\n' +
                    'infer_time:' +
                    f' {self.infer_time_sum/self.forward_count}\n' +
                    'body_post_time:' +
                    f' {self.body_post_time_sum/self.forward_count}' + '\n')
            self.forward_count = 0
            self.last_profile_time = time.time()
            self.hand_crop_time_sum = 0.0
            self.body_pre_time_sum = 0.0
            self.infer_time_sum = 0.0
            self.body_post_time_sum = 0.0
        return result

    def __del__(self):
        if self.context is not None:
            self.context.pop()
            self.context.__del__()
            delattr(self, 'context')
        if self.self.pycuda_context is not None:
            self.pycuda_context.pop()
