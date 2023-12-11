# yapf: disable
import numpy as np
import time
import torch

from realtime_mocap.motion_capture.estimator.base_estimator import (
    BaseEstimator,
)
from realtime_mocap.motion_capture.optimizer.builder import build_optimizer
from realtime_mocap.utils.trt_utils import HostDeviceMem
from ..cropper.builder import build_cropper
from .postprocess import flip_rotmat
from .preprocess import normalize_img_intaghand, pad_resize_img_intaghand

# yapf: enable

try:
    import pycuda.driver as cuda
    import tensorrt as trt
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    has_trt_intaghand = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_trt_intaghand = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception


class HandsIntagTRTEstimator(BaseEstimator):

    def __init__(self,
                 device,
                 trt_engine_path,
                 mano_model_path,
                 optimizer: dict,
                 cropper: dict,
                 profile_period: float = 10.0,
                 verbose: bool = True,
                 ret_crop: bool = False,
                 dtype: str = 'fp32',
                 logger=None):
        super().__init__(logger=logger)
        self.context = None
        self.pycuda_context = None
        if not has_trt_intaghand:
            self.logger.error(import_exception)
            raise ImportError
        self.mano_model_path = mano_model_path
        self.ret_crop = ret_crop
        self.trt_engine_path = trt_engine_path
        self.profile_period = profile_period
        self.verbose = verbose
        self.device = device
        self.dtype = dtype

        cropper['logger'] = self.logger
        self.cropper = build_cropper(cropper)

        self.init_trt_engine()

        self.left_optimizer = None
        self.right_optimizer = None
        if isinstance(optimizer, dict):
            optimizer['logger'] = self.logger
            if optimizer['type'] == 'AIKOptimizer':
                optimizer['mano_config']['side'] = 'right'
                self.right_optimizer = build_optimizer(optimizer)
                optimizer['mano_config']['side'] = 'left'
                self.left_optimizer = build_optimizer(optimizer)
            else:
                self.right_optimizer = build_optimizer(optimizer)

        self.forward_count = 0
        self.last_profile_time = time.time()

        self.crop_time_sum = 0.0
        self.estimate_time_sum = 0.0
        self.smooth_time_sum = 0.0
        self.rotmax_time_sum = 0.0

        self.video_writer = None
        self.video_frame_count = 0

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
            # binding = input / left / right
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
        assert len(self.trt_inputs) == 1
        self.context = self.trt_runtime_engine.create_execution_context()

    def joints2rotmat_iknet(self, hand_dict):
        results = dict()
        with torch.no_grad():
            for side, kps3d in hand_dict.items():
                if side == 'left':
                    wrist_j = kps3d[:, 0, 0]
                    kps3d[:, :, 0] = -kps3d[:, :, 0]
                    offset_j = wrist_j - kps3d[:, 0, 0]
                    kps3d[:, :, 0] += offset_j
                rotmat = self.right_optimizer(kps3d, side=side).squeeze(0)
                if side == 'left':
                    rotmat = flip_rotmat(rotmat)
                results[side] = np.expand_dims(rotmat, axis=0)
        return results

    def joints2rotmat_aik(self, hand_dict):
        results = dict()
        for side, kps3d in hand_dict.items():
            for _ in range(1):
                smplx_data = dict()
                if side == 'left':
                    ret = self.left_optimizer(
                        smplx_data,
                        kps3d[0].cpu().numpy(),
                        do_eval=False,
                        output_tensor=False,
                        return_joints=False)
                else:
                    ret = self.right_optimizer(
                        smplx_data,
                        kps3d[0].cpu().numpy(),
                        do_eval=False,
                        output_tensor=False,
                        return_joints=False)
                hand_pose, hand_shape = ret[:2]
            results[side] = hand_pose
        return results

    def forward(self, img_arr, k4a_pose, hands_keypoints2d=None, **kwargs):
        result = dict()
        start_time = time.time()
        # crop hands from image
        cropped_img, bbox_dict = self.cropper.forward(
            img_arr=img_arr,
            k4a_pose=k4a_pose,
            mp_keypoints2d_dict=hands_keypoints2d)
        if self.ret_crop:
            result['cropped_img'] = cropped_img.copy()
        cropped_img_resized = pad_resize_img_intaghand(cropped_img)
        input_np = normalize_img_intaghand(cropped_img_resized)
        crop_time = time.time()
        self.crop_time_sum += crop_time - start_time
        # infer
        # create context for inferring
        self.pycuda_context.push()
        kps3d_dict = {}
        input_mem = self.trt_inputs['input']
        np.copyto(input_mem.host, input_np.astype('>f4').reshape(-1).ravel())

        cuda.memcpy_htod_async(input_mem.device, input_mem.host, self.stream)
        # cuda.memcpy_htod(input_mem.device, input_mem.host)
        self.context.execute_async_v2(
            bindings=self.buffer_ptrs, stream_handle=self.stream.handle)
        # context.execute(batch_size=1, bindings=self.buffer_ptrs)
        for output_mem in self.trt_outputs.values():
            cuda.memcpy_dtoh_async(output_mem.host, output_mem.device,
                                   self.stream)
            # cuda.memcpy_dtoh(output_mem.host, output_mem.device)
        self.stream.synchronize()
        left_np = self.trt_outputs['left'].host.reshape(1, 21, 3)
        right_np = self.trt_outputs['right'].host.reshape(1, 21, 3)
        kps3d_dict['left'] = torch.tensor(left_np, device=self.device)
        kps3d_dict['right'] = torch.tensor(right_np, device=self.device)
        estimate_time = time.time()
        self.pycuda_context.pop()
        self.estimate_time_sum += estimate_time - crop_time
        # skip smooth
        smooth_time = time.time()
        self.smooth_time_sum += smooth_time - estimate_time
        if self.left_optimizer is None and self.right_optimizer is not None:
            with torch.no_grad():
                handpose = self.joints2rotmat_iknet(kps3d_dict)
        elif self.left_optimizer is not None \
                and self.right_optimizer is not None:
            handpose = self.joints2rotmat_aik(kps3d_dict)
        else:
            raise NotImplementedError
        calc_rotmax_time = time.time()
        result.update(handpose)
        # if some hand is default,
        # reset it to None for aggregator
        if bbox_dict['left'] is None:
            result['left'] = None
        if bbox_dict['right'] is None:
            result['right'] = None
        self.rotmax_time_sum += calc_rotmax_time - smooth_time
        time_diff = time.time() - self.last_profile_time
        self.forward_count += 1
        if time_diff >= self.profile_period:
            if self.verbose:
                self.logger.info(
                    '\n' + 'crop_time:' +
                    f' {self.crop_time_sum/self.forward_count}\n' +
                    'estimate_time:' +
                    f' {self.estimate_time_sum/self.forward_count}\n' +
                    'smooth_time:' +
                    f' {self.smooth_time_sum/self.forward_count}\n' +
                    'calc_rotmax_time:' +
                    f' {self.rotmax_time_sum/self.forward_count}' + '\n')
            self.forward_count = 0
            self.last_profile_time = time.time()
            self.crop_time_sum = 0.0
            self.estimate_time_sum = 0.0
            self.smooth_time_sum = 0.0
            self.rotmax_time_sum = 0.0
        return result

    def __del__(self):
        if self.context is not None:
            self.context.pop()
            self.context.__del__()
            delattr(self, 'context')
        if self.self.pycuda_context is not None:
            self.pycuda_context.pop()
