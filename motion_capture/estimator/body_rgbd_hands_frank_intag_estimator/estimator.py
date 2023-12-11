# yapf: disable
import numpy as np
import time
import torch

from realtime_mocap.motion_capture.estimator.base_estimator import (
    BaseEstimator,
)
from realtime_mocap.utils.trt_utils import HostDeviceMem
from ..body_rgbd_estimator.nn_module import BodyRgbdRotmatModule
from ..body_rgbd_estimator.pre_post_processor import (
    RgbdBodyposePrePoseProcessor,
)
from ..cropper.builder import build_cropper
from ..hands_frank_estimator.preprocess import get_input_batch_frank
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


class BodyRgbdHandsFrankIntagEstimator(BaseEstimator):

    def __init__(self,
                 body_model_dir,
                 pykinect_path,
                 betas_path,
                 rgbd_checkpoint_path,
                 hidden_init_path,
                 intag_cropper: dict,
                 frank_cropper: dict,
                 intag_trt_engine_path: str,
                 frank_trt_engine_path: str,
                 profile_period: float = 10.0,
                 verbose: bool = True,
                 device='cuda',
                 gender='male',
                 dtype: str = 'fp32',
                 logger=None):
        super().__init__(logger=logger)
        self.context = None
        self.pycuda_context = None
        if not has_trt:
            self.logger.error(import_exception)
            raise ImportError
        self.intag_trt_engine_path = intag_trt_engine_path
        self.frank_trt_engine_path = frank_trt_engine_path

        self.profile_period = profile_period
        self.verbose = verbose
        self.device = device
        self.dtype = dtype

        hand_cropper_cfgs = dict(intag=intag_cropper, frank=frank_cropper)
        self.hand_cropper_dict = dict()
        for key, cfg in hand_cropper_cfgs.items():
            cfg['logger'] = self.logger
            self.hand_cropper_dict[key] = build_cropper(cfg)

        self.body_processor = RgbdBodyposePrePoseProcessor(
            body_model_dir=body_model_dir,
            pykinect_path=pykinect_path,
            betas_path=betas_path,
            gender=gender,
            device=device)
        self.init_trt_engine()
        self.init_rgbd_module(rgbd_checkpoint_path, hidden_init_path)

        self.forward_count = 0
        self.last_profile_time = time.time()

        self.hand_crop_time_sum = 0.0
        self.body_pre_time_sum = 0.0
        self.infer_time_sum = 0.0
        self.body_post_time_sum = 0.0

    def init_rgbd_module(self, rnn_checkpoint_path, hidden_init_path):
        self.bodypose_module = BodyRgbdRotmatModule(
            rnn_checkpoint_path, device=self.device)
        self.hidden = torch.from_numpy(np.load(hidden_init_path)).to(
            self.device)

    def init_trt_engine(self):
        self.pycuda_context = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        # do not import pycuda.autoinit, which causes conflicts with smplx_mpr
        # Infer TensorRT Engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(trt_logger, namespace='')
        self.trt_engine_dict = dict()
        with open(self.intag_trt_engine_path,
                  'rb') as f_rb, trt.Runtime(trt_logger) as trt_runtime:
            intag_trt_engine = trt_runtime.deserialize_cuda_engine(f_rb.read())
            self.trt_engine_dict['intag'] = intag_trt_engine

        with open(self.frank_trt_engine_path,
                  'rb') as f_rb, trt.Runtime(trt_logger) as trt_runtime:
            frank_trt_engine = trt_runtime.deserialize_cuda_engine(f_rb.read())
            self.trt_engine_dict['frank'] = frank_trt_engine
        self.trt_input_dict = dict()
        self.trt_output_dict = dict()
        self.buffer_ptr_dict = dict()
        self.context_dict = dict()
        for engine_name, engine in self.trt_engine_dict.items():
            self.trt_input_dict[engine_name] = dict()
            self.trt_output_dict[engine_name] = dict()
            self.buffer_ptr_dict[engine_name] = []
            for binding in engine:
                size = trt.volume(engine.get_binding_shape(binding))
                # size: how many bytes for this binding
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                # dtype: fp32-np.float32
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype=dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                # Append the device buffer to device bindings.
                self.buffer_ptr_dict[engine_name].append(int(device_mem))
                # Append to the appropriate list.
                if engine.binding_is_input(binding):
                    self.trt_input_dict[engine_name][binding] = HostDeviceMem(
                        host_mem, device_mem)
                else:
                    self.trt_output_dict[engine_name][binding] = HostDeviceMem(
                        host_mem, device_mem)
            self.context_dict[engine_name] = engine.create_execution_context()

    def forward(self,
                img_arr,
                k4a_pose,
                hands_keypoints2d=None,
                hands_bboxes=None,
                **kwargs):
        result = dict()
        start_time = time.time()
        # crop hands from image
        if hands_bboxes is not None and \
                hands_bboxes['together'] is None:
            hand_algorithm = 'frank'
        elif hands_bboxes is not None and \
                hands_bboxes['together'] is not None:
            hand_algorithm = 'intag'
        else:
            hand_algorithm = None
        if hand_algorithm == 'frank':
            cropped_imgs, bbox_dict = self.hand_cropper_dict[
                hand_algorithm].forward(
                    img_arr=img_arr,
                    k4a_pose=k4a_pose,
                    mp_keypoints2d_dict=hands_keypoints2d,
                    hands_bboxes=hands_bboxes)
            normed_img_np = get_input_batch_frank(cropped_imgs, bbox_dict)
        elif hand_algorithm == 'intag':
            cropped_img, bbox_dict = self.hand_cropper_dict[
                hand_algorithm].forward(
                    img_arr=img_arr,
                    k4a_pose=k4a_pose,
                    mp_keypoints2d_dict=hands_keypoints2d,
                    hands_bboxes=hands_bboxes)
            cropped_img_resized = pad_resize_img_intaghand(cropped_img)
            normed_img_np = normalize_img_intaghand(cropped_img_resized)
        else:
            pass
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
        # upload hands input
        if hand_algorithm == 'frank':
            for input_name, input_mem in self.trt_input_dict['frank'].items():
                np.copyto(input_mem.host,
                          normed_img_np.astype('>f4').reshape(-1).ravel())
                cuda.memcpy_htod_async(input_mem.device, input_mem.host,
                                       self.stream)
        elif hand_algorithm == 'intag':
            for input_name, input_mem in self.trt_input_dict['intag'].items():
                np.copyto(input_mem.host,
                          normed_img_np.astype('>f4').reshape(-1).ravel())
                cuda.memcpy_htod_async(input_mem.device, input_mem.host,
                                       self.stream)
        # execute gpu inference
        engines_to_exe = []
        if hand_algorithm == 'frank':
            engines_to_exe.append('frank')
        elif hand_algorithm == 'intag':
            engines_to_exe.append('intag')
        for engine_name in engines_to_exe:
            self.context_dict[engine_name].execute_async_v2(
                bindings=self.buffer_ptr_dict[engine_name],
                stream_handle=self.stream.handle)
            for output_mem in self.trt_output_dict[engine_name].values():
                cuda.memcpy_dtoh_async(output_mem.host, output_mem.device,
                                       self.stream)
        body_pose_rotmat, self.hidden = self.bodypose_module.forward(
            kinect_kp=module_input_dict['kinect_kp'].to(
                self.bodypose_module.device),
            pose_init=module_input_dict['pose_init'].to(
                self.bodypose_module.device),
            twists=module_input_dict['twists'].to(self.bodypose_module.device),
            hidden=self.hidden)
        self.hidden = self.hidden.detach()
        self.stream.synchronize()
        if hand_algorithm == 'frank':
            # get predicted params for pose
            trt_output = self.trt_output_dict['frank'][
                'dhands_rotmat'].host.reshape(2, -1)
            left_hand_rotmat = trt_output[0:1].reshape(1, 16, 3, 3)
            right_hand_rotmat = trt_output[1:2].reshape(1, 16, 3, 3)
        elif hand_algorithm == 'intag':
            trt_output = self.trt_output_dict['intag']
            left_hand_rotmat = trt_output['left_rotmat'].host.reshape(
                1, 16, 3, 3)
            right_hand_rotmat = trt_output['right_rotmat'].host.reshape(
                1, 16, 3, 3)
        else:
            left_hand_rotmat = None
            right_hand_rotmat = None
        body_pose_rotmat = body_pose_rotmat.reshape(21, 3, 3).to(
            self.body_processor.device)
        self.pycuda_context.pop()
        infer_time = time.time()
        self.infer_time_sum += infer_time - pre_time
        smplx_body_result = self.body_processor.post_process(
            body_pose_rotmat, input_kinect_joints=input_kinect_joints)
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
        if hand_algorithm == 'frank' and bbox_dict['left'] is None:
            result['left_hand_rotmat'] = None
        if hand_algorithm == 'frank' and bbox_dict['right'] is None:
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
        if self.pycuda_context is not None:
            self.pycuda_context.pop()
