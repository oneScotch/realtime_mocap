import numpy as np
import torch
from einops import rearrange
from manopth import manolayer
from manopth.rot6d import compute_rotation_matrix_from_ortho6d

from realtime_mocap.utils.trt_utils import HostDeviceMem
from .base_optimizer import BaseOptimizer

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


class IKNetTRTOptimizer(BaseOptimizer):
    """IKNet for optimizing rotations from given keypoints.

    It is only for right MANO model.
    """

    def __init__(self,
                 device: str,
                 side: str,
                 trt_engine_path: str,
                 mano_root: str,
                 logger=None) -> None:
        """Initialization of AIKOptimizer.

        Args:
            logger: Logger of the optimizer. Defaults to None.
        """
        super().__init__(logger=logger)
        if not has_trt:
            self.logger.error(import_exception)
            raise ImportError
        self.device = device
        if side != 'right':
            self.logger.error(
                'The IKNet is trained for right hand, please set it to right.')
            raise TypeError
        self.trt_engine_path = trt_engine_path
        self.init_trt_engine()
        self.right_mano_layer = manolayer.ManoLayer(
            mano_root=mano_root,
            side='right',
            use_pca=False,
            flat_hand_mean=True,
            root_rot_mode='rotmat',
            joint_rot_mode='rotmat').to(device)

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
            # binding = input / theta_raw / shape
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

    def forward(self, kps3d, **kwargs):
        """Forward function of AIKOptimizer.

        Args:
            kps3d (np.ndarray): (b, 21, 3) positions.
        """
        ref_bone_length = torch.linalg.norm(
            kps3d[:, 0] - kps3d[:, 9], dim=1,
            keepdim=True)[:, None, :].repeat(1, 21, 3)
        kps3d = (kps3d -
                 kps3d[:, 9][:, None, :].repeat(1, 21, 1)) / ref_bone_length
        # IKNet forwarding
        self.pycuda_context.push()
        input_np = kps3d.detach().cpu().numpy()
        input_mem = self.trt_inputs['kps']
        np.copyto(input_mem.host, input_np.astype('>f4').reshape(-1).ravel())
        cuda.memcpy_htod_async(input_mem.device, input_mem.host, self.stream)
        self.context.execute_async_v2(
            bindings=self.buffer_ptrs, stream_handle=self.stream.handle)
        for output_mem in self.trt_outputs.values():
            cuda.memcpy_dtoh_async(output_mem.host, output_mem.device,
                                   self.stream)
        self.stream.synchronize()
        rot6d = self.trt_outputs['theta_raw'].host.reshape(1, 16, 6)
        rot6d = torch.tensor(rot6d, device=self.device)
        self.pycuda_context.pop()
        rot6d = rearrange(rot6d, 'b j c -> (b j) c')
        rotmat = compute_rotation_matrix_from_ortho6d(rot6d)
        rotmat = rearrange(rotmat, '(b j) h w -> b j h w', j=16)
        return rotmat.cpu().numpy()

    def __call__(self, kps3d, **kwargs):
        return self.forward(kps3d, **kwargs)

    def __del__(self):
        if self.context is not None:
            self.context.pop()
            self.context.__del__()
            delattr(self, 'context')
        if self.self.pycuda_context is not None:
            self.pycuda_context.pop()


def norm_vec(vec):
    vec = vec.reshape(3)
    norm = vec[0]**2 + vec[1]**2 + vec[2]**2
    norm = torch.sqrt(norm)
    return norm
