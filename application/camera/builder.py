from mmcv.utils import Registry
from xrprimer.data_structure.camera import PinholeCameraParameter

CAMERA = Registry('camera')
CAMERA.register_module(
    name='PinholeCameraParameter', module=PinholeCameraParameter)


def build_camera(cfg) -> PinholeCameraParameter:
    """Build camera."""
    return CAMERA.build(cfg)
