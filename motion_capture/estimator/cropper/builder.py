from mmcv.utils import Registry

from .body_frank_cropper import BodyFrankCropper
from .hand_frank_cropper import HandFrankCropper
from .hand_intag_cropper import HandIntagCropper

CROPPER = Registry('cropper')

CROPPER.register_module(name='HandIntagCropper', module=HandIntagCropper)
CROPPER.register_module(name='HandFrankCropper', module=HandFrankCropper)
CROPPER.register_module(name='BodyFrankCropper', module=BodyFrankCropper)


def build_cropper(cfg) -> HandIntagCropper:
    """Build cropper."""
    return CROPPER.build(cfg)
