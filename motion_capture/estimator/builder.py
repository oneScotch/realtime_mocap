# yapf: disable
from mmcv.utils import Registry

from .base_estimator import BaseEstimator
from .body_frank_estimator.estimator import BodyFrankEstimator
from .body_frank_hands_frank_intag_estimator.estimator import (
    BodyFrankHandsFrankIntagEstimator,
)
from .body_rgbd_estimator.body_rgbd_pytorch_estimator import (
    BodyRgbdPytorchEstimator,
)
from .body_rgbd_estimator.body_rgbd_trt_estimator import BodyRgbdTRTEstimator
from .body_rgbd_hands_frank_intag_estimator.estimator import (
    BodyRgbdHandsFrankIntagEstimator,
)
from .body_rgbd_hands_intag_estimator.estimator import (
    BodyRgbdHandsIntagEstimator,
)
from .dummy_bodypose_estimator import DummyRgbdPytorchBodyPoseEstimator
from .dummy_hands_estimator import DummyHandsEstimator
from .hands_frank_estimator.estimator import HandsFrankEstimator
from .hands_intag_estimator.trt_estimator import HandsIntagTRTEstimator
from .hands_mediapipe_estimator.estimator import HandsMediapipeEstimator
from .hands_mmdet_estimator.hands_mmdet_estimator import HandsMmdetEstimator
from .hands_mmdet_estimator.hands_mmdet_trt_estimator import (
    HandsMmdetTRTEstimator,
)

# yapf: enable

ESTIMATOR = Registry('estimator')
ESTIMATOR.register_module(
    name='DummyRgbdPytorchBodyPoseEstimator',
    module=DummyRgbdPytorchBodyPoseEstimator)
ESTIMATOR.register_module(
    name='DummyHandsEstimator', module=DummyHandsEstimator)
ESTIMATOR.register_module(
    name='BodyRgbdPytorchEstimator', module=BodyRgbdPytorchEstimator)
ESTIMATOR.register_module(
    name='BodyRgbdTRTEstimator', module=BodyRgbdTRTEstimator)
ESTIMATOR.register_module(
    name='HandsIntagTRTEstimator', module=HandsIntagTRTEstimator)
ESTIMATOR.register_module(
    name='HandsMediapipeEstimator', module=HandsMediapipeEstimator)
ESTIMATOR.register_module(
    name='BodyRgbdHandsIntagEstimator', module=BodyRgbdHandsIntagEstimator)
ESTIMATOR.register_module(
    name='HandsMmdetEstimator', module=HandsMmdetEstimator)
ESTIMATOR.register_module(
    name='HandsMmdetTRTEstimator', module=HandsMmdetTRTEstimator)
ESTIMATOR.register_module(
    name='BodyRgbdHandsFrankIntagEstimator',
    module=BodyRgbdHandsFrankIntagEstimator)
ESTIMATOR.register_module(
    name='HandsFrankEstimator', module=HandsFrankEstimator)
ESTIMATOR.register_module(name='BodyFrankEstimator', module=BodyFrankEstimator)
ESTIMATOR.register_module(
    name='BodyFrankHandsFrankIntagEstimator',
    module=BodyFrankHandsFrankIntagEstimator)


def build_estimator(cfg) -> BaseEstimator:
    """Build estimator."""
    return ESTIMATOR.build(cfg)
