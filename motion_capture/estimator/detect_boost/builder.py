from mmcv.utils import Registry

from .base_boost import BaseBoost
from .crop_body_boost import CropBodyBoost
from .crop_hands_boost import CropHandsBoost

DETECT_BOOST = Registry('detect_boost')
DETECT_BOOST.register_module(name='CropBodyBoost', module=CropBodyBoost)
DETECT_BOOST.register_module(name='CropHandsBoost', module=CropHandsBoost)


def build_detect_boost(cfg) -> BaseBoost:
    """Build detect_boost."""
    return DETECT_BOOST.build(cfg)
