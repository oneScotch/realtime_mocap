from mmcv.utils import Registry

from .base_filter import BaseFilter
from .kalman_filter import KalmanFilter
from .one_euro_filter import OneEuroFilter

SMOOTH_FILTER = Registry('smooth_filter')
SMOOTH_FILTER.register_module(name='OneEuroFilter', module=OneEuroFilter)
SMOOTH_FILTER.register_module(name='KalmanFilter', module=KalmanFilter)


def build_smooth_filter(cfg) -> BaseFilter:
    """Build smooth_filter."""
    return SMOOTH_FILTER.build(cfg)
