from mmcv.utils import Registry

from .base_application import BaseApplication
from .fast_retargeting import FastRetargeting
from .motion_utils_retargeting import MotionUtilsRetargeting
from .smplx_cv2_visualization import SmplxCv2Visualization
from .smplx_mpr_visualization import SmplxMprVisualization
from .xrmort_retargeting import XRMoRTRetargeting

APPLICATION = Registry('application')
APPLICATION.register_module(
    name='SmplxMprVisualization', module=SmplxMprVisualization)
APPLICATION.register_module(
    name='SmplxCv2Visualization', module=SmplxCv2Visualization)
APPLICATION.register_module(name='FastRetargeting', module=FastRetargeting)
APPLICATION.register_module(
    name='MotionUtilsRetargeting', module=MotionUtilsRetargeting)
APPLICATION.register_module(name='XRMoRTRetargeting', module=XRMoRTRetargeting)


def build_application(cfg) -> BaseApplication:
    """Build application."""
    return APPLICATION.build(cfg)
