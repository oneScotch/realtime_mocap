from mmcv.utils import Registry

from .aik_optimizer import AIKOptimizer
from .base_optimizer import BaseOptimizer
from .bbox_smooth_optimizer import BboxSmoothOptimizer
from .dummy_optimizer import DummyOptimizer
from .iknet_optimizer import IKNetOptimizer
from .iknet_trt_optimizer import IKNetTRTOptimizer
from .k4abt_smooth_optimizer import K4abtSmoothOptimizer
from .smplifyx_optimizer import SMPLifyxOptimizer
from .smplx_smooth_optimizer import SmplxSmoothOptimizer

OPTIMIZER = Registry('optimizer')
OPTIMIZER.register_module(name='DummyOptimizer', module=DummyOptimizer)
OPTIMIZER.register_module(name='AIKOptimizer', module=AIKOptimizer)
OPTIMIZER.register_module(name='IKNetOptimizer', module=IKNetOptimizer)
OPTIMIZER.register_module(name='IKNetTRTOptimizer', module=IKNetTRTOptimizer)
OPTIMIZER.register_module(name='SMPLifyxOptimizer', module=SMPLifyxOptimizer)
OPTIMIZER.register_module(
    name='SmplxSmoothOptimizer', module=SmplxSmoothOptimizer)
OPTIMIZER.register_module(
    name='BboxSmoothOptimizer', module=BboxSmoothOptimizer)
OPTIMIZER.register_module(
    name='K4abtSmoothOptimizer', module=K4abtSmoothOptimizer)


def build_optimizer(cfg) -> BaseOptimizer:
    """Build optimizer."""
    return OPTIMIZER.build(cfg)
