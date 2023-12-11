from mmcv.utils import Registry

from .aggregator_process import AggregatorProcess
from .estimator_process import EstimatorProcess
from .optimizer_process import OptimizerProcess
from .preperception_process import PreperceptionProcess
from .retargeting_process import RetargetingProcess
from .sub_base_process import SubBaseProcess
from .visualization_process import VisualizationProcess

SUB_MP_PROCESS = Registry('sub_mp_process')
SUB_MP_PROCESS.register_module(
    name='EstimatorProcess', module=EstimatorProcess)
SUB_MP_PROCESS.register_module(
    name='AggregatorProcess', module=AggregatorProcess)
SUB_MP_PROCESS.register_module(
    name='VisualizationProcess', module=VisualizationProcess)
SUB_MP_PROCESS.register_module(
    name='RetargetingProcess', module=RetargetingProcess)
SUB_MP_PROCESS.register_module(
    name='OptimizerProcess', module=OptimizerProcess)
SUB_MP_PROCESS.register_module(
    name='PreperceptionProcess', module=PreperceptionProcess)


def build_sub_mp_process(cfg) -> SubBaseProcess:
    """Build sub_mp_process."""
    return SUB_MP_PROCESS.build(cfg)
