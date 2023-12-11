from mmcv.utils import Registry

from .app_process import AppProcess
from .base_process import BaseProcess
from .mocap_process import MoCapProcess
from .producer_process import MPProducerProcess
from .producer_record_process import MPProducerRecordProcess

MP_PROCESS = Registry('mp_process')
MP_PROCESS.register_module(name='MPProducerProcess', module=MPProducerProcess)
MP_PROCESS.register_module(name='MoCapProcess', module=MoCapProcess)
MP_PROCESS.register_module(
    name='MPProducerRecordProcess', module=MPProducerRecordProcess)
MP_PROCESS.register_module(name='AppProcess', module=AppProcess)


def build_mp_process(cfg) -> BaseProcess:
    """Build mp_process."""
    return MP_PROCESS.build(cfg)
