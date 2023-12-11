from mmcv.utils import Registry

from .app_pipeline import AppPipeline
from .live_pipeline import LivePipeline
from .record_pipeline import RecordPipeline

PIPELINE = Registry('pipeline')
PIPELINE.register_module(name='LivePipeline', module=LivePipeline)
PIPELINE.register_module(name='RecordPipeline', module=RecordPipeline)
PIPELINE.register_module(name='AppPipeline', module=AppPipeline)


def build_pipeline(cfg) -> LivePipeline:
    """Build pipeline."""
    return PIPELINE.build(cfg)
