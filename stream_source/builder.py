from mmcv.utils import Registry

from .base_source import BaseSource
from .dummy_source import DummySource
from .k4a_file_source import K4AFileSource
from .k4a_stream_source import K4AStreamSource
from .smplx_file_source import SMPLXFileSource
from .smplx_handpose_source import SMPLXHandposeSource

STREAM_SRC = Registry('stream_source')

STREAM_SRC.register_module(name='DummyProducer', module=DummySource)
STREAM_SRC.register_module(name='K4AStreamSource', module=K4AStreamSource)
STREAM_SRC.register_module(name='K4AFileSource', module=K4AFileSource)
STREAM_SRC.register_module(
    name='SMPLXHandposeSource', module=SMPLXHandposeSource)
STREAM_SRC.register_module(name='SMPLXFileSource', module=SMPLXFileSource)


def build_stream_src(cfg) -> BaseSource:
    """Build stream_src."""
    return STREAM_SRC.build(cfg)
