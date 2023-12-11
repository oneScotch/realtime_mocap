from mmcv.utils import Registry
from smplx import SMPLX

BODY_MODEL = Registry('body_model')
BODY_MODEL.register_module(name='SMPLX', module=SMPLX)


def build_body_model(cfg) -> SMPLX:
    """Build body_model."""
    return BODY_MODEL.build(cfg)
