from mmcv.utils import Registry

from .base_aggregator import BaseAggregator
from .body_hand_aggregator import BodyHandAggregator
from .key_aggregator import KeyAggregator
from .mano16_aggregator import Mano16Aggregator

AGGREGATOR = Registry('aggregator')
AGGREGATOR.register_module(name='KeyAggregator', module=KeyAggregator)
AGGREGATOR.register_module(name='Mano16Aggregator', module=Mano16Aggregator)
AGGREGATOR.register_module(
    name='BodyHandAggregator', module=BodyHandAggregator)


def build_aggregator(cfg) -> BaseAggregator:
    """Build aggregator."""
    return AGGREGATOR.build(cfg)
