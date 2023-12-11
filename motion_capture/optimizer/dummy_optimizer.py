import numpy as np
import time

from .base_optimizer import BaseOptimizer


class DummyOptimizer(BaseOptimizer):
    BASE_TIME = 0.01

    def __init__(self, logger) -> None:
        BaseOptimizer.__init__(logger=logger)

    def forward(self, smplx_data, **kwargs):
        time_offset = (np.random.rand() - 0.5) * 0.5
        forward_time = self.__class__.BASE_TIME * (1 + time_offset)
        time.sleep(forward_time)
        return smplx_data
