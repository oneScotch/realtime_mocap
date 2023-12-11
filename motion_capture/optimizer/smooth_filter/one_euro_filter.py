import numpy as np

from .base_filter import BaseFilter


class LowPassFilter:

    def __init__(self):
        self.prev_raw_value = None
        self.prev_filtered_value = None

    def process(self, value, alpha):
        if self.prev_raw_value is None:
            s = value
        else:
            s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
        self.prev_raw_value = value
        self.prev_filtered_value = s
        return s


class OneEuroFilter(BaseFilter):
    """OneEuroFilter for optimizing rotmax."""

    def __init__(self,
                 mincutoff=1.0,
                 beta=0.0,
                 dcutoff=1.0,
                 freq=30,
                 logger=None) -> None:

        super().__init__(logger=logger)
        self.logger = logger
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def compute_alpha(self, cutoff):
        t = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / t)

    def forward(self, x: np.ndarray, **kwargs):
        """Forward function of OneEuroFilter.

        Args:
            x (np.ndarray): The input data.
        """
        prev_x = self.x_filter.prev_raw_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        return self.x_filter.process(x, self.compute_alpha(cutoff))

    def __call__(self, x, **kwargs):
        return self.forward(x, **kwargs)
