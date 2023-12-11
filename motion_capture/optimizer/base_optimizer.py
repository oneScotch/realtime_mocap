from xrprimer.utils.log_utils import get_logger


class BaseOptimizer:

    def __init__(self, logger) -> None:
        self.logger = get_logger(logger)

    def forward(self, smplx_data, **kwargs):
        ...
