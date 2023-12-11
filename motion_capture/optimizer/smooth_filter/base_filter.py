from xrprimer.utils.log_utils import get_logger


class BaseFilter:

    def __init__(self, logger) -> None:
        self.logger = get_logger(logger)

    def forward(self, array_data, **kwargs):
        ...
