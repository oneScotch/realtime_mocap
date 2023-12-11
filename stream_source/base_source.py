from xrprimer.utils.log_utils import get_logger


class BaseSource:

    def __init__(self, logger) -> None:
        self.logger = get_logger(logger)

    def get_data(self, **kwargs):
        ...
