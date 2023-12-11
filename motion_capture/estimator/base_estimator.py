from xrprimer.utils.log_utils import get_logger


class BaseEstimator:

    def __init__(self, logger) -> None:
        self.logger = get_logger(logger)

    def forward(self, img_arr, **kwargs):
        ...
