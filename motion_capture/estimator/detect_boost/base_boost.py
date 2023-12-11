from xrprimer.utils.log_utils import get_logger


class BaseBoost:

    def __init__(self, logger) -> None:
        self.logger = get_logger(logger)

    def get_image(self, img_arr, **kwargs):
        ...

    def get_points(self, points2d, **kwargs):
        ...
