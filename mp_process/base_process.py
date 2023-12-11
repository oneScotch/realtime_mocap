import multiprocessing
import os
from typing import Union
from xrprimer.utils.log_utils import get_logger, setup_logger


class BaseProcess(multiprocessing.Process):

    def __init__(self,
                 name: str,
                 logger: Union[dict, str, None],
                 *args,
                 verbose: bool = True,
                 profile_period: float = 10.0,
                 **kwargs) -> None:
        if not isinstance(logger, dict):
            self.logger = get_logger(logger)
        else:
            self.logger = logger
        self.verbose = verbose
        self.profile_period = profile_period
        multiprocessing.Process.__init__(self, *args, name=name, **kwargs)

    def run(self):
        if isinstance(self.logger, dict):
            self.logger = setup_logger(**self.logger)
        pid = os.getpid()
        self.logger.debug(f'PID of process {self.name}' + f' is {pid}.')
