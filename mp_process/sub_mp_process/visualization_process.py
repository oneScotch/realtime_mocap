import multiprocessing
import time

from realtime_mocap.application.builder import build_application
from .sub_base_process import SubBaseProcess


class VisualizationProcess(SubBaseProcess):

    def __init__(self,
                 mp_queue_src: multiprocessing.Queue,
                 application,
                 *args,
                 name='visualization_process',
                 logger=None,
                 verbose: bool = True,
                 profile_period: float = 10.0,
                 **kwargs) -> None:
        SubBaseProcess.__init__(
            self,
            *args,
            name=name,
            logger=logger,
            verbose=verbose,
            profile_period=profile_period,
            **kwargs)
        self.application_cfg = application
        self.application = None
        self.mp_queue_src = mp_queue_src

    def init_application(self):
        self.application_cfg['logger'] = self.logger
        self.application = build_application(self.application_cfg)

    def run(self) -> None:
        SubBaseProcess.run(self)
        self.init_application()
        application_time = 0.0
        iter_count = 0
        last_profile_time = time.time()
        while True:
            # get data dict from stream source
            # block until there's one input element
            data_dict = self.mp_queue_src.get(block=True)
            # estimate mocap results
            application_start_time = time.time()
            self.application.forward(**data_dict)
            application_time += time.time() - application_start_time
            iter_count += 1
            cur_time = time.time()
            if cur_time - last_profile_time >= self.profile_period:
                if self.verbose:
                    self.logger.info(
                        f'{self.name} time analysis:' +
                        f'\napplication_time: {application_time/iter_count}' +
                        f'\ntheoretical fps: {iter_count / application_time}' +
                        '\n')
                application_time = 0.0
                iter_count = 0
                last_profile_time = cur_time

    def __del__(self):
        pass
