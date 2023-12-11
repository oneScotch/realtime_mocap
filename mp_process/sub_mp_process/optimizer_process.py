import multiprocessing
import time

from realtime_mocap.motion_capture.optimizer.builder import build_optimizer
from realtime_mocap.utils.mp_utils import put_nowait_force
from .sub_base_process import SubBaseProcess


class OptimizerProcess(SubBaseProcess):

    def __init__(self,
                 mp_queue_src: multiprocessing.Queue,
                 mp_queue_dst: multiprocessing.Queue,
                 optimizer,
                 *args,
                 force_push: bool = False,
                 name='optimizer_process',
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
        self.optimizer_cfg = optimizer
        self.optimizer = None
        self.mp_queue_src = mp_queue_src
        self.mp_queue_dst = mp_queue_dst
        self.force_push = force_push

    def init_optimizer(self):
        self.optimizer_cfg['logger'] = self.logger
        self.optimizer = build_optimizer(self.optimizer_cfg)

    def run(self) -> None:
        SubBaseProcess.run(self)
        self.init_optimizer()
        optim_time = 0.0
        iter_count = 0
        last_profile_time = time.time()
        while True:
            # get data dict from stream source
            # block until there's one input element
            data_dict = self.mp_queue_src.get(block=True)
            # aggregate mocap results
            optim_start_time = time.time()
            result_dict = self.optimizer.forward(**data_dict)
            optim_time += time.time() - optim_start_time
            # put immediately, if fail, clean one slot
            if self.force_push:
                put_nowait_force(data=result_dict, queue_dst=self.mp_queue_dst)
            # block until there's one empty slot
            else:
                self.mp_queue_dst.put(result_dict, block=True)
            iter_count += 1
            cur_time = time.time()
            if cur_time - last_profile_time >= self.profile_period:
                if self.verbose:
                    self.logger.info(
                        f'{self.name} time analysis:' +
                        f'\noptim_time: {optim_time/iter_count}' +
                        f'\ntheoretical fps: {iter_count / optim_time}' + '\n')
                optim_time = 0.0
                iter_count = 0
                last_profile_time = cur_time

    def __del__(self):
        pass
