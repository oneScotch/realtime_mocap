import multiprocessing
import time

from realtime_mocap.motion_capture.aggregator.builder import build_aggregator
from realtime_mocap.motion_capture.optimizer.builder import build_optimizer
from realtime_mocap.utils.mp_utils import put_nowait_force
from .sub_base_process import SubBaseProcess


class AggregatorProcess(SubBaseProcess):

    def __init__(self,
                 mp_queue_src_dict: dict,
                 mp_queue_dst: multiprocessing.Queue,
                 aggregator,
                 *args,
                 force_push: bool = False,
                 name='aggregator_process',
                 fast_optimizer=None,
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
        self.aggregator_cfg = aggregator
        self.aggregator = None
        self.optimizer_cfg = fast_optimizer
        self.optimizer = None
        self.mp_queue_src_dict = mp_queue_src_dict
        self.mp_queue_dst = mp_queue_dst
        self.force_push = force_push

    def init_aggregator(self):
        self.aggregator_cfg['logger'] = self.logger
        self.aggregator = build_aggregator(self.aggregator_cfg)

    def init_optimizer(self):
        if self.optimizer_cfg is not None:
            self.optimizer_cfg['logger'] = self.logger
            self.optimizer = build_optimizer(self.optimizer_cfg)

    def run(self) -> None:
        SubBaseProcess.run(self)
        self.init_aggregator()
        self.init_optimizer()
        aggregate_time = 0.0
        latency_time = 0.0
        iter_count = 0
        last_profile_time = time.time()
        while True:
            # get data dict from stream source
            # block until there's one input element
            smplx_nested_dict = {}
            for k, queue_src in self.mp_queue_src_dict.items():
                smplx_nested_dict[k] = queue_src.get(block=True)
            # aggregate mocap results
            aggregate_start_time = time.time()
            result_dict = self.aggregator.forward(smplx_nested_dict)
            # timestamp can only be located in the merged result dict
            latency_time += aggregate_start_time - result_dict['timestamp']
            if self.optimizer is not None:
                result_dict = self.optimizer.forward(**result_dict)
            aggregate_time += time.time() - aggregate_start_time
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
                        f'\naggregator_time: {aggregate_time/iter_count}' +
                        f'\ntheoretical fps: {iter_count / aggregate_time}' +
                        '\nlatency: ' + f'{latency_time/iter_count}' + '\n')
                aggregate_time = 0.0
                latency_time = 0.0
                iter_count = 0
                last_profile_time = cur_time

    def __del__(self):
        pass
