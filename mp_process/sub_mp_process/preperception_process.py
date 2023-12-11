import multiprocessing
import time

from realtime_mocap.motion_capture.estimator.builder import build_estimator
from .sub_base_process import SubBaseProcess


class PreperceptionProcess(SubBaseProcess):

    def __init__(self,
                 mp_queue_src: multiprocessing.Queue,
                 mp_queue_dst_dict: dict,
                 estimator,
                 *args,
                 name='preperception_process',
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
        self.estimator_cfg = estimator
        self.estimator = None
        self.mp_queue_src = mp_queue_src
        self.mp_queue_dst_dict = mp_queue_dst_dict

    def init_estimator(self):
        self.estimator_cfg['logger'] = self.logger
        self.estimator = build_estimator(self.estimator_cfg)

    def run(self) -> None:
        SubBaseProcess.run(self)
        self.init_estimator()
        queue_src_recv_time = 0.0
        queue_dst_send_time = 0.0
        estimation_time = 0.0
        latency_time = 0.0
        iter_count = 0
        last_profile_time = time.time()
        while True:
            # get data dict from stream source
            # block until there's one input element
            queue_src_start_time = time.time()
            data_dict = self.mp_queue_src.get(block=True)
            recv_time = time.time()
            queue_src_recv_time += recv_time - queue_src_start_time
            # estimate mocap results
            estimation_start_time = time.time()
            latency_time += estimation_start_time - data_dict['timestamp']
            result_dict = self.estimator.forward(**data_dict)
            estimation_end_time = time.time()
            estimation_time += estimation_end_time - estimation_start_time
            # block until there's one empty slot
            for k, queue_dst in self.mp_queue_dst_dict.items():
                queue_dst.put(result_dict, block=True)
            cur_time = time.time()
            queue_dst_send_time += cur_time - estimation_end_time
            iter_count += 1
            if cur_time - last_profile_time >= self.profile_period:
                if self.verbose:
                    self.logger.info(
                        f'{self.name} time analysis:' +
                        '\nqueue_src_recv_time: ' +
                        f'{queue_src_recv_time/iter_count}' +
                        f'\nestimation_time: {estimation_time/iter_count}' +
                        '\nqueue_dst_send_time: ' +
                        f'{queue_dst_send_time/iter_count}' +
                        f'\ntheoretical fps: {iter_count / estimation_time}' +
                        '\nlatency: ' + f'{latency_time/iter_count}' + '\n')
                estimation_time = 0.0
                queue_src_recv_time = 0.0
                queue_dst_send_time = 0.0
                latency_time = 0.0
                iter_count = 0
                last_profile_time = cur_time

    def __del__(self):
        pass
