import multiprocessing
import queue
import time

from realtime_mocap.utils.mp_utils import put_nowait_force
from .base_process import BaseProcess
from .sub_mp_process.builder import build_sub_mp_process


class AppProcess(BaseProcess):

    def __init__(self,
                 mp_queue_src: multiprocessing.Queue,
                 app_processes: dict,
                 *args,
                 name='multiprocessing_app_process',
                 block_by_app: bool = False,
                 wait_time: float = 0.03,
                 app_queue_len: int = 10,
                 verbose: bool = True,
                 profile_period: float = 10.0,
                 logger=None,
                 **kwargs) -> None:
        BaseProcess.__init__(
            self,
            *args,
            name=name,
            logger=logger,
            verbose=verbose,
            profile_period=profile_period,
            **kwargs)
        self.mp_queue_src = mp_queue_src
        self.wait_time = wait_time
        self.block_by_app = block_by_app

        self.app_queues = None
        self.app_processes_cfg = {}
        for key, app_process_cfg in app_processes.items():
            if app_process_cfg is not None:
                self.app_processes_cfg[key] = app_process_cfg
        self.app_queue_len = app_queue_len
        self.app_processes = None

    def init_queues(self):
        self.app_queues = {}
        for k in self.app_processes_cfg.keys():
            self.app_queues[k] = {}
            mp_queue_src = multiprocessing.Queue(maxsize=self.app_queue_len)
            self.app_queues[k]['mp_queue_src'] = mp_queue_src

    def init_app_processes(self):
        self.app_processes = {}
        for k, v, in self.app_processes_cfg.items():
            v['mp_queue_src'] = self.app_queues[k]['mp_queue_src']
            self.app_processes[k] = build_sub_mp_process(v)
            self.app_processes[k].start()

    def run(self):
        BaseProcess.run(self)
        self.init_queues()
        self.init_app_processes()
        last_profile_time = time.time()
        queue_src_recv_time = 0.0
        latency_time = 0.0
        valid_iter_count = 0
        while True:
            # get data dict from stream source
            queue_src_start_time = time.time()
            try:
                data_dict = self.mp_queue_src.get(
                    block=True, timeout=self.wait_time)
            except queue.Empty:
                cur_time = time.time()
                queue_src_recv_time += cur_time - queue_src_start_time
                continue
            cur_time = time.time()
            queue_src_recv_time += cur_time - queue_src_start_time
            latency_time += cur_time - data_dict['timestamp']
            # send data to app
            for _, app_process in self.app_processes.items():
                if not self.block_by_app:
                    put_nowait_force(
                        data=data_dict, queue_dst=app_process.mp_queue_src)
                else:
                    app_process.mp_queue_src.put(data_dict, block=True)
            valid_iter_count += 1
            cur_time = time.time()
            time_diff = cur_time - last_profile_time
            if time_diff >= self.profile_period:
                if self.verbose:
                    self.logger.info(
                        f'{self.name} time analysis:'
                        '\nqueue_src_recv_time: ' +
                        f'{queue_src_recv_time/valid_iter_count}' +
                        '\nlatency: ' + f'{latency_time/valid_iter_count}' +
                        f'\nfps: {valid_iter_count/time_diff}' + '\n')
                queue_src_recv_time = 0.0
                latency_time = 0.0
                valid_iter_count = 0
                last_profile_time = cur_time

    def __del__(self):
        pass
