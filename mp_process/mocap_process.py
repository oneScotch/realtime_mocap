import multiprocessing
import queue
import time
from typing import Union

from .base_process import BaseProcess
from .sub_mp_process.builder import build_sub_mp_process


class MoCapProcess(BaseProcess):

    def __init__(self,
                 mp_queue_src: multiprocessing.Queue,
                 mp_queue_dst: multiprocessing.Queue,
                 estimator_processes: dict,
                 aggregator_process: dict,
                 *args,
                 name='multiprocessing_mocap_process',
                 preperception_process: Union[dict, None] = None,
                 optimizer_process: Union[dict, None] = None,
                 wait_time: float = 0.03,
                 preperception_queue_len=5,
                 estimator_queue_len=5,
                 aggregator_queue_len=5,
                 verbose: bool = True,
                 profile_period: float = 10.0,
                 logger: Union[None, str] = None,
                 **kwargs) -> None:
        """在线动捕进程。

        Args:
            mp_queue_src (multiprocessing.Queue):
                动捕进程整体的输入队列。
            mp_queue_dst (multiprocessing.Queue):
                动捕进程整体的输出队列。
            estimator_processes (dict):
                多个预测次级进程，字典的每个value是一个EstimatorProcess的
                配置字典。
            aggregator_process (dict):
                合并次级进程配置字典，用于合并多个EstimatorProcess的结果。
            name (str, optional):
                进程名称。
                Defaults to 'multiprocessing_mocap_process'.
            preperception_process (Union[dict, None], optional):
                预感知次级进程配置字典。若非None，动捕进程输入先交给
                预感知次级进程处理，再分发给多个预测次级进程。
                若为None，输入直接分发给多个预测次级进程。
                Defaults to None.
            optimizer_process (Union[dict, None], optional):
                优化器次级进程配置字典。若非None，合并次级进程的输出
                交给优化器进程处理，再送入动捕进程整体的输出队列。
                若为None，则跳过优化器进程，合并后直接输出。
                Defaults to None.
            wait_time (float, optional):
                从输入队列取数据的超时时间。 Defaults to 0.03.
            preperception_queue_len (int, optional):
                预感知次级进程的输入队列长度。Defaults to 10.
            estimator_queue_len (int, optional):
                多个预测次级进程的输入输出队列长度。Defaults to 10.
            aggregator_queue_len (int, optional):
                合并次级进程的输出队列长度。Defaults to 10.
            verbose (bool, optional):
                是否在log中记录耗时分析。Defaults to True.
            profile_period (float, optional):
                耗时分析的记录周期。Defaults to 10.0.
            logger (Union[None, str], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        BaseProcess.__init__(
            self,
            *args,
            name=name,
            logger=logger,
            verbose=verbose,
            profile_period=profile_period,
            **kwargs)
        self.mp_queue_src = mp_queue_src
        self.mp_queue_dst = mp_queue_dst
        self.wait_time = wait_time

        # queue len attr
        self.preperception_queue_len = preperception_queue_len \
            if preperception_process is not None \
            else None
        self.estimator_queue_len = estimator_queue_len
        self.aggregator_queue_len = aggregator_queue_len

        # sub_process and queue attr
        self.preperception_queue = None
        self.preperception_process_cfg = preperception_process
        self.preperception_process = None

        self.estimator_queues = None
        self.estimator_processes_cfg = {}
        for key, estimator_process_cfg in estimator_processes.items():
            self.estimator_processes_cfg[key] = estimator_process_cfg
        self.estimator_processes = None

        self.aggregator_queue = None
        self.aggregator_process_cfg = aggregator_process
        self.aggregator_process = None

        self.optimizer_process_cfg = optimizer_process
        self.optimizer_process = None

    def init_queues(self):
        if self.preperception_queue_len is not None:
            self.preperception_src_queue = multiprocessing.Queue(
                maxsize=self.preperception_queue_len)
        self.estimator_queues = {}
        for k in self.estimator_processes_cfg.keys():
            self.estimator_queues[k] = {}
            mp_queue_src = multiprocessing.Queue(
                maxsize=self.estimator_queue_len)
            mp_queue_dst = multiprocessing.Queue(
                maxsize=self.estimator_queue_len)
            self.estimator_queues[k]['mp_queue_src'] = mp_queue_src
            self.estimator_queues[k]['mp_queue_dst'] = mp_queue_dst
        if self.optimizer_process_cfg is not None:
            self.aggregator_queue = multiprocessing.Queue(
                maxsize=self.aggregator_queue_len)

    def init_preperception_process(self):
        if self.preperception_process_cfg is not None:
            mp_queue_dst_dict = {}
            for k in self.estimator_queues.keys():
                mp_queue_dst_dict[k] = self.estimator_queues[k]['mp_queue_src']
            self.preperception_process_cfg[
                'mp_queue_dst_dict'] = mp_queue_dst_dict
            self.preperception_process_cfg[
                'mp_queue_src'] = self.preperception_src_queue
            self.preperception_process = build_sub_mp_process(
                self.preperception_process_cfg)
            self.preperception_process.start()

    def init_estimator_processes(self):
        self.estimator_processes = {}
        for k, v, in self.estimator_processes_cfg.items():
            v['mp_queue_src'] = self.estimator_queues[k]['mp_queue_src']
            v['mp_queue_dst'] = self.estimator_queues[k]['mp_queue_dst']
            self.estimator_processes[k] = build_sub_mp_process(v)
            self.estimator_processes[k].start()

    def init_aggregator_process(self):
        mp_queue_src_dict = {}
        for k in self.estimator_queues.keys():
            mp_queue_src_dict[k] = self.estimator_queues[k]['mp_queue_dst']
        # aggregator sends data to optimizer
        if self.optimizer_process_cfg is not None:
            self.aggregator_process_cfg['mp_queue_dst'] = self.aggregator_queue
            self.aggregator_process_cfg['force_push'] = False
        # aggregator sends data directly to the next stage
        else:
            self.aggregator_process_cfg['mp_queue_dst'] = self.mp_queue_dst
            self.aggregator_process_cfg['force_push'] = True
        self.aggregator_process_cfg['mp_queue_src_dict'] = mp_queue_src_dict
        self.aggregator_process = build_sub_mp_process(
            self.aggregator_process_cfg)
        self.aggregator_process.start()

    def init_optimizer_process(self):
        if self.optimizer_process_cfg is not None:
            self.optimizer_process_cfg['force_push'] = True
            self.optimizer_process_cfg['mp_queue_dst'] = self.mp_queue_dst
            self.optimizer_process_cfg['mp_queue_src'] = self.aggregator_queue
            self.optimizer_process = build_sub_mp_process(
                self.optimizer_process_cfg)
            self.optimizer_process.start()

    def run(self):
        BaseProcess.run(self=self)
        self.init_queues()
        self.init_optimizer_process()
        self.init_aggregator_process()
        self.init_estimator_processes()
        self.init_preperception_process()
        last_profile_time = time.time()
        queue_src_recv_time = 0.0
        latency_time = 0.0
        valid_iter_count = 0
        total_iter_count = 0
        while True:
            # get data dict from stream source
            if self.mocap_full():
                time.sleep(self.wait_time)
                continue
            queue_src_start_time = time.time()
            try:
                data_dict = self.mp_queue_src.get(
                    block=True, timeout=self.wait_time)
            # if there's no input data,
            # no need to estimate
            except queue.Empty:
                continue
            recv_time = time.time()
            total_iter_count += 1
            queue_src_recv_time += recv_time - queue_src_start_time
            # estimate mocap results
            valid_iter_count += 1
            latency_time += recv_time - data_dict['timestamp']
            if self.preperception_process is not None:
                self.preperception_src_queue.put(data_dict, block=True)
            else:
                for _, est_process in self.estimator_processes.items():
                    est_process.mp_queue_src.put(data_dict, block=True)
            cur_time = time.time()
            time_diff = cur_time - last_profile_time
            if time_diff >= self.profile_period:
                if self.verbose:
                    self.logger.info(
                        f'{self.name} time analysis:'
                        '\nqueue_src_recv_time: ' +
                        f'{queue_src_recv_time/total_iter_count}' + '\nfps: ' +
                        f'{valid_iter_count/time_diff}' + '\nlatency: ' +
                        f'{latency_time/valid_iter_count}' + '\n')
                queue_src_recv_time = 0.0
                latency_time = 0.0
                valid_iter_count = 0
                total_iter_count = 0
                last_profile_time = cur_time

    def __del__(self):
        pass

    def mocap_full(self):
        full_flag = False
        if self.preperception_process is not None:
            full_flag = self.preperception_src_queue.full()
        else:
            for _, est_process in self.estimator_processes.items():
                full_flag = full_flag or est_process.mp_queue_src.full()
        return full_flag
