import cv2
import multiprocessing
import time
from typing import Union

from realtime_mocap.stream_source.builder import build_stream_src
from realtime_mocap.utils.mp_utils import put_nowait_force
from .base_process import BaseProcess


class MPProducerProcess(BaseProcess):

    def __init__(self,
                 mp_queue_dst: multiprocessing.Queue,
                 *args,
                 name='multiprocessing_producer_process',
                 stream_src=dict(type='DummyProducer'),
                 verbose: bool = True,
                 profile_period: float = 10.0,
                 window_size: Union[int, None] = None,
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
        self.mp_queue_dst = mp_queue_dst
        self.stream_src_cfg = stream_src
        self.stream_src = None
        self.frame_idx = 0
        self.window_size = window_size

    def init_stream_src(self):
        self.stream_src_cfg['logger'] = self.logger
        self.stream_src = build_stream_src(self.stream_src_cfg)

    def run(self):
        BaseProcess.run(self=self)
        self.init_stream_src()
        last_profile_time = time.time()
        data_get_time = 0.0
        queue_dst_send_time = 0.0
        iter_count = 0

        while True:
            # get data dict from stream source
            data_start_time = time.time()
            data_dict = self.stream_src.get_data()
            if data_dict is None:
                self.frame_idx += 1
                iter_count += 1
                data_get_time += time.time() - data_start_time
                continue
            data_dict['frame_idx'] = self.frame_idx
            data_get_time += time.time() - data_start_time
            # send data to queue
            send_start_time = time.time()
            put_nowait_force(data=data_dict, queue_dst=self.mp_queue_dst)
            queue_dst_send_time += time.time() - send_start_time
            iter_count += 1
            self.frame_idx = (self.frame_idx + 1) % 999999999
            cur_time = time.time()
            if self.window_size is not None:
                img_arr = data_dict['img_arr']
                scale = self.window_size / img_arr.shape[0]
                target_size = (int(scale * img_arr.shape[1]),
                               int(scale * img_arr.shape[0]))
                vis_image = cv2.resize(
                    img_arr, target_size, interpolation=cv2.INTER_AREA)
                cv2.imshow(self.name, vis_image)
                cv2.waitKey(1)
            if cur_time - last_profile_time >= self.profile_period:
                if self.verbose:
                    self.logger.info(
                        f'{self.name} time analysis:'
                        '\ndata_get_time: ' + f'{data_get_time/iter_count}' +
                        '\nqueue_dst_send_time: ' +
                        f'{queue_dst_send_time/iter_count}' +
                        f'\nfps: {iter_count/(cur_time - last_profile_time)}' +
                        '\n')
                data_get_time = 0.0
                queue_dst_send_time = 0.0
                iter_count = 0
                last_profile_time = cur_time

    def __del__(self):
        pass
