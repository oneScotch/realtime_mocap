import cv2
import numpy as np
import time

from realtime_mocap.stream_source.builder import build_stream_src
from .base_process import BaseProcess


class MPProducerRecordProcess(BaseProcess):

    def __init__(self,
                 imshow: bool = True,
                 record_path: str = 'data/k4a_default_stream_file.npz',
                 max_frame_idx: int = 900,
                 *args,
                 name='multiprocessing_producer_process',
                 stream_src=dict(type='DummyProducer'),
                 logger=None,
                 **kwargs) -> None:
        BaseProcess.__init__(self, *args, name=name, logger=logger, **kwargs)

        self.record_path = record_path
        self.max_frame_idx = max_frame_idx
        self.imshow = imshow

        stream_src['logger'] = self.logger
        self.stream_src_cfg = stream_src
        self.stream_src = None
        self.frame_idx = 0

    def init_stream_src(self):
        self.stream_src = build_stream_src(self.stream_src_cfg)

    def run(self):
        BaseProcess.run(self=self)
        self.init_stream_src()
        last_profile_time = time.time()
        data_get_time = 0.0
        iter_count = 0
        dict_to_dump = {}

        while True:
            # get data dict from stream source
            data_start_time = time.time()
            data_dict = self.stream_src.get_data()
            dict_to_dump[str(self.frame_idx)] = data_dict
            if data_dict is not None and self.imshow:
                img_arr = data_dict['img_arr']
                scale = 720 / img_arr.shape[0]
                target_size = (int(scale * img_arr.shape[1]),
                               int(scale * img_arr.shape[0]))
                vis_image = cv2.resize(
                    img_arr, target_size, interpolation=cv2.INTER_AREA)
                cv2.imshow('img_arr', vis_image)
                cv2.waitKey(1)
            if self.frame_idx == self.max_frame_idx:
                cv2.destroyAllWindows()
                np.savez_compressed(self.record_path, **dict_to_dump)
                break
            if data_dict is None:
                self.frame_idx += 1
                iter_count += 1
                data_get_time += time.time() - data_start_time
                continue
            data_dict['frame_idx'] = self.frame_idx
            data_get_time += time.time() - data_start_time
            iter_count += 1
            self.frame_idx = (self.frame_idx + 1) % 999999999
            cur_time = time.time()
            if cur_time - last_profile_time >= 10.0:
                self.logger.info(
                    f'{self.name} time analysis:'
                    f'\ndata_get_time: {data_get_time/iter_count}' +
                    f'\nfps: {iter_count/(cur_time - last_profile_time)}' +
                    '\n')
                data_get_time = 0.0
                iter_count = 0
                last_profile_time = cur_time

    def __del__(self):
        pass
