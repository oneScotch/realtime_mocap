import time
from xrmocap.data_structure.body_model.smplx_data import SMPLXData

from .base_source import BaseSource


class SMPLXFileSource(BaseSource):
    BASE_TIME = 0.033333

    def __init__(self, smplx_data_path: str, logger=None) -> None:
        BaseSource.__init__(self=self, logger=logger)
        self.smplx_data_path = smplx_data_path
        self.smplx_data = SMPLXData.fromfile(smplx_data_path)
        self.frame_idx = 0
        self.prev_time = None

    def get_data(self, **kwargs):
        if self.prev_time is None:
            self.prev_time = time.time()
        else:
            time_to_wait = self.__class__.BASE_TIME + \
                self.prev_time - time.time()
            if time_to_wait > 0:
                time.sleep(time_to_wait)
            self.prev_time = time.time()
        # might be None
        ret_dict = dict()
        fullpose = self.smplx_data.get_fullpose()[
            self.frame_idx:self.frame_idx + 1, ...].copy()
        transl = self.smplx_data.get_transl()[self.frame_idx:self.frame_idx +
                                              1, ...].copy()
        betas = self.smplx_data.get_betas(
            repeat_betas=True)[self.frame_idx:self.frame_idx + 1, ...].copy()
        smplx_data = SMPLXData(fullpose=fullpose, transl=transl, betas=betas)
        if self.frame_idx >= self.smplx_data.get_batch_size() - 1:
            self.frame_idx = 0
        else:
            self.frame_idx += 1
        ret_dict['smplx_data'] = smplx_data
        ret_dict['timestamp'] = time.time()
        return ret_dict
