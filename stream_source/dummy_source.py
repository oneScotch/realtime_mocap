import numpy as np
import time

from .base_source import BaseSource


class DummySource(BaseSource):
    BASE_TIME = 0.033333

    def __init__(self, logger) -> None:
        BaseSource.__init__(self=self, logger=logger)
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
        img = np.random.randint(
            low=0, high=255, size=(3, 1080, 1920), dtype=np.uint8)
        # np.zeros(shape=(3, 1080, 1920), dtype=np.uint8)
        pose = np.zeros(shape=(1, 1, 17, 4))
        time_stamp = str(time.time())
        data_dict = {'img_arr': img, 'timestamp': time_stamp, 'k4a_pose': pose}
        return data_dict
