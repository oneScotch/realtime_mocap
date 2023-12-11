import multiprocessing
from xrprimer.utils.log_utils import get_logger

from realtime_mocap.mp_process.builder import build_mp_process


class LivePipeline:

    def __init__(self,
                 producer_process,
                 stream_queue_len,
                 mocap_process,
                 mocap_queue_len,
                 app_process,
                 logger=None) -> None:
        self.logger = get_logger(logger)

        self.stream_queue = multiprocessing.Queue(maxsize=stream_queue_len)
        self.mocap_queue = multiprocessing.Queue(maxsize=mocap_queue_len)

        producer_process['mp_queue_dst'] = self.stream_queue
        self.producer_process = build_mp_process(producer_process)

        mocap_process['mp_queue_src'] = self.stream_queue
        mocap_process['mp_queue_dst'] = self.mocap_queue
        self.mocap_process = build_mp_process(mocap_process)

        app_process['mp_queue_src'] = self.mocap_queue
        self.app_process = build_mp_process(app_process)

    def run(self):
        self.app_process.start()
        self.mocap_process.start()
        self.producer_process.start()

        self.app_process.join()
