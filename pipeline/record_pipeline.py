from xrprimer.utils.log_utils import get_logger

from realtime_mocap.mp_process.builder import build_mp_process


class RecordPipeline:

    def __init__(self, producer_process, logger=None) -> None:
        self.logger = get_logger(logger)

        producer_process['logger'] = self.logger
        self.producer_process = build_mp_process(producer_process)

    def run(self):
        self.producer_process.run()
