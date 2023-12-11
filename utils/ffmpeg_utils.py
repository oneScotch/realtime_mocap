import numpy as np
from mmcv.utils import Registry
from xrprimer.utils.ffmpeg_utils import VideoWriter

VIDEO_WRITER = Registry('video_writer')
VIDEO_WRITER.register_module(name='VideoWriter', module=VideoWriter)


def build_video_writer(cfg) -> VideoWriter:
    """Build video_writer."""
    return VIDEO_WRITER.build(cfg)


def try_to_write_frame(video_writer: VideoWriter, img_arr: np.ndarray):
    if video_writer.len >= video_writer.n_frames:
        return False
    if video_writer is not None:
        video_writer.write(img_arr)
    return True
