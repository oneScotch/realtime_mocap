import multiprocessing
import queue
from typing import Any


def put_nowait_force(data: Any, queue_dst: multiprocessing.Queue):
    """Put data into queue_dst without waiting. If queue_dst is full, get the
    first one to make an empty slot. It only works when the caller is the only
    producer for queue_dst.

    Args:
        data (Any):
            Data to put.
        queue_dst (multiprocessing.Queue):
            Target queue.
    """
    # try to send data
    try:
        queue_dst.put_nowait(data)
    except queue.Full:
        # failed, try to remove the first one
        try:
            queue_dst.get_nowait()
        except queue.Empty:
            pass
        # after trying, one slot is available
        queue_dst.put(data)
