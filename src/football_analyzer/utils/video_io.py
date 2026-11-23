"""Video reading helpers.

Frames are streamed one at a time instead of loaded into memory:
a 30 s 1080p clip is ~750 frames x 6 MB = ~4.5 GB of RAM if loaded at once.
"""

from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path

import numpy as np
import supervision as sv


def get_video_info(video_path: str | Path) -> sv.VideoInfo:
    """Resolution, FPS and frame count of a video."""
    return sv.VideoInfo.from_video_path(str(video_path))


def iter_frames(video_path: str | Path, max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield BGR frames one by one, optionally stopping after `max_frames`."""
    yield from sv.get_video_frames_generator(str(video_path), end=max_frames)


def batched(items: Iterable, size: int) -> Iterator[list]:
    """Group an iterable into lists of `size` items (the last list may be shorter)."""
    iterator = iter(items)
    while batch := list(islice(iterator, size)):
        yield batch