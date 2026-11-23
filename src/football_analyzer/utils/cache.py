"""Disk cache for expensive results (detection + tracking).

Running YOLO on a CPU is the slowest step by far, so we run it once per video
and reuse the saved result while developing the later stages.
"""

import pickle
from pathlib import Path
from typing import Any


def cache_file(cache_dir: str | Path, video_path: str | Path, tag: str) -> Path:
    """e.g. data/cache/08fd33_0_tracks_300.pkl"""
    return Path(cache_dir) / f"{Path(video_path).stem}_{tag}.pkl"


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)