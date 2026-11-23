"""Clean up the ball trajectory: drop false detections, then fill short gaps.

The ball is the hardest object to detect (tiny, blurred, hidden by players), so
the raw per-frame positions have holes and occasional false positives.
"""

import numpy as np

from football_analyzer.detection.tracker import FrameTracks


def raw_ball_positions(tracks: list[FrameTracks]) -> np.ndarray:
    """(N, 2) ball centre per frame in pixels, NaN where no ball was detected."""
    xy = np.full((len(tracks), 2), np.nan)
    for i, frame in enumerate(tracks):
        if len(frame.ball):
            x1, y1, x2, y2 = frame.ball.xyxy[0]
            xy[i] = ((x1 + x2) / 2, (y1 + y2) / 2)
    return xy


def remove_outliers(xy: np.ndarray, max_jump_px: float, max_rejections: int = 5) -> np.ndarray:
    """Drop detections that jump impossibly far from the last accepted position.

    A real ball moves at most `max_jump_px` pixels per frame. If we reject
    `max_rejections` detections in a row, we assume the *previous* position was the
    wrong one and accept the new position (this stops one early false positive from
    poisoning the whole clip).
    """
    out = xy.copy()
    last_idx: int | None = None
    rejected_in_a_row = 0
    for i in np.where(~np.isnan(xy[:, 0]))[0]:
        if last_idx is None:
            last_idx = i
            continue
        allowed = max_jump_px * (i - last_idx)
        if np.linalg.norm(xy[i] - out[last_idx]) > allowed and rejected_in_a_row < max_rejections:
            out[i] = np.nan
            rejected_in_a_row += 1
        else:
            last_idx = i
            rejected_in_a_row = 0
    return out


def interpolate_gaps(xy: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill NaN gaps of at most `max_gap` frames that have a known position on both sides.

    Longer gaps are left empty: guessing the ball across 2+ seconds would be invented data.
    Returns (filled positions, boolean mask of frames that were interpolated).
    """
    filled = xy.copy()
    interpolated = np.zeros(len(xy), dtype=bool)
    known = np.where(~np.isnan(xy[:, 0]))[0]
    for a, b in zip(known[:-1], known[1:], strict=True):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            t = np.arange(1, gap + 1) / (gap + 1)  # fractions between the two known frames
            filled[a + 1 : b] = xy[a] + t[:, None] * (xy[b] - xy[a])
            interpolated[a + 1 : b] = True
    return filled, interpolated


def clean_ball_track(
    tracks: list[FrameTracks], max_gap: int, max_jump_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Full ball clean-up: raw positions -> outlier removal -> gap interpolation."""
    xy = remove_outliers(raw_ball_positions(tracks), max_jump_px)
    return interpolate_gaps(xy, max_gap)