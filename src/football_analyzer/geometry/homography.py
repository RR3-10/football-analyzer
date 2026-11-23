"""Homography: map image pixels to pitch metres.

A homography is a 3x3 matrix H that maps points of one plane to another plane:
    [x', y', w']^T = H @ [u, v, 1]^T,   pitch point = (x'/w', y'/w')
The pitch is (almost) flat, so one H maps every pixel on the grass to its real
position in metres. H has 8 degrees of freedom, so we need at least 4 point pairs
(each gives 2 equations): pitch keypoints seen in the image <-> known pitch positions.
"""

from itertools import combinations

import cv2
import numpy as np


def white_line_mask(frame: np.ndarray, kernel_px: int = 15, min_contrast: int = 40) -> np.ndarray:
    """Boolean mask of thin bright structures (painted pitch lines).

    A morphological top-hat keeps only bright details narrower than the kernel, so the
    lines survive but large bright areas (sunlit grass, sky) do not. Dilating the mask
    tolerates a few pixels of error.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    mask = (tophat > min_contrast).astype(np.uint8)
    return cv2.dilate(mask, np.ones((5, 5), np.uint8)).astype(bool)


def line_alignment_score(
    H_image_to_pitch: np.ndarray, mask: np.ndarray, line_points_m: np.ndarray, min_points: int = 50
) -> float:
    """Fraction of the projected pitch lines that land on white pixels of the image."""
    if not is_valid_homography(H_image_to_pitch):
        return 0.0
    pts = transform_points(line_points_m, np.linalg.inv(H_image_to_pitch))
    h, w = mask.shape
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    if inside.sum() < min_points:
        return 0.0
    px = pts[inside].astype(int)
    return float(mask[px[:, 1], px[:, 0]].mean())


def is_valid_homography(H: np.ndarray | None, min_abs_det: float = 1e-10) -> bool:
    """False for missing, non-finite or (near-)singular matrices.

    A singular H squashes the plane onto a line or a point: it cannot be inverted and
    is never a real camera view, so it must be rejected instead of crashing later.
    """
    return (
        H is not None
        and bool(np.isfinite(H).all())
        and abs(float(np.linalg.det(H))) > min_abs_det
    )


def _is_degenerate(points_m: np.ndarray, min_area_m2: float = 1.0) -> bool:
    """True if any 3 of the 4 pitch points are (almost) on one line -> H is undetermined."""
    for a, b, c in combinations(points_m, 3):
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
        if area < min_area_m2:
            return True
    return False


def candidate_homographies(
    src: np.ndarray, dst: np.ndarray, threshold_m: float, min_inliers: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Every distinct homography supported by >= `min_inliers` keypoints.

    Like RANSAC, but exhaustive over all 4-point subsets (cheap: a frame has < 20 visible
    keypoints) and returning ALL well-supported solutions instead of just one, so that
    ties can be broken with image evidence.
    """
    found: dict[frozenset, np.ndarray] = {}
    for subset in combinations(range(len(src)), 4):
        idx = list(subset)
        if _is_degenerate(dst[idx]):
            continue
        H = cv2.getPerspectiveTransform(src[idx], dst[idx])
        if not is_valid_homography(H):
            continue
        errors = np.linalg.norm(transform_points(src, H) - dst, axis=1)
        inliers = errors < threshold_m
        if inliers.sum() >= min_inliers:
            found.setdefault(frozenset(np.where(inliers)[0]), inliers)
    candidates = []
    for inliers in found.values():
        H, _ = cv2.findHomography(src[inliers], dst[inliers], 0)  # least squares on inliers
        if H is not None and abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]
            if is_valid_homography(H):
                candidates.append((H, inliers))
    return candidates


def estimate_homography(
    image_points: np.ndarray,
    confidences: np.ndarray,
    pitch_points: np.ndarray,
    min_conf: float = 0.5,
    ransac_threshold_m: float = 1.0,
    min_inliers: int = 5,
    frame: np.ndarray | None = None,
    line_points_m: np.ndarray | None = None,
    min_line_score: float = 0.0,
) -> np.ndarray | None:
    """H (image -> pitch) from the detected keypoints, or None if unreliable.

    1. Find every homography that >= `min_inliers` keypoints agree with (within
       `ransac_threshold_m` metres). Why 5 and not 4: ANY 4 points fit exactly, so 4
       agreeing points prove nothing.
    2. If several candidates exist (e.g. a few keypoints are confidently WRONG and form
       their own consistent group), let the image decide: project the pitch lines with
       each candidate and keep the one whose lines land on real white line pixels.
       Without a frame, fall back to the most inliers.
    """
    visible = confidences > min_conf
    if visible.sum() < min_inliers:
        return None
    src = image_points[visible].astype(np.float32)
    dst = pitch_points[visible].astype(np.float32)
    candidates = candidate_homographies(src, dst, ransac_threshold_m, min_inliers)
    if not candidates:
        return None

    if frame is None or line_points_m is None:
        return max(candidates, key=lambda c: c[1].sum())[0]

    mask = white_line_mask(frame)
    scored = [(line_alignment_score(H, mask, line_points_m), inl.sum(), H) for H, inl in candidates]
    best_score, _, best_H = max(scored, key=lambda s: (s[0], s[1]))
    return best_H if best_score >= min_line_score else None


def camera_footprint(H: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Pitch positions (metres) of 6 fixed points of the image: 'where the camera looks'."""
    w, h = image_size
    grid = np.array(
        [(w * fx, h * fy) for fy in (0.5, 0.8) for fx in (0.25, 0.5, 0.75)], dtype=np.float32
    )
    return transform_points(grid, H)


def remove_temporal_outliers(
    key_frames: list[int],
    key_H: list[np.ndarray],
    image_size: tuple[int, int],
    window: int = 3,
    max_jump_m: float = 5.0,
) -> tuple[list[int], list[np.ndarray]]:
    """Drop key-frame homographies that disagree with their neighbours in time.

    A broadcast camera pans smoothly, so where it looks on the pitch changes only a few
    metres between nearby key frames. For each key frame we compare its camera footprint
    with the MEDIAN footprint of up to `window` key frames on each side (the median
    ignores a minority of bad neighbours). If it is more than `max_jump_m` metres away on
    average, this homography is a one-off error (e.g. wrongly detected keypoints) and is
    dropped; interpolation then fills the gap from the good neighbours.
    """
    if len(key_frames) < 3:
        return key_frames, key_H
    footprints = np.stack([camera_footprint(H, image_size) for H in key_H])  # (K, 6, 2)
    keep = np.ones(len(key_frames), dtype=bool)
    for k in range(len(key_frames)):
        neighbours = [
            j for j in range(max(0, k - window), min(len(key_frames), k + window + 1))
            if j != k and abs(key_frames[j] - key_frames[k]) <= window * 10
        ]
        if len(neighbours) < 2:
            continue
        reference = np.nanmedian(footprints[neighbours], axis=0)
        deviation = np.nanmean(np.linalg.norm(footprints[k] - reference, axis=1))
        if not np.isfinite(deviation) or deviation > max_jump_m:
            keep[k] = False
    return [f for f, ok in zip(key_frames, keep, strict=True) if ok], [
        H for H, ok in zip(key_H, keep, strict=True) if ok
    ]


def interpolate_homographies(
    key_frames: list[int], key_H: list[np.ndarray], n_frames: int, max_gap: int = 25
) -> list[np.ndarray | None]:
    """One H per frame from H's computed on some frames only (the 'key frames').

    - Two key frames at most `max_gap` frames apart: blend their matrices linearly
      (the camera moves smoothly, so this is a good approximation; it also smooths jitter).
    - Otherwise a frame may borrow the nearest key frame's H only if that key frame is
      within `max_gap // 2` frames. Further away, the camera has probably moved, so an old
      H would put players in the WRONG place: return None (no radar) instead.
    """
    out: list[np.ndarray | None] = [None] * n_frames
    if not key_frames:
        return out
    keys = np.asarray(key_frames)
    hold = max_gap // 2
    for i in range(n_frames):
        j = int(np.searchsorted(keys, i, side="right")) - 1  # last key frame <= i
        prev_ok = j >= 0
        next_ok = j + 1 < len(keys)
        if prev_ok and keys[j] == i:
            out[i] = key_H[j]
        elif prev_ok and next_ok and keys[j + 1] - keys[j] <= max_gap:
            a, b = keys[j], keys[j + 1]
            t = (i - a) / (b - a)
            H = (1 - t) * key_H[j] + t * key_H[j + 1]
            out[i] = H / H[2, 2]
        else:
            candidates = []
            if prev_ok:
                candidates.append((i - keys[j], key_H[j]))
            if next_ok:
                candidates.append((keys[j + 1] - i, key_H[j + 1]))
            dist, H = min(candidates, key=lambda c: c[0])
            out[i] = H if dist <= hold else None
    return out


def transform_points(points: np.ndarray, H: np.ndarray | None) -> np.ndarray:
    """(K, 2) image points -> (K, 2) pitch points in metres (NaN if H is unknown)."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if H is None or len(points) == 0:
        return np.full((len(points), 2), np.nan, dtype=np.float32)
    out = np.full((len(points), 2), np.nan, dtype=np.float32)
    valid = ~np.isnan(points).any(axis=1)
    if valid.any():
        out[valid] = cv2.perspectiveTransform(points[valid].reshape(-1, 1, 2), H).reshape(-1, 2)
    return out