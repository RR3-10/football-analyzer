"""End-to-end pipeline.

Pass 1  track_video():          frames -> YOLO -> ByteTrack -> tracks         (slow, cached)
        compute_homographies(): frames -> pitch keypoints -> H per frame     (slow, cached)
Pass 2  analyze():              teams, ball clean-up, possession, pitch coords (fast)
Pass 3  render_video():         frames + analysis -> annotated output video  (medium)

Splitting the work this way means the expensive models only run once per video.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv
from tqdm import tqdm

from football_analyzer.analytics.ball import clean_ball_track
from football_analyzer.analytics.possession import PossessionResult, compute_possession
from football_analyzer.detection.postprocess import stabilize_classes
from football_analyzer.detection.tracker import FrameTracks, ObjectTracker
from football_analyzer.geometry.homography import (
    estimate_homography,
    interpolate_homographies,
    remove_temporal_outliers,
    transform_points,
)
from football_analyzer.geometry.pitch import PitchConfig
from football_analyzer.teams.team_assigner import TeamResult, assign_teams
from football_analyzer.utils.cache import cache_file, load_pickle, save_pickle
from football_analyzer.utils.video_io import batched, get_video_info, iter_frames
from football_analyzer.visualization.annotators import TrackAnnotator
from football_analyzer.visualization.overlays import draw_possession_bar
from football_analyzer.visualization.radar import REFEREE, Radar, overlay_bottom_right


@dataclass
class MatchAnalysis:
    tracks: list[FrameTracks]
    teams: TeamResult
    ball_xy: np.ndarray            # (N, 2) cleaned ball position per frame, pixels (NaN = unknown)
    ball_interpolated: np.ndarray  # (N,) True where the ball position was interpolated
    possession: PossessionResult
    pitch: PitchConfig
    homographies: list            # per frame: 3x3 image->pitch matrix, or None
    people_pitch_xy: list         # per frame: (K, 2) player positions in metres
    ball_pitch_xy: np.ndarray     # (N, 2) ball position in metres (assumes ball on the ground)


def _n_frames(video_path: str | Path, max_frames: int | None) -> int:
    total = get_video_info(video_path).total_frames
    return min(total, max_frames) if max_frames else total


def track_video(
    video_path: str | Path,
    cfg: dict[str, Any],
    max_frames: int | None = None,
    use_cache: bool = True,
) -> list[FrameTracks]:
    """Detect and track every frame. Results are cached to disk."""
    info = get_video_info(video_path)
    n_frames = _n_frames(video_path, max_frames)

    cache = cache_file(cfg["paths"]["cache_dir"], video_path, f"tracks_{n_frames}")
    if use_cache and cache.exists():
        print(f"[cache] loading tracks from {cache}")
        return load_pickle(cache)

    # imported here so the analysis/render steps work without loading torch
    from football_analyzer.detection.detector import Detector

    det_cfg = cfg["detection"]
    detector = Detector(
        cfg["models"]["players"],
        device=str(cfg["models"]["device"]),
        imgsz=det_cfg["imgsz"],
        conf=det_cfg["conf"],
    )
    if detector.class_ids != cfg["classes"]:
        raise ValueError(f"Model classes {detector.class_ids} != config classes {cfg['classes']}")

    tracker = ObjectTracker(
        ball_class_id=cfg["classes"]["ball"],
        frame_rate=info.fps,
        people_nms_iou=det_cfg["people_nms_iou"],
        **cfg["tracking"],
    )

    tracks: list[FrameTracks] = []
    start = time.perf_counter()
    frames = iter_frames(video_path, max_frames=n_frames)
    with tqdm(total=n_frames, desc="Detect + track", unit="frame") as bar:
        for batch in batched(frames, det_cfg["batch_size"]):
            for detections in detector.detect_batch(batch):
                tracks.append(tracker.update(detections))
            bar.update(len(batch))
    elapsed = time.perf_counter() - start
    print(f"Processed {len(tracks)} frames in {elapsed:.1f}s -> {len(tracks) / elapsed:.2f} FPS")

    save_pickle(tracks, cache)
    print(f"[cache] saved tracks to {cache}")
    return tracks


def compute_homographies(
    video_path: str | Path,
    cfg: dict[str, Any],
    pitch: PitchConfig,
    max_frames: int | None = None,
    use_cache: bool = True,
) -> list[np.ndarray | None]:
    """Image->pitch homography for every frame (pitch model runs every `keypoint_stride`)."""
    pcfg = cfg["pitch"]
    stride = pcfg["keypoint_stride"]
    n_frames = _n_frames(video_path, max_frames)

    cache = cache_file(cfg["paths"]["cache_dir"], video_path, f"homography_{n_frames}_s{stride}")
    if use_cache and cache.exists():
        print(f"[cache] loading homographies from {cache}")
        return load_pickle(cache)

    from football_analyzer.geometry.keypoints import PitchKeypointDetector

    detector = PitchKeypointDetector(
        cfg["models"]["pitch"], device=str(cfg["models"]["device"]), imgsz=pcfg["keypoint_imgsz"]
    )
    line_points = pitch.line_points()
    info = get_video_info(video_path)
    key_frames, key_H = [], []
    sampled = (
        (idx, frame)
        for idx, frame in enumerate(iter_frames(video_path, max_frames=n_frames))
        if idx % stride == 0
    )
    with tqdm(total=(n_frames + stride - 1) // stride, desc="Pitch keypoints", unit="frame") as bar:
        for batch in batched(sampled, cfg["detection"]["batch_size"]):
            results = detector.detect_batch([frame for _, frame in batch])
            for (idx, frame), (xy, conf) in zip(batch, results, strict=True):
                H = estimate_homography(
                    xy, conf, pitch.vertices,
                    min_conf=pcfg["keypoint_conf"],
                    ransac_threshold_m=pcfg["ransac_threshold_m"],
                    min_inliers=pcfg["min_inliers"],
                    frame=frame,
                    line_points_m=line_points,
                    min_line_score=pcfg["min_line_score"],
                )
                if H is not None:
                    key_frames.append(idx)
                    key_H.append(H)
            bar.update(len(batch))

    n_keys = (n_frames + stride - 1) // stride
    n_found = len(key_frames)
    key_frames, key_H = remove_temporal_outliers(
        key_frames, key_H, (info.width, info.height),
        window=pcfg["temporal_window"], max_jump_m=pcfg["max_jump_m"],
    )
    print(f"Pitch model: homography found on {n_found}/{n_keys} sampled frames, "
          f"{n_found - len(key_frames)} rejected as jumps -> {len(key_frames)} kept")
    homographies = interpolate_homographies(
        key_frames, key_H, n_frames, max_gap=pcfg["max_gap_frames"]
    )
    save_pickle(homographies, cache)
    print(f"[cache] saved homographies to {cache}")
    return homographies


def analyze(
    video_path: str | Path,
    cfg: dict[str, Any],
    max_frames: int | None = None,
    use_cache: bool = True,
) -> MatchAnalysis:
    """Tracking + everything computed from the tracks."""
    tracks = track_video(video_path, cfg, max_frames=max_frames, use_cache=use_cache)
    stabilize_classes(tracks)

    print("Assigning teams by jersey colour...")
    teams = assign_teams(
        str(video_path), tracks, cfg["classes"], sample_stride=cfg["teams"]["sample_stride"]
    )

    ball_xy, ball_interp = clean_ball_track(
        tracks, max_gap=cfg["ball"]["max_gap"], max_jump_px=cfg["ball"]["max_jump_px"]
    )
    possession = compute_possession(tracks, ball_xy, teams.track_team, **cfg["possession"])

    pitch = PitchConfig(length=cfg["pitch"]["length"], width=cfg["pitch"]["width"])
    homographies = compute_homographies(
        video_path, cfg, pitch, max_frames=len(tracks), use_cache=use_cache
    )
    people_pitch_xy = [
        transform_points(
            frame.people.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER), H
        )
        for frame, H in zip(tracks, homographies, strict=True)
    ]
    ball_pitch_xy = np.array(
        [transform_points(xy, H)[0] for xy, H in zip(ball_xy, homographies, strict=True)]
    )
    return MatchAnalysis(
        tracks, teams, ball_xy, ball_interp, possession,
        pitch, homographies, people_pitch_xy, ball_pitch_xy,
    )


def render_video(
    video_path: str | Path,
    analysis: MatchAnalysis,
    output_path: str | Path,
    cfg: dict[str, Any],
) -> Path:
    """Draw the analysis on the original frames and write a new video."""
    info = get_video_info(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = analysis.teams.team_colors_bgr
    team_to_color = {0: colors[0], 1: colors[1], -1: REFEREE}
    annotator = TrackAnnotator(colors)
    radar = Radar(analysis.pitch, width_px=int(info.width * cfg["radar"]["width_ratio"]))
    running_pct = analysis.possession.running_percentages()
    n = len(analysis.tracks)

    with sv.VideoSink(str(output_path), video_info=info) as sink:
        frames = iter_frames(video_path, max_frames=n)
        for i, frame in enumerate(tqdm(frames, total=n, desc="Render", unit="frame")):
            tracks = analysis.tracks[i]
            holder = int(analysis.possession.holder_id[i])
            out = annotator.annotate(
                frame,
                tracks,
                analysis.teams.track_team,
                analysis.ball_xy[i],
                bool(analysis.ball_interpolated[i]),
                holder,
            )
            out = draw_possession_bar(out, tuple(running_pct[i]), colors)

            if analysis.homographies[i] is not None:
                ids = tracks.people.tracker_id
                dot_colors = [
                    team_to_color.get(analysis.teams.track_team.get(int(t)), (158, 158, 158))
                    for t in ids
                ]
                radar_img = radar.render(
                    analysis.people_pitch_xy[i], dot_colors, analysis.ball_pitch_xy[i],
                    highlight=ids == holder,
                )
                out = overlay_bottom_right(out, radar_img, opacity=cfg["radar"]["opacity"])
            sink.write_frame(out)
    return output_path