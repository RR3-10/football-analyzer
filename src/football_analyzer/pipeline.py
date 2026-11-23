"""End-to-end pipeline.

Pass 1  track_video():  frames -> YOLO -> ByteTrack -> list[FrameTracks]   (slow, cached)
Pass 2  (Phase 2+)      analytics on the tracks                             (fast, no video)
Pass 3  render_video(): frames + tracks -> annotated output video           (medium)

Splitting the work this way means the expensive model only runs once per video.
"""

import time
from pathlib import Path
from typing import Any

import supervision as sv
from tqdm import tqdm

from football_analyzer.detection.detector import Detector
from football_analyzer.detection.tracker import FrameTracks, ObjectTracker
from football_analyzer.utils.cache import cache_file, load_pickle, save_pickle
from football_analyzer.utils.video_io import batched, get_video_info, iter_frames
from football_analyzer.visualization.annotators import TrackAnnotator


def track_video(
    video_path: str | Path,
    cfg: dict[str, Any],
    max_frames: int | None = None,
    use_cache: bool = True,
) -> list[FrameTracks]:
    """Detect and track every frame. Results are cached to disk."""
    info = get_video_info(video_path)
    n_frames = min(info.total_frames, max_frames) if max_frames else info.total_frames

    cache = cache_file(cfg["paths"]["cache_dir"], video_path, f"tracks_{n_frames}")
    if use_cache and cache.exists():
        print(f"[cache] loading tracks from {cache}")
        return load_pickle(cache)

    det_cfg = cfg["detection"]
    detector = Detector(
        cfg["models"]["players"],
        device=str(cfg["models"]["device"]),
        imgsz=det_cfg["imgsz"],
        conf=det_cfg["conf"],
    )
    tracker = ObjectTracker(
        ball_class_id=detector.class_ids["ball"],
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


def render_video(
    video_path: str | Path,
    tracks: list[FrameTracks],
    output_path: str | Path,
) -> Path:
    """Draw the tracks on the original frames and write a new video."""
    info = get_video_info(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotator = TrackAnnotator()

    with sv.VideoSink(str(output_path), video_info=info) as sink:
        frames = iter_frames(video_path, max_frames=len(tracks))
        for frame, frame_tracks in tqdm(
            zip(frames, tracks), total=len(tracks), desc="Render", unit="frame"
        ):
            sink.write_frame(annotator.annotate(frame, frame_tracks))
    return output_path