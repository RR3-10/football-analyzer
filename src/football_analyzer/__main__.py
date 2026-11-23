"""Command-line entry point.

Usage:
    python -m football_analyzer --video data/videos/08fd33_0.mp4 --max-frames 150
"""

import argparse
from pathlib import Path

from football_analyzer.config import load_config
from football_analyzer.pipeline import render_video, track_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Football match analyzer")
    parser.add_argument("--video", required=True, help="path to the input match video")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument("--max-frames", type=int, default=None, help="only process the first N frames")
    parser.add_argument("--no-cache", action="store_true", help="ignore cached detections")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    tracks = track_video(args.video, cfg, max_frames=args.max_frames, use_cache=not args.no_cache)

    out = Path(cfg["paths"]["output_dir"]) / f"{Path(args.video).stem}_tracked.mp4"
    render_video(args.video, tracks, out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()