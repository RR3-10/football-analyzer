"""Command-line entry point.

Usage:
    python -m football_analyzer --video data/videos/08fd33_0.mp4 --max-frames 150
"""

import argparse
from pathlib import Path

import numpy as np

from football_analyzer.config import load_config
from football_analyzer.pipeline import analyze, render_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Football match analyzer")
    parser.add_argument("--video", required=True, help="path to the input match video")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="only process the first N frames"
    )
    parser.add_argument("--no-cache", action="store_true", help="ignore cached detections")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    analysis = analyze(args.video, cfg, max_frames=args.max_frames, use_cache=not args.no_cache)

    n = len(analysis.tracks)
    detected = int(np.sum(~np.isnan(analysis.ball_xy[:, 0]) & ~analysis.ball_interpolated))
    filled = int(analysis.ball_interpolated.sum())
    pct = analysis.possession.percentages()
    teams = analysis.teams.track_team
    print("\n===== Summary =====")
    print(f"Frames analysed      : {n}")
    print(f"Ball detected        : {detected} frames ({100 * detected / n:.0f}%)")
    print(f"Ball interpolated    : {filled} frames ({100 * filled / n:.0f}%)")
    print(f"Tracks per team      : team 1 = {sum(t == 0 for t in teams.values())}, "
          f"team 2 = {sum(t == 1 for t in teams.values())}, "
          f"referees = {sum(t == -1 for t in teams.values())}")
    print(f"Team colours (BGR)   : {analysis.teams.team_colors_bgr}")
    print(f"Possession           : team 1 = {pct[0]:.1f}%, team 2 = {pct[1]:.1f}%")
    covered = sum(H is not None for H in analysis.homographies)
    print(f"Pitch homography     : {covered}/{n} frames ({100 * covered / n:.0f}%)")

    out = Path(cfg["paths"]["output_dir"]) / f"{Path(args.video).stem}_analysis.mp4"
    render_video(args.video, analysis, out, cfg)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()