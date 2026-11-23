"""Visual check of the pitch model + homography on ONE frame.

Prints each keypoint's confidence and every candidate homography with its
line-alignment score (the best-scoring candidate is chosen), then draws:
(1) the detected keypoints with their numbers, (2) the pitch lines projected back
onto the frame with the chosen homography (magenta), and (3) the empty radar.
If the magenta lines sit on the real white lines, the homography is correct.

Usage:
    python scripts/check_homography.py --video data/videos/08fd33_0.mp4 --frame 100
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from football_analyzer.config import load_config
from football_analyzer.geometry.homography import (
    candidate_homographies,
    estimate_homography,
    line_alignment_score,
    white_line_mask,
)
from football_analyzer.geometry.keypoints import PitchKeypointDetector
from football_analyzer.geometry.pitch import PitchConfig
from football_analyzer.visualization.radar import Radar, draw_projected_pitch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    pcfg = cfg["pitch"]

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {args.frame} from {args.video}")

    pitch = PitchConfig(length=pcfg["length"], width=pcfg["width"])
    detector = PitchKeypointDetector(cfg["models"]["pitch"], imgsz=pcfg["keypoint_imgsz"])
    xy, conf = detector.detect_batch([frame])[0]
    visible = conf > pcfg["keypoint_conf"]

    # 1) keypoint confidences
    print(f"\nFrame {args.frame}: {visible.sum()}/32 keypoints above conf {pcfg['keypoint_conf']}")
    print(" kp | conf")
    for i in np.argsort(-conf):
        if conf[i] >= 0.2:
            flag = "" if visible[i] else "   (ignored: low conf)"
            print(f" {i + 1:2d} | {conf[i]:.2f}{flag}")

    # 2) every homography that >= min_inliers keypoints agree with, and its line score
    line_points = pitch.line_points()
    mask = white_line_mask(frame)
    candidates = candidate_homographies(
        xy[visible].astype(np.float32), pitch.vertices[visible],
        pcfg["ransac_threshold_m"], pcfg["min_inliers"],
    )
    print(f"\nCandidate homographies: {len(candidates)}")
    for H_c, inliers in candidates:
        ids = [int(i) + 1 for i in np.where(visible)[0][inliers]]
        score = line_alignment_score(H_c, mask, line_points)
        print(f"  agreeing keypoints {ids} -> line score {score:.2f}")

    # 3) the chosen homography
    H = estimate_homography(
        xy, conf, pitch.vertices,
        min_conf=pcfg["keypoint_conf"],
        ransac_threshold_m=pcfg["ransac_threshold_m"],
        min_inliers=pcfg["min_inliers"],
        frame=frame,
        line_points_m=line_points,
        min_line_score=pcfg["min_line_score"],
    )

    out = frame.copy()
    for i in np.where(visible)[0]:
        x, y = xy[i].astype(int)
        cv2.circle(out, (x, y), 7, (0, 255, 255), -1)
        cv2.putText(out, str(i + 1), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

    if H is None:
        print("\nHomography: NOT reliable on this frame -> the video borrows it from "
              "neighbouring frames. Try another --frame.")
    else:
        score = line_alignment_score(H, mask, line_points)
        print(f"\nHomography: OK (line score {score:.2f}) - "
              "check that the magenta lines sit on the real pitch lines.")
        out = draw_projected_pitch(out, H, pitch)
        radar = Radar(pitch, width_px=int(frame.shape[1] * 0.32)).render(
            np.empty((0, 2)), [], None
        )
        rh, rw = radar.shape[:2]
        out[-rh - 20 : -20, -rw - 20 : -20] = radar

    out_path = Path("outputs") / f"homography_check_{args.frame}.jpg"
    out_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"Saved -> {out_path}")
    cv2.imshow("Homography check (any key to close)", cv2.resize(out, None, fx=0.6, fy=0.6))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()