# ⚽ Football Match Analyzer

Computer vision pipeline that turns broadcast football footage into an annotated video
with **player tracking, automatic team assignment, ball possession and a live bird's-eye radar**.

> 🚧 **Work in progress.** Phases 1–3 are done (detection, tracking, teams, possession, radar).
> Next: speed & distance, passes, player cards, demo app.

## Features (done)
- **Detection:** fine-tuned **YOLO26m** at 1280 px for players, goalkeepers, referees and the ball
- **Tracking:** ByteTrack with stable IDs, plus majority-vote class stabilisation (goalkeeper vs player)
- **Team assignment (unsupervised):** jersey colour per track → K-means (k=3) in LAB colour space;
  the two largest clusters are the teams; goalkeepers are assigned by position
- **Ball:** physics-based outlier removal + gap interpolation
- **Possession:** nearest player to the ball (perspective-scaled threshold), live possession bar
- **Bird's-eye radar:** **YOLO26m-pose** detects 32 pitch keypoints → homography → players in real metres
  - robust homography: confidence filter, exhaustive candidate search,
    **white-line alignment scoring** to break ties, and **temporal consistency** checks

## Model results
| Model | Task | Metric | Score |
|---|---|---|---|
| YOLO26m @1280 | players / GK / referee / ball | mAP50 (val) | 0.91 |
| YOLO26m-pose @640 | 32 pitch keypoints | keypoint mAP50 (test) | 0.95 |

Inference speed: **0.73 FPS on a laptop CPU** (no GPU); GPU benchmarks coming.

## How it works
```
video → YOLO (detect) → ByteTrack (IDs) → team colours (K-means) → ball clean-up → possession
      → pitch keypoints → homography (pixels → metres) → radar + overlays → output video
```
Detections and homographies are **cached**, so the slow models run once per video.

## How to run
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
pip install -e .
python -m football_analyzer --video data/videos/08fd33_0.mp4 --max-frames 150
```
Model weights are not stored in the repo. Train them with `notebooks/train_colab.ipynb`
(free Colab T4 GPU) and place `players.pt` and `pitch.pt` in `models/`.

Check the homography on a single frame:
```bash
python scripts/check_homography.py --video data/videos/08fd33_0.mp4 --frame 100
```

## Engineering notes (problems found and fixed)
- **K-means split "goalkeeper vs everyone" instead of two teams** → per-track median colours, k=3, keepers by position.
- **RANSAC tie between a correct and a wrong homography** (false keypoints + collinear halfway-line points)
  → candidate search + white-line alignment score; confidence threshold set from measured data.
- **Stale / outlier homographies** → limited reuse window + temporal consistency check.

## Known limitations
- Single camera, no depth: a high ball passing "over" a player can be counted as possession.
- Track IDs can change when a player leaves the frame for a long time (no re-identification).
- Radar is hidden on frames where no reliable homography is found.

## Roadmap
- [ ] Possession fix for balls in flight (speed + persistence check)
- [ ] Speed (km/h) and distance (m) per player
- [ ] Pass detection, info panel, player cards
- [ ] Unit tests + CI, stats export (CSV/JSON + charts)
- [ ] Demo GIF, local-league generalisation test

## Tech stack
Python 3.12 · Ultralytics YOLO26 · supervision · trackers (ByteTrack) · OpenCV · scikit-learn · NumPy