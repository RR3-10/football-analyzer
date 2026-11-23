"""Pitch keypoint detector (YOLO pose model trained on 32 pitch landmarks)."""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

N_KEYPOINTS = 32


class PitchKeypointDetector:
    def __init__(self, weights: str | Path, device: str = "cpu", imgsz: int = 640) -> None:
        if not Path(weights).exists():
            raise FileNotFoundError(f"Model weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz

    def detect_batch(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each frame: (32, 2) keypoint pixels and (32,) confidences (zeros if no pitch)."""
        results = self.model.predict(frames, imgsz=self.imgsz, device=self.device, verbose=False)
        out = []
        for r in results:
            if r.keypoints is None or len(r.keypoints) == 0 or r.keypoints.conf is None:
                out.append((np.zeros((N_KEYPOINTS, 2)), np.zeros(N_KEYPOINTS)))
                continue
            best = int(r.boxes.conf.argmax()) if r.boxes is not None and len(r.boxes) else 0
            xy = r.keypoints.xy[best].cpu().numpy()
            conf = r.keypoints.conf[best].cpu().numpy()
            out.append((xy, conf))
        return out