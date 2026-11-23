"""YOLO object detector: ball, goalkeeper, player, referee."""

from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO


class Detector:
    """Thin wrapper around an Ultralytics YOLO model that returns `sv.Detections`."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        imgsz: int = 1280,
        conf: float = 0.25,
    ) -> None:
        if not Path(weights).exists():
            raise FileNotFoundError(f"Model weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        # {0: 'ball', 1: 'goalkeeper', ...} -> {'ball': 0, 'goalkeeper': 1, ...}
        self.class_ids: dict[str, int] = {name: idx for idx, name in self.model.names.items()}

    def detect_batch(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        """Run the model on several frames in one forward pass (faster than one by one)."""
        results = self.model.predict(
            frames, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )
        return [sv.Detections.from_ultralytics(r) for r in results]