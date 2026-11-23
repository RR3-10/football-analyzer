"""Sanity check: print library versions and run YOLO26 on a sample image."""

import platform

import cv2
import numpy as np
import supervision as sv
import torch
import ultralytics
from trackers import ByteTrackTracker  # noqa: F401  (import check only)
from ultralytics import YOLO


def main() -> None:
    print(f"Python      : {platform.python_version()}")
    print(f"torch       : {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    print(f"ultralytics : {ultralytics.__version__}")
    print(f"supervision : {sv.__version__}")
    print(f"opencv      : {cv2.__version__} | numpy: {np.__version__}")
    print("trackers    : ByteTrackTracker import OK")

    # Tiny model, just to prove inference works end to end on this machine.
    model = YOLO("yolo26n.pt")
    result = model("https://ultralytics.com/images/bus.jpg", verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    print(f"\nYOLO26n test: {len(detections)} objects detected -> environment OK")


if __name__ == "__main__":
    main()
    