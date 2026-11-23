"""Multi-object tracking with ByteTrack.

People (players, goalkeepers, referees) are tracked together by ByteTrack so that
an ID survives when the detector flips a box between 'player' and 'goalkeeper'.
The ball is NOT tracked by ByteTrack: it is small, fast and often missed, so IoU
matching between frames fails. We keep the single most confident ball box per
frame and fill the gaps later with interpolation (Phase 2).
"""

from dataclasses import dataclass

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


@dataclass
class FrameTracks:
    """Everything we know about one frame after tracking."""

    people: sv.Detections  # players + goalkeepers + referees, each with a tracker_id
    ball: sv.Detections    # 0 or 1 box, no tracker_id


class ObjectTracker:
    def __init__(
        self,
        ball_class_id: int,
        frame_rate: float,
        people_nms_iou: float = 0.6,
        **bytetrack_kwargs,
    ) -> None:
        self.ball_class_id = ball_class_id
        self.people_nms_iou = people_nms_iou
        self.tracker = ByteTrackTracker(frame_rate=frame_rate, **bytetrack_kwargs)

    def update(self, detections: sv.Detections) -> FrameTracks:
        """Split ball / people, track the people, return both for this frame."""
        is_ball = detections.class_id == self.ball_class_id

        ball = detections[is_ball]
        if len(ball) > 1:  # keep only the most confident ball candidate
            ball = ball[[int(np.argmax(ball.confidence))]]

        people = detections[~is_ball]
        # The same person can get two boxes with different classes (player AND goalkeeper).
        # Class-agnostic NMS keeps the most confident one.
        people = people.with_nms(threshold=self.people_nms_iou, class_agnostic=True)
        people = self.tracker.update(people)
        people = people[people.tracker_id != -1]  # -1 = track not confirmed yet

        return FrameTracks(people=people, ball=ball)