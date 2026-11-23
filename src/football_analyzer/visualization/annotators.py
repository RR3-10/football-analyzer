"""Draw tracking results on frames (broadcast-style ellipses under the players)."""

import numpy as np
import supervision as sv

from football_analyzer.detection.tracker import FrameTracks

# Palette is indexed by class id: 0 ball, 1 goalkeeper, 2 player, 3 referee.
# (In Phase 2 players will be coloured by team instead of by class.)
CLASS_PALETTE = sv.ColorPalette.from_hex(["#FFD700", "#00BFFF", "#FF1493", "#FFD700"])
BALL_COLOR = sv.Color.from_hex("#FFD700")


class TrackAnnotator:
    def __init__(self) -> None:
        self.ellipse = sv.EllipseAnnotator(color=CLASS_PALETTE, thickness=2)
        self.label = sv.LabelAnnotator(
            color=CLASS_PALETTE,
            text_color=sv.Color.BLACK,
            text_position=sv.Position.BOTTOM_CENTER,
            text_scale=0.45,
            text_padding=4,
        )
        self.ball = sv.TriangleAnnotator(
            color=BALL_COLOR, base=20, height=17, outline_thickness=1
        )

    def annotate(self, frame: np.ndarray, tracks: FrameTracks) -> np.ndarray:
        out = frame.copy()
        people = tracks.people
        out = self.ellipse.annotate(out, people)
        labels = [f"#{tid}" for tid in people.tracker_id]
        out = self.label.annotate(out, people, labels=labels)
        out = self.ball.annotate(out, tracks.ball)
        return out