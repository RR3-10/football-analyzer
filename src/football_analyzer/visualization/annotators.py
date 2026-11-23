"""Draw tracking results on frames: team-coloured ellipses, IDs, ball and ball holder."""

import numpy as np
import supervision as sv

from football_analyzer.detection.tracker import FrameTracks

REFEREE_COLOR = sv.Color.from_hex("#FFD700")   # yellow
UNKNOWN_COLOR = sv.Color.from_hex("#9E9E9E")   # grey (track never sampled for colour)
BALL_COLOR = sv.Color.from_hex("#FFD700")      # ball detected by the model
BALL_INTERP_COLOR = sv.Color.WHITE             # ball position filled by interpolation
HOLDER_COLOR = sv.Color.from_hex("#FF1744")    # red marker above the player on the ball

# palette index used for each team value
_TEAM_TO_INDEX = {0: 0, 1: 1, -1: 2}
_UNKNOWN_INDEX = 3


def bgr_to_sv(color_bgr: tuple[int, int, int]) -> sv.Color:
    b, g, r = color_bgr
    return sv.Color(r=r, g=g, b=b)


def point_detection(xy: np.ndarray, size: float = 6.0) -> sv.Detections:
    """A tiny box around a point, so supervision annotators can draw it."""
    x, y = xy
    return sv.Detections(
        xyxy=np.array([[x - size, y - size, x + size, y + size]]), class_id=np.array([0])
    )


class TrackAnnotator:
    def __init__(self, team_colors_bgr: list[tuple[int, int, int]]) -> None:
        team_a, team_b = (bgr_to_sv(c) for c in team_colors_bgr)
        palette = sv.ColorPalette([team_a, team_b, REFEREE_COLOR, UNKNOWN_COLOR])
        self.ellipse = sv.EllipseAnnotator(color=palette, thickness=2)
        self.label = sv.LabelAnnotator(
            color=palette,
            text_color=sv.Color.BLACK,
            text_position=sv.Position.BOTTOM_CENTER,
            text_scale=0.45,
            text_padding=4,
        )
        self.ball = sv.TriangleAnnotator(color=BALL_COLOR, base=20, height=17, outline_thickness=1)
        self.ball_interp = sv.TriangleAnnotator(
            color=BALL_INTERP_COLOR, base=20, height=17, outline_thickness=1
        )
        self.holder = sv.TriangleAnnotator(
            color=HOLDER_COLOR, base=22, height=20, outline_thickness=1
        )

    def annotate(
        self,
        frame: np.ndarray,
        tracks: FrameTracks,
        track_team: dict[int, int],
        ball_xy: np.ndarray,
        ball_interpolated: bool,
        holder_id: int,
    ) -> np.ndarray:
        out = frame.copy()
        people = tracks.people

        teams = [track_team.get(int(t)) for t in people.tracker_id]
        color_idx = np.array([_TEAM_TO_INDEX.get(t, _UNKNOWN_INDEX) for t in teams], dtype=int)
        out = self.ellipse.annotate(out, people, custom_color_lookup=color_idx)
        labels = [f"#{tid}" for tid in people.tracker_id]
        out = self.label.annotate(out, people, labels=labels, custom_color_lookup=color_idx)

        if holder_id != -1:
            out = self.holder.annotate(out, people[people.tracker_id == holder_id])

        if not np.isnan(ball_xy[0]):
            annotator = self.ball_interp if ball_interpolated else self.ball
            out = annotator.annotate(out, point_detection(ball_xy))
        return out