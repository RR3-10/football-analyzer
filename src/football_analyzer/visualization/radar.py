"""Bird's-eye 'radar' view: a top-down 2D pitch with every player as a dot."""

import cv2
import numpy as np

from football_analyzer.geometry.homography import transform_points
from football_analyzer.geometry.pitch import PitchConfig

GRASS = (60, 120, 50)        # BGR
LINE = (235, 235, 235)
REFEREE = (0, 215, 255)      # yellow in BGR
BALL = (255, 255, 255)


class Radar:
    """Draws the pitch once, then the players on a copy of it for every frame."""

    def __init__(self, pitch: PitchConfig, width_px: int, padding_px: int = 12) -> None:
        self.pitch = pitch
        self.pad = padding_px
        self.scale = (width_px - 2 * padding_px) / pitch.length  # pixels per metre
        self.size = (width_px, int(pitch.width * self.scale) + 2 * padding_px)  # (w, h)
        self.base = self._draw_pitch()

    def to_px(self, xy_m: np.ndarray) -> np.ndarray:
        """Pitch metres -> radar pixels."""
        return (np.asarray(xy_m) * self.scale + self.pad).astype(int)

    def _draw_pitch(self) -> np.ndarray:
        p, s = self.pitch, self.scale
        w, h = self.size
        img = np.full((h, w, 3), GRASS, dtype=np.uint8)
        t = max(1, int(round(s * 0.25)))

        def rect(x0, y0, x1, y1):
            cv2.rectangle(img, tuple(self.to_px((x0, y0))), tuple(self.to_px((x1, y1))), LINE, t)

        L, W = p.length, p.width
        rect(0, 0, L, W)
        cv2.line(img, tuple(self.to_px((L / 2, 0))), tuple(self.to_px((L / 2, W))), LINE, t)
        centre = tuple(self.to_px((L / 2, W / 2)))
        cv2.circle(img, centre, int(p.centre_circle_radius * s), LINE, t, cv2.LINE_AA)
        cv2.circle(img, centre, max(2, t), LINE, -1)
        for side in (0, 1):  # left and right ends
            def x(d, side=side):
                return d if side == 0 else L - d
            pb0, pb1 = (W - p.penalty_box_width) / 2, (W + p.penalty_box_width) / 2
            gb0, gb1 = (W - p.goal_box_width) / 2, (W + p.goal_box_width) / 2
            rect(x(0), pb0, x(p.penalty_box_length), pb1)
            rect(x(0), gb0, x(p.goal_box_length), gb1)
            cv2.circle(img, tuple(self.to_px((x(p.penalty_spot_distance), W / 2))), max(2, t),
                       LINE, -1)
        return img

    def render(
        self,
        people_xy_m: np.ndarray,
        people_colors: list[tuple[int, int, int]],
        ball_xy_m: np.ndarray | None,
        highlight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Radar image for one frame. `highlight` = boolean mask of dots to ring (ball holder)."""
        img = self.base.copy()
        r = max(4, int(self.scale * 1.0))
        for i, (xy, color) in enumerate(zip(people_xy_m, people_colors, strict=True)):
            if np.isnan(xy).any():
                continue
            c = tuple(self.to_px(xy))
            cv2.circle(img, c, r, color, -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (0, 0, 0), 1, cv2.LINE_AA)
            if highlight is not None and highlight[i]:
                cv2.circle(img, c, r + 3, (68, 23, 255), 2, cv2.LINE_AA)
        if ball_xy_m is not None and not np.isnan(ball_xy_m).any():
            c = tuple(self.to_px(ball_xy_m))
            cv2.circle(img, c, max(3, r // 2), BALL, -1, cv2.LINE_AA)
            cv2.circle(img, c, max(3, r // 2), (0, 0, 0), 1, cv2.LINE_AA)
        return img


def overlay_bottom_right(
    frame: np.ndarray, radar_img: np.ndarray, margin: int = 20, opacity: float = 0.85
) -> np.ndarray:
    """Blend the radar into the bottom-right corner of the frame."""
    h, w = radar_img.shape[:2]
    y0, x0 = frame.shape[0] - h - margin, frame.shape[1] - w - margin
    roi = frame[y0 : y0 + h, x0 : x0 + w]
    frame[y0 : y0 + h, x0 : x0 + w] = cv2.addWeighted(radar_img, opacity, roi, 1 - opacity, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    return frame


def draw_projected_pitch(
    frame: np.ndarray, H_image_to_pitch: np.ndarray, pitch: PitchConfig, n: int = 60
) -> np.ndarray:
    """Debug view: draw the pitch lines back onto the video using the inverse homography.

    If the homography is right, the drawn lines sit exactly on the real white lines.
    """
    H_inv = np.linalg.inv(H_image_to_pitch)
    out = frame.copy()
    L, W = pitch.length, pitch.width
    segments = [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)),
                ((L / 2, 0), (L / 2, W))]
    for side_x, sign in ((0, 1), (L, -1)):
        for depth, half in ((pitch.penalty_box_length, pitch.penalty_box_width / 2),
                            (pitch.goal_box_length, pitch.goal_box_width / 2)):
            a, b = (side_x, W / 2 - half), (side_x + sign * depth, W / 2 - half)
            c, d = (side_x + sign * depth, W / 2 + half), (side_x, W / 2 + half)
            segments += [(a, b), (b, c), (c, d)]
    for p0, p1 in segments:
        t = np.linspace(0, 1, n)[:, None]
        pts = (1 - t) * np.array(p0) + t * np.array(p1)
        img_pts = transform_points(pts, H_inv).astype(int)
        cv2.polylines(out, [img_pts], False, (255, 0, 255), 2, cv2.LINE_AA)
    angles = np.linspace(0, 2 * np.pi, 120)
    circle = np.stack([L / 2 + pitch.centre_circle_radius * np.cos(angles),
                       W / 2 + pitch.centre_circle_radius * np.sin(angles)], axis=1)
    cv2.polylines(out, [transform_points(circle, H_inv).astype(int)], True, (255, 0, 255), 2,
                  cv2.LINE_AA)
    return out