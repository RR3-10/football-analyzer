"""Broadcast-style overlays drawn directly with OpenCV."""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text_color(bg_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black text on light backgrounds, white text on dark ones."""
    b, g, r = bg_bgr
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_possession_bar(
    frame: np.ndarray,
    percentages: tuple[float, float],
    team_colors_bgr: list[tuple[int, int, int]],
) -> np.ndarray:
    """Semi-transparent panel (bottom-left) with a two-colour possession bar."""
    h, w = frame.shape[:2]
    s = w / 1920  # scale everything relative to a 1080p frame
    pad = int(20 * s)
    panel_w, panel_h = int(420 * s), int(90 * s)
    x0, y0 = pad, h - panel_h - pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    cv2.putText(frame, "BALL POSSESSION", (x0 + int(15 * s), y0 + int(28 * s)),
                FONT, 0.6 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)

    bar_x0, bar_x1 = x0 + int(15 * s), x0 + panel_w - int(15 * s)
    bar_y0, bar_y1 = y0 + int(42 * s), y0 + panel_h - int(14 * s)
    split = bar_x0 + int((bar_x1 - bar_x0) * percentages[0] / 100)
    cv2.rectangle(frame, (bar_x0, bar_y0), (split, bar_y1), team_colors_bgr[0], -1)
    cv2.rectangle(frame, (split, bar_y0), (bar_x1, bar_y1), team_colors_bgr[1], -1)

    text_y = bar_y1 - int(9 * s)
    for pct, x, color, align_right in (
        (percentages[0], bar_x0 + int(8 * s), team_colors_bgr[0], False),
        (percentages[1], bar_x1 - int(8 * s), team_colors_bgr[1], True),
    ):
        text = f"{pct:.0f}%"
        (tw, _), _ = cv2.getTextSize(text, FONT, 0.65 * s, max(1, int(2 * s)))
        cv2.putText(frame, text, (x - tw if align_right else x, text_y),
                    FONT, 0.65 * s, _text_color(color), max(1, int(2 * s)), cv2.LINE_AA)
    return frame