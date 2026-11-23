"""Real-world football pitch model (metres).

The 32 keypoints are in the SAME ORDER as the Roboflow 'football-field-detection'
dataset the pitch model was trained on. Coordinates: x along the length (0 = left
goal line), y along the width (0 = top touchline), in metres.

Line sizes follow the FIFA Laws of the Game. Only the overall length/width vary
between stadiums (100-110 m x 64-75 m), so they are configurable.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PitchConfig:
    length: float = 105.0
    width: float = 68.0
    penalty_box_length: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_length: float = 5.5
    goal_box_width: float = 18.32
    centre_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0

    @property
    def vertices(self) -> np.ndarray:
        """(32, 2) keypoint positions in metres, dataset order."""
        L, W = self.length, self.width
        pbl, pbw = self.penalty_box_length, self.penalty_box_width
        gbl, gbw = self.goal_box_length, self.goal_box_width
        r, ps = self.centre_circle_radius, self.penalty_spot_distance
        return np.array(
            [
                (0, 0), (0, (W - pbw) / 2), (0, (W - gbw) / 2),             # 1-3  left goal line
                (0, (W + gbw) / 2), (0, (W + pbw) / 2), (0, W),             # 4-6
                (gbl, (W - gbw) / 2), (gbl, (W + gbw) / 2),                 # 7-8  left goal box
                (ps, W / 2),                                                # 9    left penalty spot
                (pbl, (W - pbw) / 2), (pbl, (W - gbw) / 2),                 # 10-13 left penalty box
                (pbl, (W + gbw) / 2), (pbl, (W + pbw) / 2),
                (L / 2, 0), (L / 2, W / 2 - r),                             # 14-17 halfway line
                (L / 2, W / 2 + r), (L / 2, W),
                (L - pbl, (W - pbw) / 2), (L - pbl, (W - gbw) / 2),         # 18-21 right pen. box
                (L - pbl, (W + gbw) / 2), (L - pbl, (W + pbw) / 2),
                (L - ps, W / 2),                                            # 22   right spot
                (L - gbl, (W - gbw) / 2), (L - gbl, (W + gbw) / 2),         # 23-24 right goal box
                (L, 0), (L, (W - pbw) / 2), (L, (W - gbw) / 2),             # 25-30 right goal line
                (L, (W + gbw) / 2), (L, (W + pbw) / 2), (L, W),
                (L / 2 - r, W / 2), (L / 2 + r, W / 2),                     # 31-32 centre circle
            ],
            dtype=np.float32,
        )

    def line_points(self, step_m: float = 0.5) -> np.ndarray:
        """(M, 2) points sampled every `step_m` metres along all painted pitch lines."""
        L, W = self.length, self.width
        segments = [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)),
                    ((L / 2, 0), (L / 2, W))]
        for x0, sign in ((0.0, 1), (L, -1)):
            for depth, width in ((self.penalty_box_length, self.penalty_box_width),
                                 (self.goal_box_length, self.goal_box_width)):
                y0, y1 = (W - width) / 2, (W + width) / 2
                x1 = x0 + sign * depth
                segments += [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1))]
        points = []
        for (ax, ay), (bx, by) in segments:
            n = max(2, int(np.hypot(bx - ax, by - ay) / step_m))
            t = np.linspace(0, 1, n)[:, None]
            points.append((1 - t) * np.array([ax, ay]) + t * np.array([bx, by]))
        r = self.centre_circle_radius
        angles = np.linspace(0, 2 * np.pi, int(2 * np.pi * r / step_m), endpoint=False)
        points.append(np.stack([L / 2 + r * np.cos(angles), W / 2 + r * np.sin(angles)], axis=1))
        return np.concatenate(points).astype(np.float32)