"""Ball possession.

Rule: in each frame, the player whose feet are closest to the ball controls it,
if the distance is small relative to his size on screen (players far from the
camera look smaller, so a fixed pixel threshold would be unfair).
When nobody is close (ball in the air during a pass), the last team keeps
possession - the same convention TV broadcasters use.
"""

from dataclasses import dataclass

import numpy as np
import supervision as sv

from football_analyzer.detection.tracker import FrameTracks

NO_ONE = -1


@dataclass
class PossessionResult:
    holder_id: np.ndarray  # (N,) track id of the player on the ball, NO_ONE if nobody
    team: np.ndarray       # (N,) team in possession (carried forward), NO_ONE before first touch

    def percentages(self) -> dict[int, float]:
        """Share of controlled frames per team, e.g. {0: 54.2, 1: 45.8}."""
        controlled = self.team[self.team != NO_ONE]
        if len(controlled) == 0:
            return {0: 0.0, 1: 0.0}
        return {t: 100.0 * float(np.mean(controlled == t)) for t in (0, 1)}

    def running_percentages(self) -> np.ndarray:
        """(N, 2) possession % of team 0 and team 1 up to each frame (for the live bar)."""
        counts = np.stack([np.cumsum(self.team == 0), np.cumsum(self.team == 1)], axis=1)
        total = np.maximum(counts.sum(axis=1, keepdims=True), 1)
        return 100.0 * counts / total


def find_holder(
    people: sv.Detections,
    ball_xy: np.ndarray,
    track_team: dict[int, int],
    max_dist_ratio: float,
) -> int:
    """Track id of the player controlling the ball in one frame, or NO_ONE."""
    if np.isnan(ball_xy[0]) or len(people) == 0:
        return NO_ONE
    feet = people.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    heights = people.xyxy[:, 3] - people.xyxy[:, 1]
    dists = np.linalg.norm(feet - ball_xy, axis=1)

    best, best_dist = NO_ONE, np.inf
    for tid, dist, height in zip(people.tracker_id, dists, heights, strict=True):
        if track_team.get(int(tid), NO_ONE) == NO_ONE:  # referees / unknown never hold the ball
            continue
        if dist <= max_dist_ratio * height and dist < best_dist:
            best, best_dist = int(tid), dist
    return best


def compute_possession(
    tracks: list[FrameTracks],
    ball_xy: np.ndarray,
    track_team: dict[int, int],
    max_dist_ratio: float = 0.6,
) -> PossessionResult:
    holder = np.full(len(tracks), NO_ONE)
    team = np.full(len(tracks), NO_ONE)
    current_team = NO_ONE
    for i, frame in enumerate(tracks):
        holder[i] = find_holder(frame.people, ball_xy[i], track_team, max_dist_ratio)
        if holder[i] != NO_ONE:
            current_team = track_team[holder[i]]
        team[i] = current_team
    return PossessionResult(holder_id=holder, team=team)