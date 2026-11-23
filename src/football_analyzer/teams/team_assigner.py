"""Split players into two teams by jersey colour.

Step 1 - jersey colour of ONE player (per crop):
    Take the upper half of the box (shirt, not shorts/legs) and run K-means with k=2
    on its pixels: one cluster is the shirt, the other the background (grass).
    The cluster that owns most of the 4 crop corners is the background, because
    corners are almost never on the player. The other cluster's centre = jersey colour.

Step 2 - one colour per track:
    The median of a track's jersey colours over many frames, so one bad crop
    (occlusion, motion blur) cannot change a player's colour.

Step 3 - teams:
    K-means with k=3 on the track colours, in LAB colour space (where Euclidean
    distance matches how different colours look to humans, unlike RGB).
    The two largest clusters are the teams; the small third cluster catches odd kits
    (goalkeepers detected as 'player'), which are then placed by position.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

from football_analyzer.detection.tracker import FrameTracks
from football_analyzer.utils.video_io import iter_frames

REFEREE_TEAM = -1
ODD_KIT = -3  # kit matches neither team (usually a goalkeeper)
_KMEANS_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)


def bgr_to_lab(colors_bgr: np.ndarray) -> np.ndarray:
    pixels = np.asarray(colors_bgr, dtype=np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def lab_to_bgr(colors_lab: np.ndarray) -> np.ndarray:
    pixels = np.clip(colors_lab, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(pixels, cv2.COLOR_LAB2BGR).reshape(-1, 3)


def jersey_color(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Dominant shirt colour (LAB) of one player box, or None if the crop is unusable."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = xyxy.astype(int)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w_img), min(y2, h_img)
    crop = frame[y1 : y1 + (y2 - y1) // 2, x1:x2]  # top half = shirt
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return None

    pixels = crop.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)
    _, labels, centers = cv2.kmeans(
        pixels, 2, None, _KMEANS_CRITERIA, 1, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.reshape(crop.shape[:2])
    corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
    background = Counter(int(c) for c in corners).most_common(1)[0][0]
    return bgr_to_lab(centers[1 - background])[0]


@dataclass
class TeamResult:
    track_team: dict[int, int]          # track id -> 0 / 1 (REFEREE_TEAM for referees)
    team_colors_bgr: list[tuple[int, int, int]]  # display colour of team 0 and team 1


def cluster_teams(
    track_colors: dict[int, np.ndarray], max_relative_dist: float = 0.5, seed: int = 0
) -> tuple[dict, np.ndarray]:
    """Split tracks into 2 teams + 'odd kits' (goalkeepers, bad crops).

    Why 3 clusters and not 2? K-means with k=2 MUST produce 2 groups, and a single
    goalkeeper in a dark kit (often detected as 'player') is so far from everyone
    else that k=2 splits "goalkeeper vs. all players" instead of "team A vs. team B".
    With k=3 the outliers get their own cluster; the two LARGEST clusters are the teams.

    If there are no outliers, the third cluster is just a piece of a team, so every
    track of it is re-checked: a colour clearly close to one team centre (closer than
    `max_relative_dist` x the distance between the two team centres) joins that team;
    otherwise it really is an odd kit.

    Returns ({track id: 0 / 1 / ODD_KIT}, LAB centres of team 0 and team 1).
    """
    tids = list(track_colors)
    features = np.array([track_colors[t] for t in tids])
    n_clusters = min(3, len(tids))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(features)

    sizes = np.bincount(kmeans.labels_, minlength=n_clusters)
    team_a, team_b = np.argsort(sizes)[::-1][:2]  # the two biggest clusters
    centers = kmeans.cluster_centers_[[team_a, team_b]]
    team_gap = np.linalg.norm(centers[0] - centers[1])
    cluster_to_team = {int(team_a): 0, int(team_b): 1}

    assignment = {}
    for tid, label, color in zip(tids, kmeans.labels_, features, strict=True):
        if int(label) in cluster_to_team:
            assignment[tid] = cluster_to_team[int(label)]
            continue
        dists = np.linalg.norm(centers - color, axis=1)
        close = dists.min() < max_relative_dist * team_gap
        assignment[tid] = int(np.argmin(dists)) if close else ODD_KIT
    return assignment, centers


def assign_by_position(
    tracks: list[FrameTracks], track_team: dict[int, int], candidates: set[int]
) -> dict[int, int]:
    """Team of each candidate track = team whose players are closest on average.

    Used for goalkeepers: their kit differs from their team's, so colour can't work,
    but they always stand nearer to their own defenders than to the opponents.
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    for frame in tracks:
        people = frame.people
        feet = people.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        teams = np.array([track_team.get(int(t), -2) for t in people.tracker_id])
        for i, tid in enumerate(people.tracker_id):
            if int(tid) not in candidates:
                continue
            dists = [
                np.linalg.norm(feet[teams == team].mean(axis=0) - feet[i])
                if np.any(teams == team) else np.inf
                for team in (0, 1)
            ]
            if np.isfinite(min(dists)):
                votes[int(tid)][int(np.argmin(dists))] += 1
    return {tid: c.most_common(1)[0][0] for tid, c in votes.items()}


def assign_teams(
    video_path: str,
    tracks: list[FrameTracks],
    class_ids: dict[str, int],
    sample_stride: int = 5,
) -> TeamResult:
    """Jersey colour per track -> cluster into teams -> place goalkeepers by position."""
    player_cls, gk_cls, ref_cls = (class_ids[k] for k in ("player", "goalkeeper", "referee"))

    # 1) jersey colour samples (every `sample_stride` frames) for everyone labelled 'player'
    samples: dict[int, list[np.ndarray]] = defaultdict(list)
    frames = iter_frames(video_path, max_frames=len(tracks))
    for idx, frame in enumerate(frames):
        if idx % sample_stride:
            continue
        people = tracks[idx].people
        for xyxy, tid, cls in zip(people.xyxy, people.tracker_id, people.class_id, strict=True):
            if cls == player_cls and (color := jersey_color(frame, xyxy)) is not None:
                samples[int(tid)].append(color)
    if len(samples) < 2:
        raise RuntimeError("Not enough player crops to split teams.")

    # 2) one robust colour per track (median ignores a few bad crops), then cluster
    track_colors = {tid: np.median(colors, axis=0) for tid, colors in samples.items()}
    assignment, centers = cluster_teams(track_colors)
    track_team = {tid: team for tid, team in assignment.items() if team != ODD_KIT}

    # 3) goalkeepers + odd kits: decided by position, not colour
    goalkeepers = {
        int(t)
        for frame in tracks
        for t, c in zip(frame.people.tracker_id, frame.people.class_id, strict=True)
        if c == gk_cls
    }
    odd_kits = {tid for tid, team in assignment.items() if team == ODD_KIT}
    track_team.update(assign_by_position(tracks, track_team, goalkeepers | odd_kits))

    # 4) referees belong to no team
    for frame in tracks:
        for tid, cls in zip(frame.people.tracker_id, frame.people.class_id, strict=True):
            if cls == ref_cls:
                track_team[int(tid)] = REFEREE_TEAM

    team_colors = [tuple(int(v) for v in c) for c in lab_to_bgr(centers)]
    return TeamResult(track_team=track_team, team_colors_bgr=team_colors)