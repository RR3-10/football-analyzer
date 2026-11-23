"""Clean-up applied to the raw tracks after tracking."""

from collections import Counter, defaultdict

import numpy as np

from football_analyzer.detection.tracker import FrameTracks


def stabilize_classes(tracks: list[FrameTracks]) -> dict[int, int]:
    """Give every track ID a single class by majority vote over all its frames.

    The detector sometimes labels the same goalkeeper 'player' in a few frames.
    Because the tracker gives him one ID, we can vote: if the ID was 'goalkeeper'
    in most frames, it is a goalkeeper in every frame. Modifies `tracks` in place.
    """
    votes: dict[int, Counter] = defaultdict(Counter)
    for frame in tracks:
        for tid, cls in zip(frame.people.tracker_id, frame.people.class_id, strict=True):
            votes[int(tid)][int(cls)] += 1

    track_class = {tid: counter.most_common(1)[0][0] for tid, counter in votes.items()}

    for frame in tracks:
        frame.people.class_id = np.array(
            [track_class[int(tid)] for tid in frame.people.tracker_id], dtype=int
        )
    return track_class