from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np

from .track import Track


def _euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def associate_detections_to_tracks(
    detections_xyz: np.ndarray,
    tracks: Iterable[Track],
    gate_m: float = 100.0,
) -> Tuple[List[Tuple[int, str]], List[int], List[str]]:
    """
    Associate detections to tracks using nearest-neighbor Euclidean gating.

    Parameters
    ----------
    detections_xyz : np.ndarray
        Shape (N, 3)
    tracks : Iterable[Track]
        Active tracks
    gate_m : float
        Max distance allowed for association

    Returns
    -------
    matches : list of (det_idx, track_id)
    unmatched_det_idx : list of detection indices
    unmatched_track_ids : list of track ids
    """
    detections_xyz = np.asarray(detections_xyz, dtype=float)

    if detections_xyz.size == 0:
        return [], [], [tr.id for tr in tracks]

    track_list = [tr for tr in tracks if tr.state.name != "DROPPED"]
    if len(track_list) == 0:
        return [], list(range(len(detections_xyz))), []

    # build cost matrix
    cost = np.full((len(detections_xyz), len(track_list)), np.inf, dtype=float)
    for i, det in enumerate(detections_xyz):
        for j, tr in enumerate(track_list):
            pred = np.asarray(tr.kalman.position_xyz, dtype=float)
            cost[i, j] = _euclidean_dist(det, pred)

    matches: List[Tuple[int, str]] = []
    used_dets = set()
    used_tracks = set()

    # greedy nearest-neighbor assignment
    while True:
        idx = np.unravel_index(np.argmin(cost), cost.shape)
        i, j = int(idx[0]), int(idx[1])
        best = cost[i, j]

        if not np.isfinite(best) or best > gate_m:
            break

        if i in used_dets or j in used_tracks:
            cost[i, j] = np.inf
            continue

        matches.append((i, track_list[j].id))
        used_dets.add(i)
        used_tracks.add(j)

        cost[i, :] = np.inf
        cost[:, j] = np.inf

    unmatched_det_idx = [i for i in range(len(detections_xyz)) if i not in used_dets]
    unmatched_track_ids = [tr.id for k, tr in enumerate(track_list) if k not in used_tracks]

    return matches, unmatched_det_idx, unmatched_track_ids