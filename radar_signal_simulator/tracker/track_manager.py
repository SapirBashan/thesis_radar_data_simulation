"""
TrackManager
============
Owns the list of Track objects and handles all data association.

Two public entry points
-----------------------
update_from_confirm(detections, beam, timestamp)
    Called after a CONFIRM beam fires.
    - Gates detections to beam footprint
    - Associates with existing TENTATIVE tracks via GNN / Mahalanobis
    - Unmatched detections → new TENTATIVE track
    - Matched detections   → Track.update() → may promote to CONFIRMED

update_from_track(detections, beam, timestamp)
    Called after a TRACK beam fires.
    - Tighter gate: beam footprint + predicted range window
    - Associates with CONFIRMED tracks whose predicted position falls in beam
    - Matched   → Track.update()
    - Unmatched → Track.miss()  (may → DELETED)

Gating (two-stage)
------------------
Stage 1 — Beam gate (cheap, spatial)
    confirm : |az_det - beam.az| ≤ beam_az_half
              |el_det - beam.el| ≤ beam_el_half
    track   : same az/el + |range_det - range_pred| ≤ range_gate_m

Stage 2 — Mahalanobis gate
    innovation y = z_meas - z_hat      (4-vector: range, az, el, vr)
    gate       y^T S^{-1} y ≤ gamma    (chi^2 threshold, 4 DOF)

GNN association
---------------
Cost matrix C[i, j] = mahalanobis²(track_i, det_j)  for gated pairs only,
                       large penalty otherwise.
Solved with scipy.optimize.linear_sum_assignment (Hungarian algorithm).
Assignments above gamma are rejected (unmatched).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from typing import List, Optional

from psp.detections import Detection
from tracker.beam import Beam

# reuse the Track class from the existing module
from tracker.track import Track


# ── constants ─────────────────────────────────────────────────────────

# chi^2 gate threshold: 4 DOF (range, az, el, vr), 95% confidence
_CHI2_GATE = chi2.ppf(0.95, df=4)

# large cost assigned to ungated pairs in the cost matrix
_LARGE_COST = 1e9


# ── helpers ───────────────────────────────────────────────────────────

def _det_polar(det: Detection):
    """Return (range_m, az_rad, el_rad, vr_mps) from a Detection."""
    return (
        det.range,
        np.deg2rad(det.azimuth),
        np.deg2rad(det.elevation),
        det.velocity,
    )


def _beam_gate_confirm(det: Detection, beam: Beam, sm) -> bool:
    """
    Stage-1 gate for a CONFIRM beam.
    Accept if detection az/el falls inside beam footprint.
    """
    az_half = sm.azimuth_width / 2.0
    el_half = sm.elevation_width / 2.0
    return (
        abs(det.azimuth - beam.az) <= az_half and
        abs(det.elevation - beam.el) <= el_half
    )


def _beam_gate_track(det: Detection, beam: Beam, sm,
                     pred_range_m: float, range_gate_m: float) -> bool:
    """
    Stage-1 gate for a TRACK beam.
    Accept if az/el inside beam footprint AND range near prediction.
    """
    if not _beam_gate_confirm(det, beam, sm):
        return False
    return abs(det.range - pred_range_m) <= range_gate_m


def _mahalanobis2(track: Track, det: Detection) -> float:
    """
    Compute squared Mahalanobis distance between track prediction and detection.
    Uses the EKF innovation covariance S = H P H^T + R.
    Returns _LARGE_COST if S is singular.
    """
    z_hat, S = track.get_predicted_measurement()   # (4,), (4,4)
    z = np.array(_det_polar(det))

    # wrap azimuth difference to [-pi, pi]
    diff = z - z_hat
    diff[1] = (diff[1] + np.pi) % (2 * np.pi) - np.pi

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return _LARGE_COST

    return float(diff @ S_inv @ diff)


# ── TrackManager ──────────────────────────────────────────────────────

class TrackManager:
    """
    Manages a list of Track objects.

    Parameters
    ----------
    radar_config      : RadarConfig  (passed through to Track)
    chi2_gate         : Mahalanobis² threshold (default: chi^2_4 at 95%)
    range_gate_m      : extra range gate for track beams (metres)
    confirm_threshold : hits needed to promote TENTATIVE → CONFIRMED
    delete_threshold  : misses before CONFIRMED → DELETED
    sigma_a           : process noise std (m/s²) passed to Track
    track_id_start    : starting integer for track IDs
    """

    def __init__(
        self,
        radar_config,
        chi2_gate:         float = _CHI2_GATE,
        range_gate_m:      float = 1000.0,
        confirm_threshold: int   = 3,
        delete_threshold:  int   = 5,
        sigma_a:           float = 5.0,
        track_id_start:    int   = 1,
    ):
        self.radar_config      = radar_config
        self.chi2_gate         = chi2_gate
        self.range_gate_m      = range_gate_m
        self.confirm_threshold = confirm_threshold
        self.delete_threshold  = delete_threshold
        self.sigma_a           = sigma_a

        self._tracks: List[Track] = []
        self._next_id: int        = track_id_start

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def update_from_confirm(
        self,
        detections: List[Detection],
        beam:       Beam,
        sm,                          # SubmodeTemplate for the confirm beam
        timestamp:  float,
    ) -> List[Track]:
        """
        Process detections from a CONFIRM beam.

        Returns the list of tracks that were updated or created.
        """
        # Stage-1: beam footprint gate
        gated_dets = [d for d in detections if _beam_gate_confirm(d, beam, sm)]

        if not gated_dets:
            return []

        # candidate tracks: TENTATIVE tracks whose prediction overlaps the beam
        candidate_tracks = [
            t for t in self._tracks
            if not t.is_deleted() and self._track_in_beam(t, beam, sm)
        ]

        matched_tracks, matched_dets, unmatched_dets = self._gnn_associate(
            candidate_tracks, gated_dets, timestamp
        )

        updated: List[Track] = []

        # update matched
        for tr, det in zip(matched_tracks, matched_dets):
            tr.update(det, timestamp)
            updated.append(tr)

        # unmatched detections → new TENTATIVE tracks
        for det in unmatched_dets:
            tr = self._new_track(det, sm, timestamp)
            updated.append(tr)

        self._purge_deleted()
        return updated

    def update_from_track(
        self,
        detections: List[Detection],
        beam:       Beam,
        sm,                          # SubmodeTemplate for the track beam
        timestamp:  float,
    ) -> List[Track]:
        """
        Process detections from a TRACK beam.

        Only CONFIRMED tracks whose predicted position falls inside the
        beam footprint are considered.  Unmatched confirmed tracks get a miss.

        Returns the list of tracks that were updated.
        """
        # candidate tracks: CONFIRMED and predicted into this beam
        candidate_tracks = [
            t for t in self._tracks
            if t.is_confirmed() and self._track_in_beam(t, beam, sm)
        ]

        if not candidate_tracks:
            return []

        # Stage-1: beam footprint + range gate per track
        # Build a per-track gated detection list
        gated_dets_per_track: List[List[Detection]] = []
        for tr in candidate_tracks:
            pred_r, _, _ = self._predicted_polar(tr)
            gated = [
                d for d in detections
                if _beam_gate_track(d, beam, sm, pred_r, self.range_gate_m)
            ]
            gated_dets_per_track.append(gated)

        # flatten unique gated detections for the global cost matrix
        all_gated: List[Detection] = list({
            id(d): d
            for per in gated_dets_per_track
            for d in per
        }.values())

        matched_tracks, matched_dets, _ = self._gnn_associate(
            candidate_tracks, all_gated, timestamp
        )

        updated: List[Track] = []
        matched_track_ids = {id(t) for t in matched_tracks}

        for tr, det in zip(matched_tracks, matched_dets):
            tr.update(det, timestamp)
            updated.append(tr)

        # unmatched candidates → miss
        for tr in candidate_tracks:
            if id(tr) not in matched_track_ids:
                tr.miss(timestamp)

        self._purge_deleted()
        return updated

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.is_confirmed()]

    def tentative_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.state == "TENTATIVE"]

    def all_active_tracks(self) -> List[Track]:
        return [t for t in self._tracks if not t.is_deleted()]

    def get_track(self, track_id: int) -> Optional[Track]:
        for t in self._tracks:
            if t.id == track_id:
                return t
        return None

    # ------------------------------------------------------------------ #
    # GNN association core
    # ------------------------------------------------------------------ #

    def _gnn_associate(
        self,
        tracks:     List[Track],
        detections: List[Detection],
        timestamp:  float,
    ):
        """
        Global Nearest Neighbour association using Mahalanobis distance.

        Returns
        -------
        matched_tracks : List[Track]
        matched_dets   : List[Detection]   (same order as matched_tracks)
        unmatched_dets : List[Detection]
        """
        if not tracks or not detections:
            return [], [], list(detections)

        n_t = len(tracks)
        n_d = len(detections)

        # predict all candidate tracks to current timestamp before gating
        for tr in tracks:
            tr.predict(timestamp)

        # build cost matrix (n_tracks × n_dets)
        C = np.full((n_t, n_d), _LARGE_COST)
        for i, tr in enumerate(tracks):
            for j, det in enumerate(detections):
                d2 = _mahalanobis2(tr, det)
                if d2 <= self.chi2_gate:
                    C[i, j] = d2

        # Hungarian algorithm on cost matrix
        row_ind, col_ind = linear_sum_assignment(C)

        matched_tracks: List[Track]     = []
        matched_dets:   List[Detection] = []
        matched_det_ids = set()

        for r, c in zip(row_ind, col_ind):
            if C[r, c] < _LARGE_COST:          # only accept gated pairs
                matched_tracks.append(tracks[r])
                matched_dets.append(detections[c])
                matched_det_ids.add(c)

        unmatched_dets = [
            detections[j] for j in range(n_d) if j not in matched_det_ids
        ]

        return matched_tracks, matched_dets, unmatched_dets

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _new_track(self, det: Detection, sm, timestamp: float) -> Track:
        tr = Track(
            detection=det,
            radar_config=self.radar_config,
            submode=sm,
            timestamp=timestamp,
            track_id=self._next_id,
            sigma_a=self.sigma_a,
            confirm_threshold=self.confirm_threshold,
            delete_threshold=self.delete_threshold,
        )
        self._next_id += 1
        self._tracks.append(tr)
        return tr

    def _purge_deleted(self) -> None:
        self._tracks = [t for t in self._tracks if not t.is_deleted()]

    def _predicted_polar(self, tr: Track):
        """
        Returns (range_m, az_deg, el_deg) of track's current state.
        """
        pos = tr.get_position()
        x, y, z = pos
        r  = float(np.linalg.norm(pos)) + 1e-12
        az = float(np.degrees(np.arctan2(y, x)))
        el = float(np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2))))
        return r, az, el

    def _track_in_beam(self, tr: Track, beam: Beam, sm) -> bool:
        """
        True if track's predicted position falls inside the beam footprint.
        """
        _, az, el = self._predicted_polar(tr)
        az_half = sm.azimuth_width / 2.0
        el_half = sm.elevation_width / 2.0
        return (
            abs(az - beam.az) <= az_half and
            abs(el - beam.el) <= el_half
        )