from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .association import associate_detections_to_tracks
from .beam import Beam
from .states import BeamPurpose, RadarState, TrackState
from .track import Track


@dataclass(order=True)
class _QItem:
    priority: int
    beam: Beam = field(compare=False)


class RadarManager:
    """
    Radar scheduler / beam orchestrator.
    """

    SUBMODE_COARSE = "wide_search"
    SUBMODE_FINE = "point_search"
    SUBMODE_CONFIRM = "confirm"
    SUBMODE_TRACK = "track"
    SUBMODE_RECOVERY = "recovery"

    PRI_CONFIRM = 0
    PRI_TRACK = 1
    PRI_FINE = 2
    PRI_COARSE = 3
    PRI_RECOVERY = 1

    def __init__(self, radar_cfg, submode_lib):
        self.radar = radar_cfg
        self.submode_lib = submode_lib

        self.state: RadarState = RadarState.IDLE
        self.sim_time_ms: float = 0.0

        self.tracks: Dict[str, Track] = {}
        self._queue: List[_QItem] = []

        self._coarse_grid: List[Tuple[float, float]] = []
        self._coarse_cursor: int = 0

        self.current_beam: Optional[Beam] = None

    # ------------------------------------------------------------------
    # coarse scan
    # ------------------------------------------------------------------

    def set_coarse_grid(self, grid: List[Tuple[float, float]]) -> None:
        self._coarse_grid = list(grid)
        self._coarse_cursor = 0

    def enqueue_coarse_scan(self) -> None:
        if not self._coarse_grid:
            return

        for idx in range(self._coarse_cursor, len(self._coarse_grid)):
            az, el = self._coarse_grid[idx]
            self._push(self.PRI_COARSE, Beam(
                submode_name=self.SUBMODE_COARSE,
                az=az,
                el=el,
                purpose=BeamPurpose.COARSE,
                grid_level="coarse",
            ))

        self._coarse_cursor = len(self._coarse_grid)

    def mark_coarse_progress(self, n_beams_done: int = 1) -> None:
        self._coarse_cursor = min(self._coarse_cursor + n_beams_done, len(self._coarse_grid))

    # ------------------------------------------------------------------
    # track management
    # ------------------------------------------------------------------

    def add_track(self, track: Track) -> None:
        self.tracks[track.id] = track

    def drop_track(self, track_id: str) -> None:
        tr = self.tracks.get(track_id)
        if tr is None:
            return
        tr.state = TrackState.DROPPED
        tr.pending_beam = None

    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.state != TrackState.DROPPED]

    # ------------------------------------------------------------------
    # beam scheduling helpers
    # ------------------------------------------------------------------

    def _enqueue_fine(self, tr: Track) -> None:
        _, az, el = tr.kalman.polar_position
        az = float(np.clip(az, *self.radar.azimuth_range))
        el = float(np.clip(el, *self.radar.elevation_range))
        tr.pending_beam = (az, el)

        self._push(self.PRI_FINE, Beam(
            submode_name=self.SUBMODE_FINE,
            az=az,
            el=el,
            purpose=BeamPurpose.FINE,
            track_id=tr.id,
            grid_level="fine",
        ))

    def _enqueue_confirm(self, tr: Track) -> None:
        _, az, el = tr.kalman.polar_position
        az = float(np.clip(az, *self.radar.azimuth_range))
        el = float(np.clip(el, *self.radar.elevation_range))
        tr.pending_beam = (az, el)

        self._push(self.PRI_CONFIRM, Beam(
            submode_name=self.SUBMODE_CONFIRM,
            az=az,
            el=el,
            purpose=BeamPurpose.CONFIRM,
            track_id=tr.id,
            grid_level="confirm",
        ))

    def _enqueue_recovery(self, tr: Track, az: float, el: float) -> None:
        az = float(np.clip(az, *self.radar.azimuth_range))
        el = float(np.clip(el, *self.radar.elevation_range))
        tr.pending_beam = (az, el)

        self._push(self.PRI_RECOVERY, Beam(
            submode_name=self.SUBMODE_CONFIRM,
            az=az,
            el=el,
            purpose=BeamPurpose.RECOVERY,
            track_id=tr.id,
            grid_level="recovery",
        ))

    def _schedule_track_update(self, tr: Track) -> None:
        speed = max(tr.speed, 1.0)
        dt_ms = float(np.clip(1000.0 / speed, 50.0, 1000.0))

        sigma = tr.position_uncertainty
        sigma_ref = 200.0
        dt_ms *= float(np.clip(sigma / sigma_ref, 0.5, 1.0))

        tr.set_next_revisit(self.sim_time_ms, dt_ms)

    def _inject_due_track_beams(self) -> None:
        for tr in self.tracks.values():
            if tr.state != TrackState.TRACKED:
                continue
            if tr.pending_beam is not None:
                continue
            if self.sim_time_ms >= tr.next_revisit_time_ms:
                self._enqueue_fine(tr)

    def _push(self, priority: int, beam: Beam) -> None:
        heapq.heappush(self._queue, _QItem(priority=priority, beam=beam))

    # ------------------------------------------------------------------
    # beam retrieval
    # ------------------------------------------------------------------

    def pop_next_beam(self) -> Optional[Beam]:
        self._inject_due_track_beams()

        if not self._queue:
            self.state = RadarState.IDLE
            return None

        item = heapq.heappop(self._queue)
        self.current_beam = item.beam
        self.state = {
            BeamPurpose.COARSE: RadarState.COARSE_SCAN,
            BeamPurpose.FINE: RadarState.FINE_SCAN,
            BeamPurpose.CONFIRM: RadarState.CONFIRM_SCAN,
            BeamPurpose.TRACK: RadarState.TRACKING,
            BeamPurpose.RECOVERY: RadarState.RECOVERY,
        }.get(item.beam.purpose, RadarState.IDLE)

        return item.beam

    # ------------------------------------------------------------------
    # detection update
    # ------------------------------------------------------------------

    def update_with_detections(self, detections_xyz: np.ndarray) -> None:
        """
        Update tracks from detections in Cartesian coordinates.
        """
        detections_xyz = np.asarray(detections_xyz, dtype=float).reshape(-1, 3) if detections_xyz.size else np.empty((0, 3))

        active = self.active_tracks()
        matches, unmatched_det_idx, unmatched_track_ids = associate_detections_to_tracks(
            detections_xyz, active, gate_m=100.0
        )

        matched_track_ids = set()

        # matched tracks => hit
        for det_idx, track_id in matches:
            tr = self.tracks[track_id]
            tr.register_hit(detections_xyz[det_idx], self.sim_time_ms)
            matched_track_ids.add(track_id)

            if tr.state == TrackState.CANDIDATE:
                tr.state = TrackState.CONFIRMING
                self._enqueue_fine(tr)
            elif tr.state == TrackState.CONFIRMING:
                tr.state = TrackState.TRACKED
                self._schedule_track_update(tr)
            elif tr.state == TrackState.LOST:
                tr.state = TrackState.TRACKED
                self._schedule_track_update(tr)
            elif tr.state == TrackState.TRACKED:
                self._schedule_track_update(tr)

            tr.pending_beam = None

        # unmatched tracks => miss
        for track_id in unmatched_track_ids:
            tr = self.tracks[track_id]
            tr.register_miss()

            if tr.state == TrackState.LOST and tr.miss_count < tr.max_misses:
                # local recovery search around predicted location
                _, az, el = tr.kalman.polar_position
                az = float(np.clip(az, *self.radar.azimuth_range))
                el = float(np.clip(el, *self.radar.elevation_range))
                self._enqueue_recovery(tr, az, el)

            if tr.state == TrackState.DROPPED:
                tr.pending_beam = None

        # unmatched detections => spawn new candidate tracks
        for det_idx in unmatched_det_idx:
            xyz = detections_xyz[det_idx]
            new_track = Track.from_detection(
                xyz=xyz,
                sim_time_ms=self.sim_time_ms,
            )
            self.add_track(new_track)
            self._enqueue_fine(new_track)

    # ------------------------------------------------------------------
    # timing
    # ------------------------------------------------------------------

    def step_time(self, dt_ms: float) -> None:
        self.sim_time_ms += float(dt_ms)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def queue_summary(self) -> dict[str, int]:
        counts = {"track": 0, "confirm": 0, "fine": 0, "coarse": 0, "recovery": 0}
        for item in self._queue:
            counts[item.beam.purpose.name.lower()] += 1
        return counts