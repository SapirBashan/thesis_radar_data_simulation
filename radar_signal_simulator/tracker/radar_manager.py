"""
RadarManager
============
Single entry point for one radar simulation step.

Design principles
-----------------
- next_step(targets) advances the simulation by one beam dwell.
  It returns NOTHING. All state is stored internally and exposed
  via read-only properties for a visualisation layer to consume.

- Internal simulation time (sim_time_ms) advances by sm.time each step.
  This timestamp is passed to TrackManager so the EKF dt is correct.

- The manager owns a TrackManager which handles all data association,
  Mahalanobis gating and GNN assignment.

- Beam scheduling is priority-queue based:
      0  track beams   (highest — adaptive interval)
      1  confirm beams
      2  fine beams
      3  coarse beams  (lowest — fills gaps, refills when exhausted)

- The last step's data (beam, signal, detections, tracks) is stored
  on the manager so a visualisation object can pull it at any time.

Visualization API (read-only properties)
-----------------------------------------
    mgr.current_beam        → Beam fired this step
    mgr.current_signal      → raw IQ array (n_elements, n_rg, n_pri)
    mgr.current_detections  → List[Detection] after SNR filter
    mgr.current_submode     → SubmodeTemplate for current beam
    mgr.sim_time_ms         → current simulation time (ms)
    mgr.track_manager       → TrackManager (full track list access)
    mgr.queue_summary()     → dict with beam counts per purpose
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config.radar_config import RadarConfig
from config.submode_config import SubmodeLibrary, SubmodeTemplate
from config.operating_submode import OperatingSubmode
from config.target import Target
from psp import psp_process, Detection
from radar_signal.generator import RadarSimulator

from tracker.beam import Beam
from tracker.track_manager import TrackManager


# ======================================================================
# Priority queue item
# ======================================================================

_COUNTER = itertools.count()


@dataclass(order=True)
class _QItem:
    priority: int
    seq:      int  = field(default_factory=lambda: next(_COUNTER))
    beam:     Beam = field(compare=False, default=None)


# ======================================================================
# Grid helpers
# ======================================================================

def _bounded_centers(lo: float, hi: float, width: float, step: float) -> np.ndarray:
    half  = width / 2.0
    start = lo + half
    stop  = hi - half
    if stop < start:
        return np.array([])
    return np.arange(start, stop + 1e-9, step)


def _build_coarse_grid(
    radar: RadarConfig,
    sm:    SubmodeTemplate,
) -> list[tuple[float, float]]:
    az_centers = _bounded_centers(*radar.azimuth_range,  sm.azimuth_width,   sm.azimuth_width)
    el_centers  = _bounded_centers(*radar.elevation_range, sm.elevation_width, sm.elevation_width)
    return [(az, el) for el in el_centers for az in az_centers]


def _build_fine_grid(
    center_az: float,
    center_el: float,
    coarse_sm: SubmodeTemplate,
) -> list[tuple[float, float]]:
    """Subdivide the coarse beam footprint into a 4×4 fine grid."""
    az_step = coarse_sm.azimuth_width  / 4.0
    el_step = coarse_sm.elevation_width / 4.0
    az0     = center_az - coarse_sm.azimuth_width  / 2.0 + az_step / 2.0
    el0     = center_el - coarse_sm.elevation_width / 2.0 + el_step / 2.0
    return [
        (az0 + i * az_step, el0 + j * el_step)
        for j in range(4)
        for i in range(4)
    ]


def _build_confirm_grid(
    center_az: float,
    center_el: float,
    fine_sm:   SubmodeTemplate,
) -> list[tuple[float, float]]:
    """Subdivide the fine beam footprint into a 4×4 confirm grid."""
    az_step = fine_sm.azimuth_width  / 4.0
    el_step = fine_sm.elevation_width / 4.0
    az0     = center_az - fine_sm.azimuth_width  / 2.0 + az_step / 2.0
    el0     = center_el - fine_sm.elevation_width / 2.0 + el_step / 2.0
    return [
        (az0 + i * az_step, el0 + j * el_step)
        for j in range(4)
        for i in range(4)
    ]


# ======================================================================
# RadarManager
# ======================================================================

class RadarManager:
    """
    Parameters
    ----------
    radar            : RadarConfig
    submode_lib      : SubmodeLibrary  — must contain coarse / fine / confirm / track
    snr_min_db       : minimum SNR for a detection to be accepted
    track_k          : adaptive track interval scale constant (m)
    dt_min_ms        : minimum track update interval (ms)
    dt_max_ms        : maximum track update interval (ms)
    track_manager_kw : kwargs forwarded to TrackManager (chi2_gate, range_gate_m, …)
    """

    # ── submode keys (must match submodes.json) ────────────────────────
    SBM_COARSE  = "coarse_search"
    SBM_FINE    = "fine_search"
    SBM_CONFIRM = "confirm"
    SBM_TRACK   = "track"

    # ── beam queue priorities ──────────────────────────────────────────
    PRI_TRACK   = 0
    PRI_CONFIRM = 1
    PRI_FINE    = 2
    PRI_COARSE  = 3

    def __init__(
        self,
        radar:       RadarConfig,
        submode_lib: SubmodeLibrary,
        # snr_min_db:  float = 5.0,
        track_k:     float = 5000.0,
        dt_min_ms:   float = 20.0,
        dt_max_ms:   float = 2000.0,
        **track_manager_kw,
    ):
        self.radar       = radar
        self.submode_lib = submode_lib
        # self.snr_min_db  = snr_min_db
        self.track_k     = track_k
        self.dt_min_ms   = dt_min_ms
        self.dt_max_ms   = dt_max_ms

        # ── simulation time ────────────────────────────────────────────
        self.sim_time_ms: float = 0.0

        # ── signal generator ───────────────────────────────────────────
        self._sim = RadarSimulator(radar, submode_lib)

        # ── track manager ─────────────────────────────────────────────
        self.track_manager = TrackManager(radar, **track_manager_kw)

        # ── beam priority queue ───────────────────────────────────────
        self._queue: list[_QItem] = []
        self._enqueue_coarse_scan()

        # ── per-step state (read by visualisation) ────────────────────
        self._current_beam:       Optional[Beam]            = None
        self._current_signal:     Optional[np.ndarray]      = None
        self._current_detections: List[Detection]           = []
        self._current_submode:    Optional[SubmodeTemplate] = None

        # ── track scheduling: track_id → next_update_time_ms ──────────
        self._track_next_update: dict[int, float] = {}

    # ==================================================================
    # Public step API
    # ==================================================================

    def next_step(self, targets: List[Target]) -> None:
        """
        Advance the simulation by one beam dwell.

        1. Pick the next beam from the priority queue.
        2. Advance sim_time_ms by sm.time.
        3. Generate signal + add no noise (caller may add noise outside,
           but here we let the generator handle it via noise_sigma).
        4. Run PSP pipeline.
        5. Filter by SNR.
        6. Route detections to TrackManager.
        7. Schedule follow-up beams (fine / confirm / track).
        8. Store everything for the visualisation layer.
        """
        self._inject_due_track_beams()

        if not self._queue:
            self._enqueue_coarse_scan()

        beam: Beam             = heapq.heappop(self._queue).beam
        sm:   SubmodeTemplate  = self.submode_lib.submodes[beam.submode_name]
        op:   OperatingSubmode = OperatingSubmode.from_values(
            beam.submode_name, beam.az, beam.el
        )

        # ── advance time ──────────────────────────────────────────────
        self.sim_time_ms += sm.time

        # ── generate signal ───────────────────────────────────────────
        signal = self._sim.generate(targets, op, noise_sigma=0.0)

        # ── PSP pipeline ──────────────────────────────────────────────
        cfar_params = sm.cfar
        detections  = psp_process(signal, self.radar, self.submode_lib, op, cfar_params)

        # ── SNR filter ────────────────────────────────────────────────
        # detections = [d for d in detections if d.snr >= self.snr_min_db]

        # ── route to TrackManager ─────────────────────────────────────
        if beam.submode_name == self.SBM_CONFIRM:
            self.track_manager.update_from_confirm(
                detections, beam, sm, self.sim_time_ms
            )
            # schedule confirm grid follow-up only if detections found
            if detections:
                self._on_confirm_detections(beam, detections)

        elif beam.submode_name == self.SBM_TRACK:
            self.track_manager.update_from_track(
                detections, beam, sm, self.sim_time_ms
            )
            # reschedule next track update for this track
            if beam.track_id is not None:
                self._reschedule_track(beam.track_id)

        elif beam.submode_name == self.SBM_FINE:
            # fine detections → queue confirm beams
            if detections:
                self._on_fine_detections(beam, detections)

        elif beam.submode_name == self.SBM_COARSE:
            # coarse detections → queue fine beams
            if detections:
                self._on_coarse_detections(beam)

        # ── store for visualisation ───────────────────────────────────
        self._current_beam       = beam
        self._current_signal     = signal
        self._current_detections = detections
        self._current_submode    = sm

    # ==================================================================
    # Visualisation properties (read-only)
    # ==================================================================

    @property
    def current_beam(self) -> Optional[Beam]:
        return self._current_beam

    @property
    def current_signal(self) -> Optional[np.ndarray]:
        return self._current_signal

    @property
    def current_detections(self) -> List[Detection]:
        return self._current_detections

    @property
    def current_submode(self) -> Optional[SubmodeTemplate]:
        return self._current_submode

    def queue_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {
            "coarse": 0, "fine": 0, "confirm": 0, "track": 0
        }
        for item in self._queue:
            p = item.beam.purpose
            counts[p] = counts.get(p, 0) + 1
        return counts

    # ==================================================================
    # Detection routing → beam scheduling
    # ==================================================================

    def _on_coarse_detections(self, beam: Beam) -> None:
        """Coarse hit → enqueue 4×4 fine grid over that beam footprint."""
        coarse_sm = self.submode_lib.submodes[self.SBM_COARSE]
        for az, el in _build_fine_grid(beam.az, beam.el, coarse_sm):
            az = float(np.clip(az, *self.radar.azimuth_range))
            el = float(np.clip(el, *self.radar.elevation_range))
            self._push(self.PRI_FINE, Beam(self.SBM_FINE, az, el))

    def _on_fine_detections(self, beam: Beam, detections: List[Detection]) -> None:
        """Fine hit → enqueue confirm grid centred on each detection cluster."""
        fine_sm = self.submode_lib.submodes[self.SBM_FINE]
        # use the beam centre (detections are within fine beam footprint)
        for az, el in _build_confirm_grid(beam.az, beam.el, fine_sm):
            az = float(np.clip(az, *self.radar.azimuth_range))
            el = float(np.clip(el, *self.radar.elevation_range))
            self._push(self.PRI_CONFIRM, Beam(self.SBM_CONFIRM, az, el))

    def _on_confirm_detections(
        self,
        beam:       Beam,
        detections: List[Detection],
    ) -> None:
        """
        Confirm hit → schedule first track beam for every newly confirmed track.
        TrackManager.update_from_confirm already created/updated the tracks;
        here we just schedule their first track-beam update.
        """
        for tr in self.track_manager.confirmed_tracks():
            if tr.id not in self._track_next_update:
                self._reschedule_track(tr.id)

    # ==================================================================
    # Adaptive track beam scheduling
    # ==================================================================

    def _reschedule_track(self, track_id: int) -> None:
        """
        Compute adaptive next-update time for a confirmed track.

            dt_ms = clip(track_k / speed, dt_min, dt_max)

        A faster target gets updated more frequently.
        """
        tr = self.track_manager.get_track(track_id)
        if tr is None or tr.is_deleted():
            self._track_next_update.pop(track_id, None)
            return

        speed = float(np.linalg.norm(tr.get_velocity()))
        speed = max(speed, 1.0)   # avoid division by zero

        dt_ms = float(np.clip(self.track_k / speed, self.dt_min_ms, self.dt_max_ms))
        self._track_next_update[track_id] = self.sim_time_ms + dt_ms

    def _inject_due_track_beams(self) -> None:
        """
        For every confirmed track whose next-update time has arrived,
        push a TRACK beam pointed at the predicted position.
        Skips tracks that already have a track beam in the queue.
        """
        # build set of track_ids already queued
        queued_ids = {
            item.beam.track_id
            for item in self._queue
            if item.beam.purpose == "track" and item.beam.track_id is not None
        }

        for tr in self.track_manager.confirmed_tracks():
            if tr.id in queued_ids:
                continue

            due = self._track_next_update.get(tr.id, 0.0)
            if self.sim_time_ms < due:
                continue

            # predict position to now and point beam there
            pos   = tr.get_position()
            r     = float(np.linalg.norm(pos)) + 1e-12
            az    = float(np.degrees(np.arctan2(pos[1], pos[0])))
            el    = float(np.degrees(np.arctan2(pos[2], np.sqrt(pos[0]**2 + pos[1]**2))))

            az = float(np.clip(az, *self.radar.azimuth_range))
            el = float(np.clip(el, *self.radar.elevation_range))

            self._push(
                self.PRI_TRACK,
                Beam(self.SBM_TRACK, az, el),
            )

    # ==================================================================
    # Queue helpers
    # ==================================================================

    def _enqueue_coarse_scan(self) -> None:
        sm = self.submode_lib.submodes[self.SBM_COARSE]
        for az, el in _build_coarse_grid(self.radar, sm):
            self._push(self.PRI_COARSE, Beam(self.SBM_COARSE, az, el))

    def _push(self, priority: int, beam: Beam) -> None:
        heapq.heappush(self._queue, _QItem(priority=priority, beam=beam))