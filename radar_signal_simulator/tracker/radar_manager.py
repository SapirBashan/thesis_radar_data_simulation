"""
RadarManager
============
Manages the full radar scan/track loop as discrete steps.

Usage
-----
    mgr = RadarManager(radar_cfg, submode_lib, cfar_params)

    while True:
        beam = mgr.next_beam()           # what to transmit next
        sig  = sim.generate(targets, beam.op_submode)
        dets = psp_process(sig, ...)
        frame = mgr.process(dets, beam)  # feed detections back
        # frame contains everything needed for visualisation

Beam priority queue
-------------------
    0  track-update beams   (highest priority)
    1  confirm beams
    2  fine-search beams
    3  coarse-search beams  (lowest — fills gaps)

Adaptive track interval
-----------------------
    dt_next = clip(k / speed,  dt_min,  dt_max)
    where k is a tunable scale constant (default 5000 m).
    When uncertainty is high the interval is shortened further:
    dt_next = dt_next * clip(sigma / sigma_ref, 0.5, 1.0)
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config.radar_config import RadarConfig
from config.submode_config import SubmodeLibrary
from config.operating_submode import OperatingSubmode
from psp import psp_process, CFARParams, Detection

from .track import Track, TrackState
from .kalman import KalmanTracker

C0 = 299_792_458.0


# ======================================================================
# helpers
# ======================================================================

def _det_to_xyz(det: Detection) -> np.ndarray:
    az = math.radians(det.azimuth)
    el = math.radians(det.elevation)
    r  = det.range
    x  = r * math.cos(el) * math.cos(az)
    y  = r * math.cos(el) * math.sin(az)
    z  = r * math.sin(el)
    return np.array([x, y, z], dtype=float)


def _xyz_to_polar(xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = xyz
    r  = float(np.sqrt(x**2 + y**2 + z**2)) + 1e-12
    az = float(math.degrees(math.atan2(y, x)))
    el = float(math.degrees(math.atan2(z, math.sqrt(x**2 + y**2))))
    return r, az, el


def _match_detection_to_track(
    xyz: np.ndarray,
    tracks: dict[str, Track],
    gate_m: float = 500.0,
) -> Optional[Track]:
    """Nearest-neighbour gating in XYZ space."""
    best_track = None
    best_dist  = gate_m
    for tr in tracks.values():
        if tr.state == TrackState.DROPPED:
            continue
        d = float(np.linalg.norm(xyz - tr.kalman.position_xyz))
        if d < best_dist:
            best_dist  = d
            best_track = tr
    return best_track


# ======================================================================
# Beam descriptor
# ======================================================================

@dataclass
class Beam:
    """Everything needed to fire one look and to interpret the result."""
    submode_name: str          # key into SubmodeLibrary
    az:           float        # beam centre azimuth  (degrees)
    el:           float        # beam centre elevation (degrees)
    purpose:      str          # 'coarse' | 'fine' | 'confirm' | 'track'
    track_id:     Optional[str] = None   # set for fine/confirm/track beams

    @property
    def op_submode(self) -> OperatingSubmode:
        return OperatingSubmode.from_values(self.submode_name, self.az, self.el)


# ======================================================================
# Step result — returned to the caller every step
# ======================================================================

@dataclass
class StepResult:
    """Everything a visualisation layer needs for one radar step."""
    beam:           Beam
    detections:     list[Detection]
    tracks:         dict[str, Track]   # full track table (snapshot)
    sim_time_ms:    float
    new_track_ids:  list[str]          # tracks created this step
    dropped_ids:    list[str]          # tracks dropped this step


# ======================================================================
# Priority queue item
# ======================================================================

_COUNTER = itertools.count()   # tie-break for heapq

@dataclass(order=True)
class _QItem:
    priority:  int
    seq:       int = field(default_factory=lambda: next(_COUNTER))
    beam:      Beam = field(compare=False, default=None)


# ======================================================================
# Coarse scan grid builder
# ======================================================================

def _bounded_centers(lo, hi, width, step):
    half  = width / 2.0
    start = lo + half
    stop  = hi - half
    if stop < start:
        return np.array([])
    return np.arange(start, stop + 1e-9, step)


# ======================================================================
# RadarManager
# ======================================================================

class RadarManager:
    """
    Step-based radar beam scheduler + track manager.

    Parameters
    ----------
    radar        : RadarConfig
    submode_lib  : SubmodeLibrary  (must contain keys listed in SUBMODE_*)
    cfar_params  : CFARParams
    max_misses   : misses before a TRACKED track retries via fine (default 3)
    gate_m       : association gate radius in metres (default 500 m)
    track_k      : scale constant for adaptive interval (metres, default 5000)
    dt_min_ms / dt_max_ms : clamp on adaptive track interval
    snr_min_db   : minimum SNR to accept a detection (default 5 dB)
    """

    # submode name constants — must exist in submodes.json
    SUBMODE_COARSE  = "coarse_search"
    SUBMODE_FINE    = "fine_search"
    SUBMODE_CONFIRM = "confirm"
    SUBMODE_TRACK   = "track"

    # beam priority levels
    PRI_TRACK   = 0
    PRI_CONFIRM = 1
    PRI_FINE    = 2
    PRI_COARSE  = 3

    def __init__(
        self,
        radar:       RadarConfig,
        submode_lib: SubmodeLibrary,
        max_misses:  int   = 15,
        gate_m:      float = 500.0,
        track_k:     float = 5000.0,
        dt_min_ms:   float = 20.0,
        dt_max_ms:   float = 2000.0,
        snr_min_db:  float = 1.0,
    ):
        self.radar       = radar
        self.submode_lib = submode_lib
        self.max_misses  = max_misses
        self.gate_m      = gate_m
        self.track_k     = track_k
        self.dt_min_ms   = dt_min_ms
        self.dt_max_ms   = dt_max_ms
        self.snr_min_db  = snr_min_db

        self.tracks:      dict[str, Track] = {}
        self.sim_time_ms: float            = 0.0

        # priority queue of upcoming beams
        self._queue: list[_QItem] = []

        # fill with a complete coarse scan
        self._enqueue_coarse_scan()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def next_beam(self) -> Beam:
        """
        Returns the next beam to fire.
        If the queue is empty (shouldn't happen normally) refills coarse.
        Also injects scheduled track-update beams before returning.
        """
        self._inject_due_track_beams()

        # if only coarse beams are left and the queue is empty → refill
        if not self._queue:
            self._enqueue_coarse_scan()

        item = heapq.heappop(self._queue)
        return item.beam

    def process(
        self,
        detections: list[Detection],
        beam:       Beam,
        beam_time_ms: float,         # duration of this beam in ms (sm.time)
    ) -> StepResult:
        """
        Feed PSP detections back into the manager.
        Updates track state machine and schedules follow-up beams.
        Returns a StepResult for visualisation.
        """
        self.sim_time_ms += beam_time_ms

        # filter by SNR
        # detections = [d for d in detections if d.snr >= self.snr_min_db]


        new_ids: list[str] = []
        dropped_ids: list[str] = []

        # predict all active tracks to current time
        for tr in self.tracks.values():
            if tr.state != TrackState.DROPPED:
                tr.predict_to(self.sim_time_ms)

        # ---- handle detections ----------------------------------------
        for det in detections:
            xyz = _det_to_xyz(det)
            matched = _match_detection_to_track(xyz, self.tracks, self.gate_m)

            if matched is None:
                # new contact — only from coarse / fine beams
                if beam.purpose in ("coarse", "fine"):
                    tr = Track.from_detection(
                        xyz,
                        self.sim_time_ms,
                        max_misses=self.max_misses,
                    )
                    self.tracks[tr.id] = tr
                    new_ids.append(tr.id)
                    self._enqueue_fine(tr)
            else:
                self._advance_state(matched, xyz, det, beam)

        # ---- handle misses for tracks that expected a return -----------
        for tr in self.tracks.values():
            if tr.state == TrackState.DROPPED:
                continue
            if tr.pending_beam is None:
                continue
            pend_az, pend_el = tr.pending_beam
            if abs(pend_az - beam.az) < 0.1 and abs(pend_el - beam.el) < 0.1:
                # this beam was meant for this track but no detection matched
                tr.register_miss()
                tr.pending_beam = None
                dropped_ids += self._handle_miss(tr)

        # ---- purge dropped tracks -------------------------------------
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].state == TrackState.DROPPED:
                dropped_ids.append(tid)
                # keep in dict for history display but mark dropped

        return StepResult(
            beam=beam,
            detections=detections,
            tracks=dict(self.tracks),
            sim_time_ms=self.sim_time_ms,
            new_track_ids=list(set(new_ids)),
            dropped_ids=list(set(dropped_ids)),
        )

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------

    def _advance_state(
        self,
        tr:   Track,
        xyz:  np.ndarray,
        det:  Detection,
        beam: Beam,
    ) -> None:
        tr.register_hit(xyz, self.sim_time_ms)
        tr.pending_beam = None

        if tr.state == TrackState.FINE:
            tr.state = TrackState.CONFIRM
            self._enqueue_confirm(tr)

        elif tr.state == TrackState.CONFIRM:
            tr.state = TrackState.TRACKED
            self._schedule_track_update(tr)

        elif tr.state == TrackState.TRACKED:
            self._schedule_track_update(tr)   # reschedule next update

    def _handle_miss(self, tr: Track) -> list[str]:
        """Returns list of dropped track ids."""
        if tr.state == TrackState.TRACKED:
            # demote to fine for recovery
            tr.state = TrackState.FINE
            tr.miss_count = 0
            self._enqueue_fine(tr)

        elif tr.state in (TrackState.FINE, TrackState.CONFIRM):
            if tr.miss_count >= tr.max_misses:
                tr.state = TrackState.DROPPED
                return [tr.id]
            else:
                # retry same mode
                if tr.state == TrackState.FINE:
                    self._enqueue_fine(tr)
                else:
                    self._enqueue_confirm(tr)
        return []

    # ------------------------------------------------------------------
    # beam scheduling helpers
    # ------------------------------------------------------------------

    def _enqueue_coarse_scan(self) -> None:
        sm = self.submode_lib.submodes[self.SUBMODE_COARSE]
        az_min, az_max = self.radar.azimuth_range
        el_min, el_max = self.radar.elevation_range

        az_centers = _bounded_centers(az_min, az_max, sm.azimuth_width, sm.azimuth_width)
        el_centers  = _bounded_centers(el_min, el_max, sm.elevation_width, sm.elevation_width)

        for el in el_centers:
            for az in az_centers:
                self._push(self.PRI_COARSE, Beam(self.SUBMODE_COARSE, az, el, "coarse"))

    def _enqueue_fine(self, tr: Track) -> None:
        _, az, el = tr.kalman.polar_position
        az = float(np.clip(az, *self.radar.azimuth_range))
        el = float(np.clip(el, *self.radar.elevation_range))
        beam = Beam(self.SUBMODE_FINE, az, el, "fine", track_id=tr.id)
        tr.pending_beam = (az, el)
        self._push(self.PRI_FINE, beam)

    def _enqueue_confirm(self, tr: Track) -> None:
        _, az, el = tr.kalman.polar_position
        az = float(np.clip(az, *self.radar.azimuth_range))
        el = float(np.clip(el, *self.radar.elevation_range))
        beam = Beam(self.SUBMODE_CONFIRM, az, el, "confirm", track_id=tr.id)
        tr.pending_beam = (az, el)
        self._push(self.PRI_CONFIRM, beam)

    def _schedule_track_update(self, tr: Track) -> None:
        """
        Adaptive scheduling:
            dt_ms = clip(track_k / speed, dt_min, dt_max)
        Then shortened if position uncertainty is high.
        """
        speed = max(tr.speed, 1.0)
        dt_ms = float(np.clip(self.track_k / speed, self.dt_min_ms, self.dt_max_ms))

        # shorten interval when uncertain
        sigma     = tr.position_uncertainty
        sigma_ref = 200.0   # metres — reference uncertainty
        dt_ms    *= float(np.clip(sigma / sigma_ref, 0.5, 1.0))

        tr._next_track_time_ms = self.sim_time_ms + dt_ms

    def _inject_due_track_beams(self) -> None:
        """Push track-update beams that are now due onto the queue."""
        for tr in self.tracks.values():
            if tr.state != TrackState.TRACKED:
                continue
            if tr.pending_beam is not None:
                continue   # already queued
            due = getattr(tr, "_next_track_time_ms", 0.0)
            if self.sim_time_ms >= due:
                _, az, el = tr.kalman.polar_position
                az   = float(np.clip(az, *self.radar.azimuth_range))
                el   = float(np.clip(el, *self.radar.elevation_range))
                beam = Beam(self.SUBMODE_TRACK, az, el, "track", track_id=tr.id)
                tr.pending_beam = (az, el)
                self._push(self.PRI_TRACK, beam)

    def _push(self, priority: int, beam: Beam) -> None:
        heapq.heappush(self._queue, _QItem(priority=priority, beam=beam))

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.state != TrackState.DROPPED]

    def queue_summary(self) -> dict[str, int]:
        counts = {"track": 0, "confirm": 0, "fine": 0, "coarse": 0}
        for item in self._queue:
            counts[item.beam.purpose] = counts.get(item.beam.purpose, 0) + 1
        return counts