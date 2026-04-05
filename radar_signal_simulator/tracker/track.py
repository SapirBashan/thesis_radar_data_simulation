from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

from .kalman import KalmanTracker


class TrackState(Enum):
    FINE    = auto()   # coarse detection → fire fine beam
    CONFIRM = auto()   # fine detection   → fire confirm beam
    TRACKED = auto()   # confirmed        → adaptive track updates
    DROPPED = auto()   # too many misses  → removed


@dataclass
class Track:
    """
    One tracked object.

    hit_history  : list of xyz arrays — every confirmed measurement (for display)
    pred_history : list of xyz arrays — every Kalman prediction     (for display)
    """

    id: str
    state: TrackState
    kalman: KalmanTracker

    # detection that spawned / last updated this track
    last_det_xyz: np.ndarray          # shape (3,)

    miss_count: int = 0
    max_misses: int = 3               # configurable per track

    # display histories
    hit_history:  list = field(default_factory=list)
    pred_history: list = field(default_factory=list)

    # timing for adaptive scheduling
    last_update_time_ms: float = 0.0  # simulation time of last update

    # beam that is currently "in flight" for this track (az, el)
    pending_beam: Optional[tuple[float, float]] = None

    # ------------------------------------------------------------------ #
    # factory
    # ------------------------------------------------------------------ #

    @staticmethod
    def from_detection(
        xyz: np.ndarray,
        sim_time_ms: float,
        max_misses: int = 3,
        process_noise_std: float = 10.0,
        meas_noise_std: float = 50.0,
    ) -> "Track":
        kf = KalmanTracker(
            initial_xyz=xyz,
            process_noise_std=process_noise_std,
            meas_noise_std=meas_noise_std,
        )
        t = Track(
            id=str(uuid.uuid4())[:8],
            state=TrackState.FINE,
            kalman=kf,
            last_det_xyz=xyz.copy(),
            max_misses=max_misses,
            last_update_time_ms=sim_time_ms,
        )
        t.hit_history.append(xyz.copy())
        return t

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def register_hit(self, xyz: np.ndarray, sim_time_ms: float) -> None:
        self.kalman.update(xyz)
        self.last_det_xyz = xyz.copy()
        self.miss_count   = 0
        self.last_update_time_ms = sim_time_ms
        self.hit_history.append(xyz.copy())

    def register_miss(self) -> None:
        self.miss_count += 1

    def predict_to(self, sim_time_ms: float) -> np.ndarray:
        """Predict kalman state to sim_time_ms; appends to pred_history."""
        dt = (sim_time_ms - self.last_update_time_ms) / 1000.0  # ms → s
        dt = max(dt, 0.0)
        xyz = self.kalman.predict(dt)
        self.pred_history.append(xyz.copy())
        return xyz

    @property
    def predicted_polar(self) -> tuple[float, float, float]:
        return self.kalman.polar_position

    @property
    def speed(self) -> float:
        return self.kalman.speed

    @property
    def position_uncertainty(self) -> float:
        return self.kalman.position_uncertainty