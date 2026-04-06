from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .kalman import KalmanTracker
from .states import TrackState


@dataclass
class Track:
    """
    One tracked object hypothesis.

    hit_history  : confirmed xyz measurements
    pred_history : predicted xyz states
    """
    id: str
    state: TrackState
    kalman: KalmanTracker

    # last confirmed measurement
    last_det_xyz: np.ndarray  # shape (3,)

    # miss handling
    miss_count: int = 0
    max_misses: int = 3

    # confirmation / tracking quality
    confirm_count: int = 0
    hit_count: int = 0
    confidence: float = 0.0

    # display / debug histories
    hit_history: list[np.ndarray] = field(default_factory=list)
    pred_history: list[np.ndarray] = field(default_factory=list)

    # timing
    last_update_time_ms: float = 0.0
    next_revisit_time_ms: float = 0.0

    # beam currently assigned/in flight
    pending_beam: Optional[tuple[float, float]] = None

    # optional predicted polar location cache
    predicted_range_m: float = 0.0
    predicted_az_deg: float = 0.0
    predicted_el_deg: float = 0.0

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_detection(
        xyz: np.ndarray,
        sim_time_ms: float,
        max_misses: int = 3,
        process_noise_std: float = 10.0,
        meas_noise_std: float = 50.0,
    ) -> "Track":
        xyz = np.asarray(xyz, dtype=float).reshape(3)

        kf = KalmanTracker(
            initial_xyz=xyz,
            process_noise_std=process_noise_std,
            meas_noise_std=meas_noise_std,
        )

        t = Track(
            id=str(uuid.uuid4())[:8],
            state=TrackState.CANDIDATE,
            kalman=kf,
            last_det_xyz=xyz.copy(),
            max_misses=max_misses,
            last_update_time_ms=sim_time_ms,
            next_revisit_time_ms=sim_time_ms,
            hit_count=1,
            confidence=0.25,
        )
        t.hit_history.append(xyz.copy())
        return t

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def register_hit(self, xyz: np.ndarray, sim_time_ms: float) -> None:
        xyz = np.asarray(xyz, dtype=float).reshape(3)

        self.kalman.update(xyz)
        self.last_det_xyz = xyz.copy()
        self.miss_count = 0
        self.last_update_time_ms = sim_time_ms
        self.hit_history.append(xyz.copy())
        self.hit_count += 1
        self.confidence = min(1.0, self.confidence + 0.2)

        if self.state == TrackState.CANDIDATE and self.hit_count >= 2:
            self.state = TrackState.CONFIRMING
        elif self.state == TrackState.CONFIRMING and self.hit_count >= 3:
            self.state = TrackState.TRACKED
        elif self.state == TrackState.LOST:
            self.state = TrackState.TRACKED

    def register_miss(self) -> None:
        self.miss_count += 1
        self.confidence = max(0.0, self.confidence - 0.1)

        if self.state == TrackState.TRACKED and self.miss_count > 0:
            self.state = TrackState.LOST
        elif self.miss_count >= self.max_misses:
            self.state = TrackState.DROPPED

    def predict_to(self, sim_time_ms: float) -> np.ndarray:
        """Predict Kalman state to sim_time_ms and store it in pred_history."""
        dt = (sim_time_ms - self.last_update_time_ms) / 1000.0
        dt = max(dt, 0.0)

        xyz = self.kalman.predict(dt)
        self.pred_history.append(np.asarray(xyz, dtype=float).copy())

        return xyz

    def set_next_revisit(self, sim_time_ms: float, dt_ms: float) -> None:
        self.next_revisit_time_ms = sim_time_ms + dt_ms

    # ------------------------------------------------------------------
    # convenience properties
    # ------------------------------------------------------------------

    @property
    def predicted_polar(self) -> tuple[float, float, float]:
        return self.kalman.polar_position

    @property
    def speed(self) -> float:
        return self.kalman.speed

    @property
    def position_uncertainty(self) -> float:
        return self.kalman.position_uncertainty