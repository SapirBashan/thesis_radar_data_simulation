import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class TrackState:
    x: np.ndarray
    P: np.ndarray


class Track:
    def __init__(
        self,
        detection,
        radar_config,
        submode,
        timestamp: float,
        track_id: int,
        sigma_a: float = 5.0,
        confirm_threshold: int = 3,
        delete_threshold: int = 5,
    ):
        self.id = track_id

        # --- Config ---
        self.radar = radar_config
        self.submode = submode
        self.sigma_a = sigma_a

        # --- Lifecycle ---
        self.hits = 1
        self.misses = 0
        self.state = "TENTATIVE"
        self.confirm_threshold = confirm_threshold
        self.delete_threshold = delete_threshold

        # --- Time ---
        self.last_update_time = timestamp

        # --- EKF state ---
        pos = self._measurement_to_cartesian(detection)
        vel = np.zeros(3)

        self.x = np.zeros(6)
        self.x[:3] = pos
        self.x[3:] = vel

        self.P = np.eye(6) * 100.0

        # --- History ---
        self.history: List[Dict[str, Any]] = []
        self._log_state(timestamp, detection)

    # =========================================================
    # Public API
    # =========================================================
    def predict(self, timestamp: float):
        dt = timestamp - self.last_update_time
        if dt <= 0:
            return

        F = self._build_F(dt)
        Q = self._build_Q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

        self.last_update_time = timestamp

    def update(self, detection, timestamp: float):
        self.predict(timestamp)

        z = self._build_measurement_vector(detection)
        z_hat, H = self._measurement_function_and_jacobian(self.x)
        R = self._measurement_covariance(detection)

        y = z - z_hat
        S = H @ self.P @ H.T + R

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

        # Lifecycle update
        self.hits += 1
        self.misses = 0

        if self.state == "TENTATIVE" and self.hits >= self.confirm_threshold:
            self.state = "CONFIRMED"

        self.last_update_time = timestamp
        self._log_state(timestamp, detection)

    def miss(self, timestamp: float):
        self.predict(timestamp)

        self.misses += 1

        if self.misses >= self.delete_threshold:
            self.state = "DELETED"

        self.last_update_time = timestamp
        self._log_state(timestamp, None)

    # =========================================================
    # Getters
    # =========================================================
    def get_state(self) -> np.ndarray:
        return self.x.copy()

    def get_position(self) -> np.ndarray:
        return self.x[:3].copy()

    def get_velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def get_predicted_measurement(self):
        z_hat, H = self._measurement_function_and_jacobian(self.x)
        R = self._measurement_covariance_from_state()
        S = H @ self.P @ H.T + R
        return z_hat, S

    def is_confirmed(self) -> bool:
        return self.state == "CONFIRMED"

    def is_deleted(self) -> bool:
        return self.state == "DELETED"

    # =========================================================
    # EKF internals
    # =========================================================
    def _build_F(self, dt):
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    def _build_Q(self, dt):
        q = self.sigma_a ** 2
        Q = np.zeros((6, 6))

        for i in range(3):
            Q[i, i] = (dt**4) / 4
            Q[i, i+3] = (dt**3) / 2
            Q[i+3, i] = (dt**3) / 2
            Q[i+3, i+3] = dt**2

        return Q * q

    # =========================================================
    # Measurement model (EKF)
    # =========================================================
    def _measurement_function_and_jacobian(self, x):
        px, py, pz, vx, vy, vz = x

        r = np.linalg.norm([px, py, pz]) + 1e-6
        az = np.arctan2(py, px)
        el = np.arcsin(pz / r)

        vr = (px * vx + py * vy + pz * vz) / r

        z_hat = np.array([r, az, el, vr])

        # Jacobian (approximate but effective)
        H = np.zeros((4, 6))

        # Position derivatives
        H[0, :3] = [px/r, py/r, pz/r]

        H[1, 0] = -py / (px**2 + py**2 + 1e-6)
        H[1, 1] = px / (px**2 + py**2 + 1e-6)

        H[2, 0] = -px * pz / (r**2 * np.sqrt(px**2 + py**2) + 1e-6)
        H[2, 1] = -py * pz / (r**2 * np.sqrt(px**2 + py**2) + 1e-6)
        H[2, 2] = np.sqrt(px**2 + py**2) / (r**2 + 1e-6)

        # Doppler derivatives
        H[3, :3] = (x[3:] * r - vr * x[:3]) / (r**2)
        H[3, 3:] = x[:3] / r

        return z_hat, H

    def _build_measurement_vector(self, detection):
        return np.array([
            detection.range,
            detection.azimuth,
            detection.elevation,
            detection.velocity
        ])

    def _measurement_covariance(self, detection):
        # Simplified (you can plug your radar-based model here)
        sigma_r = 5.0
        sigma_az = 0.01
        sigma_el = 0.01
        sigma_vr = 2.0

        return np.diag([sigma_r**2, sigma_az**2, sigma_el**2, sigma_vr**2])

    def _measurement_covariance_from_state(self):
        # fallback for prediction (used in gating)
        return self._measurement_covariance(None)

    # =========================================================
    # Utilities
    # =========================================================
    def _measurement_to_cartesian(self, detection):
        r, az, el = detection.range, detection.azimuth, detection.elevation

        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        return np.array([x, y, z])

    def _log_state(self, timestamp, detection):
        self.history.append({
            "time": timestamp,
            "state": self.x.copy(),
            "detection": detection
        })