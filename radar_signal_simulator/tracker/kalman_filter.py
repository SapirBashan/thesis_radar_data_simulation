import numpy as np
from typing import Dict

class KalmanFilter:
    """
    Kalman Filter with a constant velocity model.
    State: [x, y, z, vx, vy, vz]
    """
    def __init__(self, initial_pos: np.ndarray, noise: np.ndarray, dt: float = 1.0, process_noise: float = 1.0):
        self.x = np.zeros(6)
        self.x[:3] = initial_pos
        self.P = np.eye(6) * 500.0

        self.F = np.eye(6)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        self.Q = np.eye(6) * process_noise
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        self.R = np.diag(noise)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement_pos: np.ndarray):
        y = measurement_pos - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(6) - K @ self.H) @ self.P
        except np.linalg.LinAlgError:
            # If inversion fails, skip the update to avoid instability
            pass