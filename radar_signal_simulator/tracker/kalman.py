import numpy as np


class KalmanTracker:
    """
    6-state constant-velocity Kalman filter in Cartesian XYZ.

    State vector: x = [x, vx, y, vy, z, vz]^T

    F (transition):
        [ 1 dt  0  0  0  0 ]
        [ 0  1  0  0  0  0 ]
        [ 0  0  1 dt  0  0 ]
        [ 0  0  0  1  0  0 ]
        [ 0  0  0  0  1 dt ]
        [ 0  0  0  0  0  1 ]

    H (measurement): picks [x, y, z] from state.

    Adaptive dt: predict() receives the actual elapsed time.
    Process noise Q scales with dt^2 so fast targets stay responsive.
    """

    DIM_X = 6  # state dimension
    DIM_Z = 3  # measurement dimension (x, y, z)

    def __init__(
        self,
        initial_xyz: np.ndarray,
        initial_velocity: np.ndarray | None = None,
        process_noise_std: float = 10.0,   # m/s^2  — model acceleration uncertainty
        meas_noise_std: float = 50.0,      # m      — position measurement noise
    ):
        # ---- state & covariance ----
        self.x = np.zeros(self.DIM_X)
        self.x[0] = initial_xyz[0]
        self.x[2] = initial_xyz[1]
        self.x[4] = initial_xyz[2]
        if initial_velocity is not None:
            self.x[1] = initial_velocity[0]
            self.x[3] = initial_velocity[1]
            self.x[5] = initial_velocity[2]

        # large initial uncertainty on position; moderate on velocity
        self.P = np.diag([200.0**2, 30.0**2,
                          200.0**2, 30.0**2,
                          200.0**2, 30.0**2])

        # measurement matrix  H: (3, 6)
        self.H = np.zeros((self.DIM_Z, self.DIM_X))
        self.H[0, 0] = 1.0   # x
        self.H[1, 2] = 1.0   # y
        self.H[2, 4] = 1.0   # z

        # measurement noise covariance  R: (3, 3)
        self.R = np.eye(self.DIM_Z) * meas_noise_std ** 2

        self._q = process_noise_std   # kept for Q rebuild in predict()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> np.ndarray:
        """Propagate state forward by dt seconds. Returns predicted xyz."""
        F = self._build_F(dt)
        Q = self._build_Q(dt)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.position_xyz

    def update(self, xyz_meas: np.ndarray) -> None:
        """Correct state with a new XYZ measurement."""
        z = np.asarray(xyz_meas, dtype=float)

        y  = z - self.H @ self.x                          # innovation
        S  = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K  = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain

        self.x = self.x + K @ y
        I_KH  = np.eye(self.DIM_X) - K @ self.H
        self.P = I_KH @ self.P                             # Joseph form omitted for brevity

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def position_xyz(self) -> np.ndarray:
        return np.array([self.x[0], self.x[2], self.x[4]])

    @property
    def velocity_xyz(self) -> np.ndarray:
        return np.array([self.x[1], self.x[3], self.x[5]])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity_xyz))

    @property
    def position_uncertainty(self) -> float:
        """1-sigma position uncertainty (mean of x/y/z std devs) in metres."""
        return float(np.sqrt((self.P[0, 0] + self.P[2, 2] + self.P[4, 4]) / 3.0))

    @property
    def polar_position(self) -> tuple[float, float, float]:
        """Returns (range_m, az_deg, el_deg)."""
        x, y, z = self.position_xyz
        r  = float(np.sqrt(x**2 + y**2 + z**2)) + 1e-12
        az = float(np.degrees(np.arctan2(y, x)))
        el = float(np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2))))
        return r, az, el

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_F(self, dt: float) -> np.ndarray:
        F = np.eye(self.DIM_X)
        # couple position with velocity for each axis
        F[0, 1] = dt
        F[2, 3] = dt
        F[4, 5] = dt
        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        """
        Discrete white-noise acceleration model.
        Q_axis = q^2 * [ dt^4/4  dt^3/2 ]
                        [ dt^3/2  dt^2   ]
        """
        q  = self._q
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        block = (q**2) * np.array([
            [dt4 / 4.0, dt3 / 2.0],
            [dt3 / 2.0, dt2      ],
        ])

        Q = np.zeros((self.DIM_X, self.DIM_X))
        Q[0:2, 0:2] = block
        Q[2:4, 2:4] = block
        Q[4:6, 4:6] = block
        return Q