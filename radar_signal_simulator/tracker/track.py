from uuid import uuid4
from .kalman_filter import KalmanFilter

class Track:
    """
    A track has:
    - Position X, Y, Z
    - Velocity X, Y, Z
    - Kalman Filter
    - Noise Estimation
    """
    def __init__(self, init_pos: tuple[float, float, float]):
        self.id = str(uuid4())
        self.position = init_pos
        self.velocity = (0.0, 0.0, 0.0)
        self.kf = KalamanFilter(self.position,)

    def predict(self):
        return self.kf.predict()