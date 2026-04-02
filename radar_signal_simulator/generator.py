import numpy as np
from .simulator_config import RadarConfig
from .target import Target

C = 299792458.0

class RadarSignalGenerator:
    def __init__(self, cfg: RadarConfig):
        self.cfg = cfg

    def _array_steering_vector(self, az_deg: float, el_deg: float):
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        kx = np.cos(el) * np.cos(az)
        ky = np.cos(el) * np.sin(az)

        rows, cols = self.cfg.array.rows, self.cfg.array.cols
        dx, dy = self.cfg.array.dx, self.cfg.array.dy
        lam = self.cfg.wavelength

        # element positions
        xs = (np.arange(cols) - (cols - 1) / 2.0) * dx
        ys = (np.arange(rows) - (rows - 1) / 2.0) * dy
        xv, yv = np.meshgrid(xs, ys)
        phase = 2 * np.pi / lam * (xv * kx + yv * ky)
        return np.exp(1j * phase).reshape(-1)  # (num_elems,)

    def _db_to_linear(self, db: float):
        return 10 ** (db / 10)

    def generate(self, targets: list[Target]) -> np.ndarray:
        """
        Returns:
          signal: complex ndarray with shape (num_elems, num_pulses, num_range_gates)
        """
        cfg = self.cfg
        num_elems = cfg.array.rows * cfg.array.cols
        signal = np.zeros((num_elems, cfg.num_pulses, cfg.num_range_gates), dtype=np.complex64)

        signal_shape = signal.shape

        t_fast = np.arange(cfg.num_range_gates) / cfg.fs  # fast time within PRI
        pulse_times = np.arange(cfg.num_pulses) * cfg.pri  # slow time

        for tgt in targets:
            # Range delay -> sample index
            tau = 2 * tgt.range / C
            delay_samples = tau * cfg.fs

            # Doppler phase across pulses
            fd = tgt.doppler
            doppler_phase = np.exp(1j * 2 * np.pi * fd * pulse_times)

            # Rectangular pulse envelope shifted by delay
            # Simple: 1 where t_fast is within [tau, tau + pulse_width]
            pulse_env = ((t_fast >= tau) & (t_fast <= tau + cfg.pulse_width)).astype(float)

            # Steering vector for this target
            a = self._array_steering_vector(tgt.azimuth, tgt.elevation)  # (num_elems,)

            # Simple amplitude (RCS / R^4 can be added later)
            amp = np.sqrt(tgt.rcs)

            # Build target contribution
            # shape: (num_pulses, num_range_gates)
            target_cube = amp * doppler_phase[:, None] * pulse_env[None, :]

            # Add to each element with spatial phase
            signal += a[:, None, None] * target_cube[None, :, :]

        # Add noise
        sigma = self._db_to_linear(self.cfg.noise_level)
        mu = 0.0
        noise = np.random.normal(mu, sigma, signal_shape) + 1j * np.random.normal(mu, sigma, signal_shape)

        return signal + noise