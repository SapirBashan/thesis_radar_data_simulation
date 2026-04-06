import numpy as np
from config.radar_config import RadarConfig
from config.submode_config import SubmodeLibrary, SubmodeTemplate
from config.operating_submode import OperatingSubmode
from config.target import Target

C = 299792458.0


def _wrap_angle_deg(a):
    """Wrap angle to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def _angle_diff_deg(a, b):
    """Shortest signed diff a-b in degrees."""
    return _wrap_angle_deg(a - b)


class RadarSimulator:
    def __init__(self, radar_cfg: RadarConfig, submodes: SubmodeLibrary):
        self.radar_cfg = radar_cfg
        self.submodes = submodes

    def _array_steering_vector(self, az_deg: float, el_deg: float):
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        kx = np.cos(el) * np.cos(az)
        ky = np.cos(el) * np.sin(az)

        rows, cols = self.radar_cfg.array.rows, self.radar_cfg.array.cols
        dx, dy = self.radar_cfg.array.dx, self.radar_cfg.array.dy
        lam = self.radar_cfg.wavelength

        xs = (np.arange(cols) - (cols - 1) / 2.0) * dx
        ys = (np.arange(rows) - (rows - 1) / 2.0) * dy
        xv, yv = np.meshgrid(xs, ys)

        phase = 2 * np.pi / lam * (xv * kx + yv * ky)
        return np.exp(1j * phase).reshape(-1)  # (nelems,)

    def generate(
        self,
        targets: list[Target],
        beam,
        noise_sigma: float = 0.0
    ) -> np.ndarray:
        """
        Returns:
          signal: complex ndarray with shape (nelems, nrg, npri)
        """
        sm = self.submodes.submodes[beam.submode_name]

        nrg = sm.num_range_gates
        npri = sm.num_pulses
        fs = sm.fs

        nelems = self.radar_cfg.array.rows * self.radar_cfg.array.cols
        signal = np.zeros((nelems, nrg, npri), dtype=np.complex64)

        # Time axes
        t_fast = np.arange(nrg) / fs
        pulse_times = np.arange(npri) * sm.pri

        # Filter targets inside submode FOV
        half_az = sm.azimuth_width / 2.0
        half_el = sm.elevation_width / 2.0

        for tgt in targets:
            r, az, el = tgt.polar_position
            daz = _angle_diff_deg(az, beam.az)
            delv = _angle_diff_deg(el, beam.el)

            if abs(daz) > half_az or abs(delv) > half_el:
                continue  # outside FOV

            # Range delay
            tau = 2 * r / C

            # Doppler frequency (monostatic): fd = 2 * vr / lambda
            vr = tgt.radial_velocity
            fd = 2.0 * vr / self.radar_cfg.wavelength

            # Rectangular pulse envelope
            pulse_env = ((t_fast >= tau) & (t_fast <= tau + sm.pulse_width)).astype(np.float32)

            # Doppler phase across pulses
            doppler_phase = np.exp(1j * 2 * np.pi * fd * pulse_times).astype(np.complex64)

            # Target cube (nrg, npri)
            target_cube = pulse_env[:, None] * doppler_phase[None, :]

            # Array manifold per element
            a = self._array_steering_vector(az, el).astype(np.complex64)

            r = max(r, 1.0)  # avoid singularity
            Pt = 10 ** (self.radar_cfg.tx_power / 10.0)
            Gt = 10 ** (self.radar_cfg.tx_gain / 10.0)
            Gr = 10 ** (self.radar_cfg.rx_gain / 10.0)
            lam = self.radar_cfg.wavelength

            Pr = (Pt * Gt * Gr * (lam ** 2) * tgt.rcs) / (((4.0 * np.pi) ** 3) * (r ** 4))
            amp = np.sqrt(Pr).astype(np.float32)

            signal += (a[:, None, None] * (amp * target_cube[None, :, :]))

        # Add complex Gaussian noise N(0, sigma^2)
        if noise_sigma > 0.0:
            noise = (noise_sigma / np.sqrt(2.0)) * (
                np.random.normal(0.0, 1.0, signal.shape) +
                1j * np.random.normal(0.0, 1.0, signal.shape)
            ).astype(np.complex64)
            signal += noise

        return signal