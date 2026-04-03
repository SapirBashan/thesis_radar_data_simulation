import numpy as np

def range_doppler_power(x: np.ndarray) -> np.ndarray:
    X = np.fft.fft(x, axis=1)
    X = np.fft.fftshift(X, axes=1)
    return np.abs(X) ** 2

C0 = 299_792_458.0  # speed of light [m/s]

def doppler_axis_hz(n_pri: int, prf: float, *, fftshift: bool = True) -> np.ndarray:
    """Doppler frequency axis (Hz) for an N-point FFT with sampling rate = PRF."""
    f = np.fft.fftfreq(n_pri, d=1.0 / prf)
    return np.fft.fftshift(f) if fftshift else f

def doppler_velocity_axis_mps(n_pri: int, prf: float, fc: float, *, fftshift: bool = True) -> np.ndarray:
    """Radial velocity axis (m/s) using v = fd * lambda / 2."""
    lam = C0 / fc
    fd = doppler_axis_hz(n_pri, prf, fftshift=fftshift)
    return fd * (lam / 2.0)

def doppler_velocity_from_bin(bin_idx: int, n_pri: int, prf: float, fc: float, *, fftshift: bool = True) -> float:
    """Velocity (m/s) for a specific Doppler bin index."""
    v = doppler_velocity_axis_mps(n_pri, prf, fc, fftshift=fftshift)
    return float(v[int(bin_idx)])