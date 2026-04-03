import numpy as np

def dpc(signal: np.ndarray, fs: float, pulse_width: float) -> np.ndarray:
    n_rg = signal.shape[0]
    h = np.ones(int(round(fs * pulse_width)), dtype=signal.dtype)

    X = np.fft.fft(signal, axis=0)
    H = np.fft.fft(h, n=n_rg).reshape(n_rg, 1, 1)

    return np.fft.ifft(X * H, axis=0)