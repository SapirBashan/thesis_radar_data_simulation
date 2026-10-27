import numpy as np
from scipy.ndimage import convolve1d

def range_doppler_power(x: np.ndarray) -> np.ndarray:
    X = np.fft.fft(x, axis=1)
    X = np.fft.fftshift(X, axes=1)
    return np.abs(X) ** 2

def cfar_1d(
    x: np.ndarray,
    guard_cells: int,
    training_cells: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    g = int(guard_cells)
    t = int(training_cells)

    # Training kernel:
    # [1 ... 1] for left training cells
    # [0 ... 0] for guard cells + CUT + guard cells
    # [1 ... 1] for right training cells
    kernel = np.r_[np.ones(t), np.zeros(2 * g + 1), np.ones(t)].astype(np.float64)
    n_train = kernel.sum()

    # Convolve along range axis with 'same' size output
    # This gives the sum of training cells around each CUT
    noise_sum = convolve1d(x.astype(np.float64), kernel, axis=0, mode="constant", cval=0.0)
    noise = noise_sum / n_train

    mask = threshold * noise
    det = x > mask

    # remove detection from the start and end of the mask
    pad = g + t
    if pad > 0:
        det[:pad, ...] = False
        det[-pad:, ...] = False

    return mask, det


def dpc_processing(signal: np.ndarray, pulse_width = 10) -> np.ndarray:
    matched_filter = np.ones(pulse_width, dtype=signal.dtype)
    matched_filter = np.conj(matched_filter)

    compressed_signal = []

    for i in range(signal.shape[1]):
        compressed_signal.append(np.convolve(signal[:, i], matched_filter, mode='same'))

    compressed_signal = np.array(compressed_signal)

    return compressed_signal.T