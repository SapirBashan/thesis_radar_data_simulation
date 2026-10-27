import numpy as np

def dpc_processing(signal: np.ndarray, pulse_width: int) -> np.ndarray:
    matched_filter = np.ones(pulse_width, dtype=signal.dtype)
    matched_filter = np.conj(matched_filter)

    compressed_signal = []

    for i in range(signal.shape[1]):
        compressed_signal.append(np.convolve(signal[:, i], matched_filter, mode='same'))

    compressed_signal = np.array(compressed_signal)

    return compressed_signal.T