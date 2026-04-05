from dataclasses import dataclass

import numpy as np


@dataclass
class CFARParams:
    guard_cells: int
    training_cells: int
    threshold: float

from scipy.ndimage import convolve1d

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

def cfar_2d(
    p: np.ndarray,
    guard_rg: int,
    train_rg: int,
    guard_pri: int,
    train_pri: int,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:

    n_rg, n_pri, n_beams = p.shape
    gr, tr = int(guard_rg), int(train_rg)
    gp, tp = int(guard_pri), int(train_pri)

    # Total half-window extents
    wr = gr + tr
    wp = gp + tp

    noise = np.zeros_like(p, dtype=p.dtype)

    for r in range(n_rg):
        r0 = max(0, r - wr)
        r1 = min(n_rg, r + wr + 1)
        rg0 = max(0, r - gr)
        rg1 = min(n_rg, r + gr + 1)

        for d in range(n_pri):
            d0 = max(0, d - wp)
            d1 = min(n_pri, d + wp + 1)
            dg0 = max(0, d - gp)
            dg1 = min(n_pri, d + gp + 1)

            # Outer window sum
            outer = p[r0:r1, d0:d1, :].sum(axis=(0, 1))
            # Guard window sum (includes CUT)
            guard = p[rg0:rg1, dg0:dg1, :].sum(axis=(0, 1))

            n_outer = (r1 - r0) * (d1 - d0)
            n_guard = (rg1 - rg0) * (dg1 - dg0)
            n_train = n_outer - n_guard
            if n_train <= 0:
                continue

            noise[r, d, :] = (outer - guard) / n_train

    det = p > (threshold * noise)
    return noise, det