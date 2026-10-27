from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np


@dataclass(frozen=True)
class CFARParams:
    guard_cells: int = 2
    train_cells: int = 12
    threshold_scale: float = 3.5


def generate_raw_signal(nrg: int, npri: int, rng: np.random.Generator) -> np.ndarray:
    """
    Returns complex matrix with shape (npri, nrg):
    - rows: PRI (slow-time)
    - cols: range gate (fast-time)
    """
    # Complex noise floor.
    raw = (
        rng.normal(0.0, 1.0, size=(npri, nrg))
        + 1j * rng.normal(0.0, 1.0, size=(npri, nrg))
    ) * 0.25

    # Add a few synthetic moving targets so plots look interesting.
    n_targets = int(rng.integers(2, 6))
    pri_idx = np.arange(npri, dtype=np.float64)

    for _ in range(n_targets):
        rg0 = int(rng.integers(0, nrg))
        drift = float(rng.uniform(-0.25, 0.25))  # range drift per PRI
        doppler_hz = float(rng.uniform(-0.2, 0.2))
        amp = float(rng.uniform(2.0, 8.0))

        rg_track = np.clip(
            np.round(rg0 + drift * pri_idx).astype(np.int64),
            0,
            nrg - 1,
        )
        phase = np.exp(1j * (2.0 * np.pi * doppler_hz * pri_idx + rng.uniform(0, 2 * np.pi)))
        raw[pri_idx.astype(np.int64), rg_track] += amp * phase

    return raw


def compute_fft_map(raw_signal: np.ndarray) -> np.ndarray:
    """
    Doppler FFT along PRI axis (axis=0), per range gate.
    Output shape is same as input: (npri, nrg).
    """
    fft_map = np.fft.fft(raw_signal, axis=0)
    return np.fft.fftshift(fft_map, axes=0)


def compute_dpc_avg(raw_signal: np.ndarray, tap_size: int = 10) -> np.ndarray:
    if tap_size <= 0:
        raise ValueError("tap_size must be > 0")

    npri, nrg = raw_signal.shape
    eff_tap = min(tap_size, npri)  # key fix for short PRI modes
    kernel = np.ones(eff_tap, dtype=np.complex128)
    filtered = np.empty_like(raw_signal, dtype=np.complex128)

    for rg in range(nrg):
        filtered[:, rg] = np.convolve(raw_signal[:, rg], kernel, mode="same")

    return np.abs(np.mean(filtered, axis=0)).astype(np.float64)



def compute_cfar_mask(dpc_avg: np.ndarray, params: CFARParams) -> np.ndarray:
    """
    Basic 1D CA-CFAR over range gates.
    Returns float mask (0.0/1.0) with same length as dpc_avg.
    """
    x = np.asarray(dpc_avg, dtype=np.float64)
    n = x.size
    mask = np.zeros(n, dtype=np.float64)

    g = params.guard_cells
    t = params.train_cells
    k = params.threshold_scale
    span = g + t

    if n < 2 * span + 1:
        return mask

    for i in range(span, n - span):
        left_train = x[i - span : i - g]
        right_train = x[i + g + 1 : i + span + 1]
        noise_level = np.mean(np.concatenate((left_train, right_train)))
        threshold = k * noise_level
        if x[i] > threshold:
            mask[i] = 1.0

    return mask


def write_complex_2d_txt(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    npri, nrg = data.shape
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        f.write(f"{nrg} {npri}\n")
        flat = data.reshape(-1)
        for v in flat:
            f.write(f"{float(np.real(v)):.9e} {float(np.imag(v)):.9e}\n")

    # Retry logic with backoff
    max_retries = 5
    for attempt in range(max_retries):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.01 * (2 ** attempt))  # exponential backoff
            else:
                raise


def write_float_1d_txt(path: Path, data: np.ndarray) -> None:
    """
    Writes float vector in monitor format:
      first line: "n"
      then n lines: "value"
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as f:
        f.write(f"{x.size}\n")
        for v in x:
            f.write(f"{float(v):.9e}\n")

    tmp.replace(path)


def build_snapshot(nrg: int, npri: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = generate_raw_signal(nrg=nrg, npri=npri, rng=rng)
    fft_map = compute_fft_map(raw)
    dpc_avg = compute_dpc_avg(raw, tap_size=10)
    cfar_mask = compute_cfar_mask(dpc_avg, CFARParams())
    return raw, dpc_avg, fft_map, cfar_mask
