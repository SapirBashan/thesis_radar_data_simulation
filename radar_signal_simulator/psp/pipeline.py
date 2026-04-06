import numpy as np

from config import RadarConfig, OperatingSubmode, SubmodeLibrary, SubmodeTemplate

from .dbf import dbf_bfcu_mode
from .dpc import dpc
from .fft import range_doppler_power, doppler_velocity_from_bin
from .cfar import cfar_1d, CFARParams
from .detections import Detection

C0 = 299_792_458.0  # m/s

def _merge_adjacent_range_hits(det_rg: np.ndarray, p_rg: np.ndarray) -> list[tuple[float, float]]:
    """
    Merge adjacent detections in a 1D boolean array.
    Returns list of (rg_index_weighted, power_sum).
    Weight is p_rg at the cell.
    """
    n_rg = det_rg.shape[0]
    out = []
    i = 0
    while i < n_rg:
        if not det_rg[i]:
            i += 1
            continue
        j = i
        while j < n_rg and det_rg[j]:
            j += 1

        idx = np.arange(i, j)
        w = p_rg[i:j].astype(np.float64)
        wsum = float(w.sum())
        if wsum > 0:
            rg_w = float((idx * w).sum() / wsum)
        else:
            rg_w = float((i + j - 1) / 2.0)

        out.append((rg_w, wsum))
        i = j
    return out


def psp_process(
    raw: np.ndarray,
    radar: RadarConfig,
    submode_lib: SubmodeLibrary,
    beam,
    cfar_params: CFARParams
) -> list[Detection]:
    sbm: SubmodeTemplate = submode_lib.submodes[beam.submode_name]
    fs = sbm.fs
    prf = sbm.prf
    pulse_width = sbm.pulse_width
    fc = radar.fc

    beam_az = beam.az
    beam_el = beam.el

    x_dbf = dbf_bfcu_mode(raw)
    # make shape (1, nrg, npri)
    x_dbf = x_dbf[np.newaxis, :]

    x_dpc = dpc(x_dbf, fs=fs, pulse_width=pulse_width)
    # reshape from (1, nrg, npri) -> (nrg, npri)
    x_dpc = x_dpc.reshape(x_dpc.shape[1], x_dpc.shape[2])

    x_pri_avg = np.sum(np.abs(x_dpc) ** 2, axis=1)
    # reshape from (nrg, ) -> (nrg, 1)
    x_pri_avg = x_pri_avg[: , np.newaxis]

    guard_cells = cfar_params.guard_cells
    training_cells = cfar_params.training_cells
    threshold = cfar_params.threshold

    noise_rg, det_rg = cfar_1d(x_pri_avg, guard_cells, training_cells, threshold)

    p_rd = range_doppler_power(x_dpc)

    n_rg, n_pri = x_dpc.shape
    detections = []

    merged = _merge_adjacent_range_hits(det_rg, x_pri_avg)
    for rg_w, p_sum in merged:
        rg_bin = int(round(rg_w))
        rg_bin = max(0, min(n_rg - 1, rg_bin))
        range_m = (rg_bin / fs) * (C0 / 2.0)

        # Doppler bin: pick max power at that range gate
        dop_bin = int(np.argmax(p_rd[rg_bin, :]))

        # Velocity
        v_mps = doppler_velocity_from_bin(dop_bin, n_pri, prf, fc, fftshift=True)

        # SNR estimate: simple ratio using CFAR noise estimate at the representative bin
        noise = float(noise_rg[rg_bin, 0])
        sig = float(x_pri_avg[rg_bin, 0])
        snr_lin = sig / (noise + 1e-12)
        snr_db = 10.0 * np.log10(snr_lin + 1e-12)

        det = Detection(
            range_m,
            v_mps,
            snr_db,
            beam_az,
            beam_el,
        )

        detections.append(det)

    return detections