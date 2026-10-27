import numpy as np


def reconstruction_metrics(original, restored, phase_floor):
    x, y = original.iq().astype(np.complex128), restored.iq().astype(np.complex128)
    if x.shape != y.shape:
        raise ValueError('Shape mismatch')
    mse = float(np.mean(np.abs(x - y) ** 2))
    power = float(np.mean(np.abs(x) ** 2))
    sqnr = float('inf') if mse == 0 else (10 * np.log10(power / mse) if power else -float('inf'))
    valid = (np.abs(x) > phase_floor) & (np.abs(y) > phase_floor)
    phase = np.angle(y[valid] * np.conj(x[valid]))
    return dict(mse=mse, sqnr_db=sqnr, phase_rmse_rad=float(np.sqrt(np.mean(phase ** 2))) if phase.size else float('nan'),
                phase_valid_samples=int(valid.sum()), phase_eligible_samples=int((np.abs(x) > phase_floor).sum()),
                bitwise_equal=original.data.tobytes() == restored.data.tobytes())
