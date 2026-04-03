import numpy as np

def mti(x: np.ndarray, *, order: int = 1, axis: int = -1) -> np.ndarray:
    x = np.asarray(x)

    if order == 0:
        return x

    if order == 1:
        y = np.diff(x, n=1, axis=axis)
        # pad one leading zero to keep same shape
        pad = [(0, 0)] * x.ndim
        pad[axis] = (1, 0)
        return np.pad(y, pad, mode="constant", constant_values=0)

    if order == 2:
        y = np.diff(x, n=2, axis=axis)
        # pad two leading zeros to keep same shape
        pad = [(0, 0)] * x.ndim
        pad[axis] = (2, 0)
        return np.pad(y, pad, mode="constant", constant_values=0)

    return x