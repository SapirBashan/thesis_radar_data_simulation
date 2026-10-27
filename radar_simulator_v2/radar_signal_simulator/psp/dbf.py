import numpy as np

def dbf_bfcu_mode(signal: np.ndarray) -> np.ndarray:
    return np.sum(signal, axis=0)

def dbf_subarray_mode(signal: np.ndarray, weight: np.ndarray) -> np.ndarray:
    # weight: (n_beam, n_elem), signal: (n_elem, n_rg, n_pri) -> (n_rg, n_pri, n_beam)
    return np.einsum("be,erp->rpb", np.conj(weight), signal, optimize=True)