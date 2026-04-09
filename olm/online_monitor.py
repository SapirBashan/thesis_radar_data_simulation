from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d


def dpc_processing(signal: np.ndarray, pulse_width: int = 10) -> np.ndarray:
    """
    Performs Digital Pulse Compression (Matched Filtering) along the range axis.
    Assuming signal shape is (npri, nrg).
    """
    # Matched filter for a simple rectangular pulse (boxcar)
    matched_filter = np.ones(pulse_width, dtype=signal.dtype)

    # Vectorized convolution along the range axis (axis 1) for performance
    return convolve1d(signal, matched_filter, axis=1, mode='constant', cval=0.0)


def cfar_1d(
        x: np.ndarray,
        guard_cells: int,
        training_cells: int,
        threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1D Cell-Averaging CFAR.
    Returns: (threshold_values, detection_mask)
    """
    g = int(guard_cells)
    t = int(training_cells)

    # Training kernel: [1...1 (T) | 0...0 (2G+1) | 1...1 (T)]
    kernel = np.r_[np.ones(t), np.zeros(2 * g + 1), np.ones(t)].astype(np.float64)
    n_train = kernel.sum()

    # Calculate noise floor by averaging training cells
    noise_sum = convolve1d(x.astype(np.float64), kernel, axis=0, mode="constant", cval=0.0)
    noise_avg = noise_sum / n_train

    # Adaptive threshold
    thresh_val = threshold * noise_avg
    det_mask = x > thresh_val

    # Blank out edges where the kernel isn't fully contained
    pad = g + t
    if pad > 0:
        det_mask[:pad] = False
        det_mask[-pad:] = False

    return thresh_val, det_mask


class OnlineMonitor:
    """
    Real-time radar signal processing monitor.
    Visualizes Raw Power, Range-Doppler, DPC+CFAR, and Integrated Doppler.
    """

    def __init__(self, show_every: int = 1):
        self.show_every = max(1, int(show_every))
        self.step = 0

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 6))
        self.fig.suptitle("Radar Online Signal Processing Monitor", fontsize=10, y=0.985)
        self.fig.subplots_adjust(left=0.07, right=0.95, bottom=0.08, top=0.90, wspace=0.32, hspace=0.38)

        self.ax11, self.ax12 = self.axes[0, 0], self.axes[0, 1]
        self.ax21, self.ax22 = self.axes[1, 0], self.axes[1, 1]

        # Persistent artist references for fast updates
        self._im11 = self._cb11 = None
        self._im12 = self._cb12 = None
        self._line21_dpc = self._line21_cfar = self._sc21 = None
        self._line22 = None

        self._meta_text = self.fig.text(
            0.5, 0.952,
            "submode: N/A | data: N/A MB | current: N/A GB/s | time: N/A ms",
            ha="center", va="center", fontsize=8,
        )

    @staticmethod
    def __safe_remove_colorbar(cb):
        if cb is None: return None
        try:
            cb.remove()
        except Exception:
            ax = getattr(cb, "ax", None)
            if ax:
                try:
                    ax.remove()
                except Exception:
                    pass
        return None

    def update(
            self,
            raw_signal: np.ndarray,
            submode_stat: dict | None = None,
            step: int | None = None
    ) -> None:
        self.step = self.step + 1 if step is None else int(step)

        if self.step % self.show_every != 0:
            return

        self.__plot_11(raw_signal)
        self.__plot_12(raw_signal)
        self.__plot_21(raw_signal)
        self.__plot_22(raw_signal)
        self.__update_submode_text(submode_stat)

        # Uniform tick styling
        for ax in self.axes.flatten():
            ax.tick_params(axis="both", which="major", labelsize=6)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)

    def __update_submode_text(self, stat: dict | None) -> None:
        if not stat:
            self._meta_text.set_text("submode: N/A | data: N/A MB | current: N/A GB/s | time: N/A ms")
            return

        amount_mb = float(stat.get("amount_bytes", 0)) / 1e6
        self._meta_text.set_text(
            f"submode: {stat.get('submode_name', 'N/A')} | "
            f"data: {amount_mb:.3f} MB | "
            f"current: {float(stat.get('throughput_gbps', 0)):.3f} GB/s | "
            f"time: {float(stat.get('elapsed_ms', 0)):.3f} ms | "
            f"running: {float(stat.get('running_avg_gbps', 0)):.3f} GB/s"
        )

    def __plot_11(self, raw_signal: np.ndarray) -> None:
        """(1,1) Raw Power Heatmap (dB)"""
        power_db = 20 * np.log10(np.abs(raw_signal) + 1e-12)

        if self._im11 is None or self._im11.get_array().shape != power_db.shape:
            self._cb11 = self.__safe_remove_colorbar(self._cb11)
            self.ax11.clear()
            self._im11 = self.ax11.imshow(power_db, aspect="auto", origin="lower", cmap="viridis")
            self.ax11.set_title("Raw Signal Power (dB)", fontsize=8)
            self.ax11.set_xlabel("Range Bin", fontsize=8)
            self.ax11.set_ylabel("PRI Index", fontsize=8)
            self._cb11 = self.fig.colorbar(self._im11, ax=self.ax11, fraction=0.046, pad=0.04)
        else:
            self._im11.set_data(power_db)
            self._im11.set_clim(vmin=float(np.min(power_db)), vmax=float(np.max(power_db)))

    def __plot_12(self, raw_signal: np.ndarray) -> None:
        """(1,2) Range-Doppler Matrix (dB)"""
        # FFT along pulse axis (0) to resolve Doppler
        rd = np.fft.fftshift(np.fft.fft(raw_signal, axis=0), axes=0)
        rd_db = 20 * np.log10(np.abs(rd) + 1e-12)

        if self._im12 is None or self._im12.get_array().shape != rd_db.shape:
            self._cb12 = self.__safe_remove_colorbar(self._cb12)
            self.ax12.clear()
            self._im12 = self.ax12.imshow(rd_db, aspect="auto", origin="lower", cmap="plasma")
            self.ax12.set_title("Range-Doppler Map (dB)", fontsize=8)
            self.ax12.set_xlabel("Range Bin", fontsize=8)
            self.ax12.set_ylabel("Doppler Bin", fontsize=8)
            self._cb12 = self.fig.colorbar(self._im12, ax=self.ax12, fraction=0.046, pad=0.04)
        else:
            self._im12.set_data(rd_db)
            self._im12.set_clim(vmin=float(np.min(rd_db)), vmax=float(np.max(rd_db)))

    def __plot_21(self, raw_signal: np.ndarray) -> None:
        """(2,1) DPC + CFAR Range Profile (dB)"""
        compressed = dpc_processing(raw_signal)
        # Average power across PRI (axis 0) for 1D range profile
        dpc_power = np.mean(np.abs(compressed) ** 2, axis=0)
        dpc_db = 10 * np.log10(dpc_power + 1e-12)

        thresh_linear, detections = cfar_1d(dpc_power, guard_cells=4, training_cells=8, threshold=2.0)
        thresh_db = 10 * np.log10(thresh_linear + 1e-12)

        x = np.arange(dpc_db.size)
        det_idx = np.where(detections)[0]

        if self._line21_dpc is None or self._line21_dpc.get_xdata().size != x.size:
            self.ax21.clear()
            (self._line21_dpc,) = self.ax21.plot(x, dpc_db, color="tab:blue", lw=1.2, label="DPC")
            (self._line21_cfar,) = self.ax21.plot(x, thresh_db, "r--", lw=1.0, alpha=0.8, label="CFAR")
            self._sc21 = self.ax21.scatter(det_idx, dpc_db[det_idx], s=30, c="red", marker="x", zorder=3)
            self.ax21.set_title("DPC Range Profile + CFAR (dB)", fontsize=8)
            self.ax21.set_ylabel("Power (dB)", fontsize=8)
            self.ax21.legend(loc="upper right", fontsize=6)
            self.ax21.grid(True, alpha=0.2)
        else:
            self._line21_dpc.set_data(x, dpc_db)
            self._line21_cfar.set_data(x, thresh_db)
            offsets = np.column_stack((det_idx, dpc_db[det_idx])) if det_idx.size > 0 else np.empty((0, 2))
            self._sc21.set_offsets(offsets)
            self.ax21.relim()
            self.ax21.autoscale_view()

    def __plot_22(self, raw_signal: np.ndarray) -> None:
        """(2,2) Integrated Doppler Spectrum (dB)"""
        # FFT along pulses, then average power across range gates
        doppler_data = np.fft.fftshift(np.fft.fft(raw_signal, axis=0), axes=0)
        mean_doppler_pwr = np.mean(np.abs(doppler_data) ** 2, axis=1)
        doppler_db = 10 * np.log10(mean_doppler_pwr + 1e-12)

        x = np.arange(doppler_db.size) - (doppler_db.size // 2)

        if self._line22 is None or self._line22.get_xdata().size != x.size:
            self.ax22.clear()
            (self._line22,) = self.ax22.plot(x, doppler_db, color="tab:purple", lw=1.2)
            self.ax22.set_title("Integrated Doppler Spectrum (dB)", fontsize=8)
            self.ax22.set_xlabel("Doppler Bin", fontsize=8)
            self.ax22.grid(True, alpha=0.2)
        else:
            self._line22.set_data(x, doppler_db)
            self.ax22.relim()
            self.ax22.autoscale_view()