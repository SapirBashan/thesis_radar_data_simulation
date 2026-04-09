from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d

def dpc_processing(signal: np.ndarray, pulse_width = 10) -> np.ndarray:
    matched_filter = np.ones(pulse_width, dtype=signal.dtype)
    matched_filter = np.conj(matched_filter)

    compressed_signal = []

    for i in range(signal.shape[1]):
        compressed_signal.append(np.convolve(signal[:, i], matched_filter, mode='same'))

    compressed_signal = np.array(compressed_signal)

    return compressed_signal.T

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

# Assuming these are imported from your processing modules
# from your_module import dpc, cfar_1d

class OnlineMonitor:
    """
    Data-driven online monitor for radar processing snapshots.
    Receives raw NumPy arrays directly from an external data source.
    """

    def __init__(self, show_every: int = 1):
        self.show_every = max(1, int(show_every))
        self.step = 0

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 6))
        self.fig.suptitle("Online Signal Processing Monitor (dB Scale)", fontsize=10, y=0.985)
        self.fig.subplots_adjust(left=0.07, right=0.95, bottom=0.08, top=0.90, wspace=0.38, hspace=0.42)

        self.ax11, self.ax12 = self.axes[0, 0], self.axes[0, 1]
        self.ax21, self.ax22 = self.axes[1, 0], self.axes[1, 1]

        # Persistent artists
        self._im11 = None;
        self._cb11 = None
        self._im12 = None;
        self._cb12 = None
        self._line21_dpc = None;
        self._line21_cfar = None;
        self._sc21 = None

        # Placeholder for (2,2)
        self._im22 = None;
        self._cb22 = None

        self._meta_text = self.fig.text(
            0.5, 0.952,
            "submode: N/A | data: N/A MB | current: N/A GB/s | time: N/A ms | running: N/A GB/s",
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
        """Called when a full frame of raw data is ready."""
        self.step = self.step + 1 if step is None else int(step)

        if self.step % self.show_every != 0:
            return

        self.__plot_11(raw_signal)
        self.__plot_12(raw_signal)
        self.__plot_21(raw_signal)
        self.__plot_22(raw_signal)  # Placeholder implementation
        self.__update_submode_text(submode_stat)

        # Formatting ticks
        for ax in (self.ax11, self.ax12, self.ax21, self.ax22):
            ax.tick_params(axis="both", which="major", labelsize=6)

        # Update colorbar tick sizes
        for cb in (self._cb11, self._cb12, self._cb22):
            if cb: cb.ax.tick_params(labelsize=6)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)

    def __update_submode_text(self, stat: dict[str, float | str | int] | None) -> None:
        if not stat:
            self._meta_text.set_text(
                "submode: N/A | data: N/A MB | current: N/A GB/s | time: N/A ms | running: N/A GB/s")
            return

        amount_mb = float(stat["amount_bytes"]) / 1e6
        self._meta_text.set_text(
            f"submode: {stat['submode_name']} | "
            f"data: {amount_mb:.3f} MB | "
            f"current: {float(stat['throughput_gbps']):.3f} GB/s | "
            f"time: {float(stat['elapsed_ms']):.3f} ms | "
            f"running: {float(stat['running_avg_gbps']):.3f} GB/s"
        )

    def __plot_11(self, raw_signal: np.ndarray) -> None:
        """(1,1) Raw signal matrix in dB"""
        # Convert to power in dB
        power_db = 20 * np.log10(np.abs(raw_signal) + 1e-12)

        if self._im11 is None or self._im11.get_array().shape != power_db.shape:
            self._cb11 = self.__safe_remove_colorbar(self._cb11)
            self.ax11.clear()
            self._im11 = self.ax11.imshow(
                power_db, aspect="auto", origin="lower", interpolation="nearest", cmap="viridis"
            )
            self.ax11.set_title("Raw Signal Power (dB)", fontsize=8)
            self.ax11.set_xlabel("Range gate (nrg)", fontsize=8)
            self.ax11.set_ylabel("PRI (npri)", fontsize=8)
            self._cb11 = self.fig.colorbar(self._im11, ax=self.ax11, fraction=0.046, pad=0.04)
        else:
            self._im11.set_data(power_db)
            self._im11.set_clim(vmin=float(np.min(power_db)), vmax=float(np.max(power_db)))

    def __plot_12(self, raw_signal: np.ndarray) -> None:
        """(1,2) Range-Doppler Matrix in dB"""
        # FFT along slow-time (axis=0 for npri) to get Doppler
        rd = np.fft.fftshift(np.fft.fft(raw_signal, axis=0), axes=0)
        rd_power_db = 20 * np.log10(np.abs(rd) + 1e-12)

        if self._im12 is None or self._im12.get_array().shape != rd_power_db.shape:
            self._cb12 = self.__safe_remove_colorbar(self._cb12)
            self.ax12.clear()
            self._im12 = self.ax12.imshow(
                rd_power_db, aspect="auto", origin="lower", interpolation="nearest", cmap="plasma"
            )
            self.ax12.set_title("Range-Doppler Map (dB)", fontsize=8)
            self.ax12.set_xlabel("Range gate (nrg)", fontsize=8)
            self.ax12.set_ylabel("Doppler bin", fontsize=8)
            self._cb12 = self.fig.colorbar(self._im12, ax=self.ax12, fraction=0.046, pad=0.04)
        else:
            self._im12.set_data(rd_power_db)
            self._im12.set_clim(vmin=float(np.min(rd_power_db)), vmax=float(np.max(rd_power_db)))

    def __plot_21(self, raw_signal: np.ndarray) -> None:
        """(2,1) DPC + CFAR detections in dB"""
        # 1. Digital Pulse Compression
        compressed_signal = dpc_processing(raw_signal)

        # 2. Average power over PRI (axis=1 as requested) and convert to dB
        # Note: If shape is (npri, nrg), axis=1 averages across range.
        # Usually, for a range profile, we average over axis=0 (PRI).
        dpc_power = np.mean(np.abs(compressed_signal) ** 2, axis=0)
        dpc_db = 10 * np.log10(dpc_power + 1e-12)

        # 3. CFAR - cfar_1d returns (threshold_values, detection_mask)
        # Using your function: mask is the adaptive power threshold, det is boolean
        thresh_linear, detections = cfar_1d(
            dpc_power, guard_cells=4, training_cells=8, threshold=2.0
        )

        # Convert the adaptive threshold to dB for the red dashed line
        mask_db = 10 * np.log10(thresh_linear + 1e-12)

        x = np.arange(dpc_db.size)
        det_idx = np.where(detections)[0]

        if self._line21_dpc is None or self._line21_dpc.get_xdata().size != x.size:
            self.ax21.clear()

            # The Signal (DPC)
            (self._line21_dpc,) = self.ax21.plot(
                x, dpc_db, color="tab:blue", linewidth=1.2, label="DPC Power"
            )

            # The Adaptive Threshold (Red Dashed Line)
            (self._line21_cfar,) = self.ax21.plot(
                x, mask_db, color="red", linestyle="--", linewidth=1.0, label="CFAR Threshold"
            )

            # The Detections (Points on the DPC line)
            self._sc21 = self.ax21.scatter(
                det_idx, dpc_db[det_idx], s=25, c="red", marker="o",
                edgecolors="white", linewidths=0.5, label="Detections", zorder=3
            )

            self.ax21.set_title("1D DPC & CFAR (dB)", fontsize=8)
            self.ax21.set_xlabel("Bin Index", fontsize=8)
            self.ax21.set_ylabel("Power (dB)", fontsize=8)
            self.ax21.legend(loc="upper right", fontsize=6)
            self.ax21.grid(True, alpha=0.3)
        else:
            # Update existing plot data
            self._line21_dpc.set_data(x, dpc_db)
            self._line21_cfar.set_data(x, mask_db)

            # Update detection scatter points
            if det_idx.size > 0:
                offsets = np.column_stack((det_idx, dpc_db[det_idx]))
                self._sc21.set_offsets(offsets)
                self._sc21.set_visible(True)
            else:
                self._sc21.set_visible(False)

            self.ax21.relim()
            self.ax21.autoscale_view()

    def __plot_22(self, raw_signal: np.ndarray) -> None:
        """(2,2) Integrated Doppler Spectrum (Average over Range Gates)"""
        # 1. Compute Range-Doppler Map (2D FFT)
        # Applying FFT shift to center 0-Doppler
        # We FFT over axis 0 (pulses) and axis 1 (range)
        rd = np.fft.fftshift(np.fft.fft2(raw_signal), axes=(0, 1))
        rd_power = np.abs(rd) ** 2

        # 2. Average over the Range Axis (axis 1) to get the Doppler Profile
        # This shows the total energy at each velocity bin
        doppler_profile_linear = np.mean(rd_power, axis=1)
        doppler_db = 10 * np.log10(doppler_profile_linear + 1e-12)

        # 3. Setup X-axis (Doppler Bins)
        num_doppler_bins = doppler_db.size
        x = np.arange(num_doppler_bins) - (num_doppler_bins // 2)  # Centered at 0

        if self._im22 is None or self._line21_dpc is None:  # Initializing artist
            # Reusing _im22 name or creating a new line reference
            if not hasattr(self, "_line22"):
                self.ax22.clear()
                (self._line22,) = self.ax22.plot(
                    x, doppler_db, color="tab:purple", linewidth=1.2, label="Mean Doppler"
                )
                self.ax22.set_title("Integrated Doppler Spectrum (dB)", fontsize=8)
                self.ax22.set_xlabel("Doppler Bin (centered)", fontsize=8)
                self.ax22.set_ylabel("Power (dB)", fontsize=8)
                self.ax22.grid(True, alpha=0.3)
                self.ax22.legend(loc="upper right", fontsize=6)
        else:
            # Fast update
            self._line22.set_data(x, doppler_db)
            self.ax22.relim()
            self.ax22.autoscale_view()