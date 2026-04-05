from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

class OnlineMonitor:
    """
    Data-driven online monitor for radar processing snapshots.
    Receives raw NumPy arrays directly from an external data source.
    """
    def __init__(self, show_every: int = 1):
        self.show_every = max(1, int(show_every))
        self.step = 0

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Online Signal Processing Monitor", fontsize=10, y=0.985)

        self.fig.subplots_adjust(left=0.07, right=0.95, bottom=0.08, top=0.90, wspace=0.38, hspace=0.42)

        self.ax11, self.ax12 = self.axes[0, 0], self.axes[0, 1]
        self.ax21, self.ax22 = self.axes[1, 0], self.axes[1, 1]

        # Persistent artists
        self._im11 = None; self._cb11 = None
        self._line12 = None
        self._line21 = None; self._cb21 = None
        self._im22 = None; self._cb22 = None; self._sc22 = None

        self._meta_text = self.fig.text(
            0.5, 0.952,
            "submode: N/A | data: N/A MB | current: N/A GB/s | time: N/A ms | running: N/A GB/s",
            ha="center", va="center", fontsize=8,
        )

    # --- (Keep __safe_remove_colorbar exactly as you had it) ---
    @staticmethod
    def __safe_remove_colorbar(cb):
        if cb is None: return None
        try: cb.remove()
        except Exception:
            ax = getattr(cb, "ax", None)
            if ax:
                try: ax.remove()
                except Exception: pass
        return None

    def update(
        self,
        raw_signal: np.ndarray,
        dpc_avg: np.ndarray,
        fft_signal: np.ndarray,
        cfar_mask: np.ndarray,
        submode_stat: dict | None = None,
        step: int | None = None
    ) -> None:
        """Called by the PipeMonitor when a full frame of data is ready."""
        self.step = self.step + 1 if step is None else int(step)

        if self.step % self.show_every != 0:
            return

        self.__plot_11(raw_signal)
        self.__plot_12(dpc_avg)
        self.__plot_21(fft_signal)
        self.__plot_22(raw_signal, cfar_mask)
        self.__update_submode_text(submode_stat)

        # Formatting ticks
        for ax in (self.ax11, self.ax12, self.ax21, self.ax22):
            ax.tick_params(axis="both", which="major", labelsize=6)
        if self._cb11: self._cb11.ax.tick_params(labelsize=6)
        if self._cb22: self._cb22.ax.tick_params(labelsize=6)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)

    def __update_submode_text(self, stat: dict[str, float | str | int] | None) -> None:
        if not stat:
            self._meta_text.set_text(
                "submode: N/A | data: N/A MB | current: N/A GB/s | time: N/A ms | running: N/A GB/s"
            )
            return

        amount_mb = float(stat["amount_bytes"]) / 1e6
        self._meta_text.set_text(
            "submode: "
            f"{stat['submode_name']} | "
            f"data: {amount_mb:.3f} MB | "
            f"current: {float(stat['throughput_gbps']):.3f} GB/s | "
            f"time: {float(stat['elapsed_ms']):.3f} ms | "
            f"running: {float(stat['running_avg_gbps']):.3f} GB/s"
        )

    def __plot_11(self, raw_signal: np.ndarray) -> None:
        # Raw signal heatmap: fast-time/range-gate vs slow-time/PRI
        z = np.abs(raw_signal)

        if self._im11 is None or self._im11.get_array().shape != z.shape:
            self._cb11 = self.__safe_remove_colorbar(self._cb11)

            self.ax11.clear()
            self._im11 = self.ax11.imshow(
                z,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="viridis",
            )
            self.ax11.set_title("Raw Signal |z|", fontsize=8)
            self.ax11.set_xlabel("Range gate", fontsize=8)
            self.ax11.set_ylabel("PRI (slow-time)", fontsize=8)
            self._cb11 = self.fig.colorbar(self._im11, ax=self.ax11, fraction=0.046, pad=0.04)
        else:
            self._im11.set_data(z)
            self._im11.set_clim(vmin=float(np.min(z)), vmax=float(np.max(z)))

    def __plot_12(self, dpc_avg: np.ndarray) -> None:
        x = np.arange(dpc_avg.size)

        if self._line12 is None or self._line12.get_xdata().size != x.size:
            self.ax12.clear()
            (self._line12,) = self.ax12.plot(x, dpc_avg, color="tab:orange", linewidth=1.2)
            self.ax12.set_title("DPC Avg Over Slow-Time", fontsize=8)
            self.ax12.set_xlabel("Range gate", fontsize=8)
            self.ax12.set_ylabel("Amplitude", fontsize=8)
            self.ax12.grid(True, alpha=0.3)
        else:
            self._line12.set_data(x, dpc_avg)
            self.ax12.relim()
            self.ax12.autoscale_view()

    def __plot_21(self, fft_signal: np.ndarray) -> None:
        # Average FFT power over PRI (axis=0) -> 1D range profile.
        avg_power = np.mean(np.abs(fft_signal) ** 2, axis=0)
        x = np.arange(avg_power.size)

        if self._line21 is None or self._line21.get_xdata().size != x.size:
            self._cb21 = self.__safe_remove_colorbar(self._cb21)

            self.ax21.clear()
            (self._line21,) = self.ax21.plot(
                x,
                avg_power,
                color="tab:green",
                linewidth=1.2,
            )
            self.ax21.set_title("FFT Avg Power Over PRI", fontsize=8)
            self.ax21.set_xlabel("Range gate", fontsize=8)
            self.ax21.set_ylabel("Avg power", fontsize=8)
            self.ax21.grid(True, alpha=0.3)
        else:
            self._line21.set_data(x, avg_power)
            self.ax21.relim()
            self.ax21.autoscale_view()

    def __plot_22(self, raw_signal: np.ndarray, cfar_mask: np.ndarray) -> None:
        # Range-Doppler power map from raw data.
        rd = np.fft.fftshift(np.fft.fft(raw_signal, axis=0), axes=0)
        power = np.abs(rd) ** 2  # shape: (npri, nrg)

        _, nrg = power.shape
        n = min(nrg, cfar_mask.size)
        det_ranges = np.where(cfar_mask[:n] > 0.5)[0]

        # For each detected range bin, place one marker at its strongest Doppler bin.
        if det_ranges.size > 0:
            peak_doppler = np.argmax(power[:, det_ranges], axis=0)
        else:
            peak_doppler = np.array([], dtype=np.int64)

        if self._im22 is None or self._im22.get_array().shape != power.shape:
            self._cb22 = self.__safe_remove_colorbar(self._cb22)

            self.ax22.clear()
            self._im22 = self.ax22.imshow(
                power,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap="plasma",
            )
            self.ax22.set_title("Range-Doppler Map + CFAR(range) overlay", fontsize=8)
            self.ax22.set_xlabel("Range gate", fontsize=8)
            self.ax22.set_ylabel("Doppler bin", fontsize=8)

            self._cb22 = self.fig.colorbar(self._im22, ax=self.ax22, fraction=0.046, pad=0.04)

            self._sc22 = self.ax22.scatter(
                det_ranges,
                peak_doppler,
                s=14,
                c="red",
                marker="o",
                linewidths=0.5,
                edgecolors="white",
                label="CFAR detections",
            )
            self.ax22.legend(loc="best")
        else:
            self._im22.set_data(power)
            self._im22.set_clim(vmin=float(np.min(power)), vmax=float(np.max(power)))

            # Update marker positions.
            offsets = (
                np.column_stack((det_ranges, peak_doppler))
                if det_ranges.size > 0
                else np.empty((0, 2))
            )
            if self._sc22 is not None:
                self._sc22.set_offsets(offsets)