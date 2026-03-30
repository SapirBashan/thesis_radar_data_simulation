from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class OnlineMonitor:
    """
    File-driven online monitor for radar processing snapshots.

    Expected files in output_dir:
      - raw_signal.txt  : complex 2D, header "nrg npri", then re im per line
      - dpc_avg.txt     : float 1D, header "nrg", then one float per line
      - fft_signal.txt  : complex 2D, header "nrg npri", then re im per line
      - cfar_mask.txt   : float 1D, header "nrg", then one float per line (0/1 or score)
    """

    def __init__(self, output_dir: str = "output_signal", show_every: int = 10):
        self.output_dir = Path(output_dir)
        self.show_every = max(1, int(show_every))
        self.step = 0

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Online Signal Processing Monitor", fontsize=10, y=0.985)

        # Extra spacing helps reduce clutter with titles/labels/colorbars.
        self.fig.subplots_adjust(
            left=0.07,
            right=0.95,
            bottom=0.08,
            top=0.90,
            wspace=0.38,
            hspace=0.42,
        )

        self.ax11 = self.axes[0, 0]
        self.ax12 = self.axes[0, 1]
        self.ax21 = self.axes[1, 0]
        self.ax22 = self.axes[1, 1]

        # Persistent artists to avoid full redraw overhead each update.
        self._im11 = None
        self._cb11 = None

        self._line12 = None

        self._line21 = None
        self._cb21 = None  # Kept for safe cleanup if old state exists.

        self._im22 = None
        self._cb22 = None
        self._sc22 = None

        self._last_error = None
        self._last_meta_error = None

        self._meta_text = self.fig.text(
            0.5,
            0.952,
            "submode: N/A | data: N/A MB | current: N/A GB/s | time: N/A ms | running: N/A GB/s",
            ha="center",
            va="center",
            fontsize=8,
        )

    @staticmethod
    def __safe_remove_colorbar(cb):
        if cb is None:
            return None
        try:
            cb.remove()
        except Exception:
            # Fallback for rare matplotlib state issues.
            ax = getattr(cb, "ax", None)
            if ax is not None:
                try:
                    ax.remove()
                except Exception:
                    pass
        return None

    def update(self, step: int | None = None) -> None:
        if step is None:
            self.step += 1
        else:
            self.step = int(step)

        if self.step % self.show_every != 0:
            return

        raw_path = self.output_dir / "raw_signal.txt"
        dpc_path = self.output_dir / "dpc_avg.txt"
        fft_path = self.output_dir / "fft_signal.txt"
        cfar_path = self.output_dir / "cfar_mask.txt"
        submode_path = self.output_dir / "submode_stat.txt"

        try:
            raw_signal = self.__read_complex_2d(raw_path)  # shape: (npri, nrg)
            dpc_avg = self.__read_float_1d(dpc_path)  # shape: (nrg,)
            fft_signal = self.__read_complex_2d(fft_path)  # shape: (npri, nrg)
            cfar_mask = self.__read_float_1d(cfar_path)  # shape: (nrg,)
        except (FileNotFoundError, ValueError) as exc:
            # Print only if error changed to avoid spamming terminal in fast loops.
            msg = f"[OnlineMonitor] Waiting for valid snapshot files: {exc}"
            if msg != self._last_error:
                print(msg)
                self._last_error = msg
            return

        self._last_error = None

        submode_stat = None
        try:
            submode_stat = self.__read_submode_stat(submode_path)
            self._last_meta_error = None
        except (FileNotFoundError, ValueError) as exc:
            msg = f"[OnlineMonitor] submode stats unavailable: {exc}"
            if msg != self._last_meta_error:
                print(msg)
                self._last_meta_error = msg

        self.__plot_11(raw_signal)
        self.__plot_12(dpc_avg)
        self.__plot_21(fft_signal)
        self.__plot_22(raw_signal, cfar_mask)
        self.__update_submode_text(submode_stat)

        # Reduce tick-label font size across all subplots.
        for ax in (self.ax11, self.ax12, self.ax21, self.ax22):
            ax.tick_params(axis="both", which="major", labelsize=6)

        # Reduce colorbar tick-label size too.
        if self._cb11 is not None:
            self._cb11.ax.tick_params(labelsize=6)
        if self._cb22 is not None:
            self._cb22.ax.tick_params(labelsize=6)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    @staticmethod
    def __read_complex_2d(path: Path) -> np.ndarray:
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split()
            if len(header) != 2:
                raise ValueError(f"{path}: first line must be 'nrg npri'")
            nrg, npri = int(header[0]), int(header[1])
            total = nrg * npri

            vals = np.empty(total, dtype=np.complex128)
            idx = 0
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) != 2:
                    raise ValueError(f"{path}: each sample line must be 're im'")
                if idx >= total:
                    raise ValueError(f"{path}: more samples than expected ({total})")
                vals[idx] = complex(float(parts[0]), float(parts[1]))
                idx += 1

        if idx != total:
            raise ValueError(f"{path}: expected {total} samples, got {idx}")

        # x -> range gate (nrg), y -> PRI/slow-time (npri)
        return vals.reshape(npri, nrg)

    @staticmethod
    def __read_float_1d(path: Path) -> np.ndarray:
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip()
            if not header:
                raise ValueError(f"{path}: missing length header line")
            n = int(header)

            vals = np.empty(n, dtype=np.float64)
            idx = 0
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if idx >= n:
                    raise ValueError(f"{path}: more values than expected ({n})")
                vals[idx] = float(s)
                idx += 1

        if idx != n:
            raise ValueError(f"{path}: expected {n} values, got {idx}")

        return vals

    @staticmethod
    def __read_submode_stat(path: Path) -> dict[str, float | str | int]:
        with path.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 5:
            raise ValueError(f"{path}: expected at least 5 non-empty lines")

        submode_name = lines[0]
        amount_bytes = int(lines[1])
        throughput_gbps = float(lines[2])
        elapsed_ms = float(lines[3])
        running_avg_gbps = float(lines[4])

        return {
            "submode_name": submode_name,
            "amount_bytes": amount_bytes,
            "throughput_gbps": throughput_gbps,
            "elapsed_ms": elapsed_ms,
            "running_avg_gbps": running_avg_gbps,
        }

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

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)
