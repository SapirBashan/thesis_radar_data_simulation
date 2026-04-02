import numpy as np
import matplotlib.pyplot as plt

from radar_signal_simulator import RadarConfig, Target, RadarSignalGenerator, C

def main():
    np.random.seed(42)

    cfg = RadarConfig.load("radar_config.json")
    targets = Target.load_list("targets.json")

    gen = RadarSignalGenerator(cfg)
    signal = gen.generate(targets)  # (num_elems, num_pulses, num_range_gates)

    # Raw data power plots: PRI vs Range Gate for each element
    num_elems = cfg.array.rows * cfg.array.cols
    power = 20 * np.log10(np.abs(signal) + 1e-12)  # (num_elems, num_pulses, num_range_gates)


    rows, cols = cfg.array.rows, cfg.array.cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            ax = axes[r, c]
            ax.imshow(
                power[idx],
                aspect="auto",
                origin="lower",
                # extent=[ranges[0], ranges[-1], pri_idx[0], pri_idx[-1]],
                cmap="viridis"
            )
            ax.set_title(f"Elem ({r},{c})")

    fig.suptitle("Raw Data Power: PRI vs Range")
    fig.text(0.5, 0.04, "Range Gate Index", ha="center")
    fig.text(0.04, 0.5, "PRI Index", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()

    # Beamform (sum)
    data = np.sum(signal, axis=0)  # (num_pulses, num_range_gates)

    #Range matched filter (rect pulse)
    pulse_samples = int(np.round(cfg.pulse_width * cfg.fs))
    mf = np.ones(pulse_samples, dtype=np.complex64)  # matched filter for rect pulse
    mf = mf[::-1].conj()

    # Convolve along fast-time for each pulse
    data_mf = np.array([
        np.convolve(data[p, :], mf, mode="same")
        for p in range(cfg.num_pulses)
    ])

    # Doppler FFT only (slow-time)
    win_p = np.hanning(cfg.num_pulses)[:, None]
    rd = np.fft.fftshift(np.fft.fft(data_mf * win_p, axis=0), axes=0)
    rd_mag = 20 * np.log10(np.abs(rd) + 1e-12)

    # Axes
    ranges = np.arange(cfg.num_range_gates) * (C / (2 * cfg.fs))
    doppler_bins = np.arange(cfg.num_pulses) - cfg.num_pulses // 2
    doppler_freq = doppler_bins * (cfg.prf / cfg.num_pulses)

    # Plot
    plt.figure(figsize=(10, 6))
    extent = [ranges[0], ranges[-1], doppler_freq[0], doppler_freq[-1]]
    plt.imshow(rd_mag, aspect="auto", extent=extent, origin="lower", cmap="viridis")
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Range (m)")
    plt.ylabel("Doppler (Hz)")
    plt.title("Range-Doppler Map")
    plt.tight_layout()
    plt.show()

    # plot an average over pri
    range_bins = np.arange(cfg.num_range_gates) * (C / (2 * cfg.fs))

    avg_pri = np.abs(np.mean(data_mf, axis=0)) ** 2

    plt.figure(figsize=(10, 6))
    plt.plot(range_bins, avg_pri)
    plt.xlabel("Range (m)")
    plt.ylabel("Power")
    plt.title("Average over PRI")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()