import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from config.radar_config import RadarConfig
from config.submode_config import SubmodeLibrary
from config.operating_submode import OperatingSubmode
from config.target import Target

from radar_signal.generator import RadarSimulator


def build_coarse_grid(radar_cfg, wide_sm):
    az_min, az_max = radar_cfg.azimuth_range
    el_min, el_max = radar_cfg.elevation_range

    az_step = wide_sm.azimuth_width
    el_step = wide_sm.elevation_width

    az_centers = np.arange(az_min + az_step / 2, az_max, az_step)
    el_centers = np.arange(el_min + el_step / 2, el_max, el_step)

    return [(az, el) for el in el_centers for az in az_centers]


def build_fine_grid(center_az, center_el, wide_sm):
    az_span = wide_sm.azimuth_width
    el_span = wide_sm.elevation_width

    az_step = az_span / 4.0
    el_step = el_span / 4.0

    az_start = center_az - az_span / 2 + az_step / 2
    el_start = center_el - el_span / 2 + el_step / 2

    az_centers = [az_start + i * az_step for i in range(4)]
    el_centers = [el_start + i * el_step for i in range(4)]

    return [(az, el) for el in el_centers for az in az_centers]


def generate_targets(radar_cfg, n=50, r_min=2000.0, r_max=15000.0):
    az_min, az_max = radar_cfg.azimuth_range
    el_min, el_max = radar_cfg.elevation_range

    targets = []
    for _ in range(n):
        r = np.random.uniform(r_min, r_max)
        az = np.deg2rad(np.random.uniform(az_min, az_max))
        el = np.deg2rad(np.random.uniform(el_min, el_max))

        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        rcs = np.random.uniform(0.5, 5.0)
        targets.append(Target.new([x, y, z], [0.0, 0.0, 0.0], rcs))
    return targets

def add_noise_snr(signal, snr_db, noise_floor=1e-12):
    """
    Adds complex Gaussian noise to reach desired SNR.
    If signal power is ~0, uses noise_floor instead.
    """
    P_sig = np.mean(np.abs(signal)**2)
    P_sig = max(P_sig, noise_floor)   # fallback when no targets

    P_noise = P_sig / (10**(snr_db / 10.0))
    sigma = np.sqrt(P_noise)

    noise = (sigma / np.sqrt(2)) * (
        np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape)
    )
    return signal + noise

def main():
    radar = RadarConfig.load("configs/radar_config.json")
    submodes = SubmodeLibrary.load("configs/submodes.json")

    sim = RadarSimulator(radar, submodes)

    wide_sm = submodes.submodes["wide_search"]

    steps = []
    coarse_grid = build_coarse_grid(radar, wide_sm)

    for (azc, elc) in coarse_grid:
        steps.append(("wide_search", azc, elc, "coarse"))
        fine_grid = build_fine_grid(azc, elc, wide_sm)
        for (azf, elf) in fine_grid:
            steps.append(("point_search", azf, elf, "fine"))

    # Random targets inside radar FOV
    targets = generate_targets(radar, n=80)

    # Initialize plot with wide_search size
    img = None
    fig, ax = plt.subplots(figsize=(12, 8))
    img = ax.imshow(
        np.zeros((wide_sm.num_pulses, wide_sm.num_range_gates)),
        origin="lower",
        aspect="auto",
        cmap="viridis"
    )
    fig.supxlabel("Range gate")
    fig.supylabel("PRI index")
    title = ax.set_title("")

    info_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
        color="white", fontsize=10
    )

    def update(i):
        mode, az, el, level = steps[i]
        sm = submodes.submodes[mode]
        op = OperatingSubmode.from_values(mode, az, el)

        sig = sim.generate(targets, op, noise_sigma=0.0)
        sig = add_noise_snr(sig, snr_db=15, noise_floor=1e-15)

        beam = sig.sum(axis=0)  # DBF sum
        power = np.abs(beam) ** 2
        power_db = 10 * np.log10(power + 1e-12)

        # update image data
        img.set_data(power_db.T)
        img.set_extent([0, sm.num_range_gates - 1, 0, sm.num_pulses - 1])
        img.set_clim(np.percentile(power_db, 5), np.percentile(power_db, 95))

        title.set_text(f"{mode} | step {i+1}/{len(steps)}")
        info_text.set_text(
            f"Mode: {mode}\n"
            f"Grid: {level}\n"
            f"Look: az={az:.1f}°, el={el:.1f}°"
        )
        return [img, title, info_text]

    ani = FuncAnimation(fig, update, frames=len(steps), interval=200, blit=False)
    plt.show()


if __name__ == "__main__":
    main()