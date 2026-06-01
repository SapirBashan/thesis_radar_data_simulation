import time
from typing import List

from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from config.radar_config import RadarConfig
from config.submode_config import SubmodeLibrary
from config.operating_submode import OperatingSubmode
from config.target import Target

from radar_signal.generator import RadarSimulator

from psp import psp_process, CFARParams, filter_target_submode, dbf_bfcu_mode, dpc, cfar_1d

C = 299792458.0


def db(x, eps=1e-12):
    return 10 * np.log10(np.abs(x) ** 2 + eps)


def add_noise_snr(signal, snr_db, noise_floor=1e-12):
    P_sig = np.mean(np.abs(signal) ** 2)
    P_sig = max(P_sig, noise_floor)
    P_noise = P_sig / (10 ** (snr_db / 10.0))
    sigma = np.sqrt(P_noise)
    noise = (sigma / np.sqrt(2)) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


def matched_filter_rect(pulse_width, fs, nrg):
    n = int(round(pulse_width * fs))
    n = max(n, 1)
    h = np.zeros(nrg, dtype=np.float32)
    h[:n] = 1.0
    return h[::-1]


def apply_matched_filter(signal, pulse_width, fs):
    nrg = signal.shape[0]
    h = matched_filter_rect(pulse_width, fs, nrg)
    H = np.fft.fft(h, n=nrg)
    S = np.fft.fft(signal, axis=0)
    return np.fft.ifft(S * H[:, None], axis=0)


def range_doppler_map(signal):
    return np.fft.fftshift(np.fft.fft(signal, axis=1), axes=1)


def spherical_to_cart(range_m, az, el):
    x = range_m * np.cos(el) * np.cos(az)
    y = range_m * np.cos(el) * np.sin(az)
    z = range_m * np.sin(el)
    return x, y, z


def draw_beam_footprint(ax, max_range, az_range, el_range, color="cyan", alpha=0.08):
    az_min, az_max = az_range
    el_min, el_max = el_range
    origin = (0, 0, 0)

    p_bl = spherical_to_cart(max_range, az_min, el_min)
    p_br = spherical_to_cart(max_range, az_max, el_min)
    p_tl = spherical_to_cart(max_range, az_min, el_max)
    p_tr = spherical_to_cart(max_range, az_max, el_max)

    for p in [p_bl, p_br, p_tl, p_tr]:
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], c=color, alpha=0.6)

    az_vals = np.linspace(az_min, az_max, 30)
    el_vals = np.linspace(el_min, el_max, 30)

    x, y, z = spherical_to_cart(max_range, az_vals, el_max)
    ax.plot(x, y, z, c=color, alpha=0.6)
    x, y, z = spherical_to_cart(max_range, az_vals, el_min)
    ax.plot(x, y, z, c=color, alpha=0.6)

    x, y, z = spherical_to_cart(max_range, az_min, el_vals)
    ax.plot(x, y, z, c=color, alpha=0.6)
    x, y, z = spherical_to_cart(max_range, az_max, el_vals)
    ax.plot(x, y, z, c=color, alpha=0.6)

    faces = [
        [origin, p_bl, p_br],
        [origin, p_tl, p_tr],
        [origin, p_bl, p_tl],
        [origin, p_br, p_tr],
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=alpha, linewidths=0))


def _bounded_centers(min_deg, max_deg, width_deg, step_deg):
    # keep centers so the beam (width) stays inside radar limits
    half = width_deg / 2
    start = min_deg + half
    stop = max_deg - half
    if stop < start:
        return np.array([])  # no valid centers
    return np.arange(start, stop + 1e-9, step_deg)


def build_coarse_grid(radar_cfg, wide_sm):
    az_min, az_max = radar_cfg.azimuth_range
    el_min, el_max = radar_cfg.elevation_range
    az_step = wide_sm.azimuth_width
    el_step = wide_sm.elevation_width

    az_centers = _bounded_centers(az_min, az_max, wide_sm.azimuth_width, az_step)
    el_centers = _bounded_centers(el_min, el_max, wide_sm.elevation_width, el_step)

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


def generate_targets(radar_cfg, n=80, r_min=2000.0, r_max=15000.0):
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
        # generate random velocity
        vx = np.random.uniform(-10.0, 10.0)
        vy = np.random.uniform(-10.0, 10.0)
        vz = np.random.uniform(-10.0, 10.0)
        targets.append(Target.new([x, y, z], [vx, vy, vz], rcs))
    return targets

def update_target_positions(targets: List[Target], dt: float):
    for t in targets:
        x, y, z = t.position_xyz
        vx, vy, vz = t.velocity_xyz
        x += dt * vx
        y += dt * vy
        z += dt * vz
        t.position_xyz = [x, y, z]

def build_scan_steps(radar, submodes, fine_prob=0.1, track_prob=0.05, confirm_prob=0.02, repeat = 1, seed=None):
    """
    Build scan steps with randomized fine/track/confirm inserts.
    fine_prob: chance to add point_search for a coarse look
    track_prob / confirm_prob: chance to add track/confirm after a coarse look
    """
    rng = np.random.default_rng(seed)
    wide_sm = submodes.submodes["coarse_search"]

    steps = []

    for _ in range(repeat):
        for (azc, elc) in build_coarse_grid(radar, wide_sm):
            # always add the coarse step
            steps.append(("coarse_search", azc, elc, "coarse"))

            # sometimes add fine (point_search)
            if rng.random() < fine_prob:
                for (azf, elf) in build_fine_grid(azc, elc, wide_sm):
                    steps.append(("fine_search", azf, elf, "fine"))

            # sprinkle track / confirm
            if rng.random() < track_prob:
                steps.append(("track", azc, elc, "track"))

            if rng.random() < confirm_prob:
                steps.append(("confirm", azc, elc, "confirm"))

    return steps

def detections_to_xyz(detections):
    pts = []
    for det in detections:
        az = np.deg2rad(det.azimuth)
        el = np.deg2rad(det.elevation)
        x, y, z = spherical_to_cart(det.range, az, el)
        pts.append([x, y, z])
    return np.array(pts) if pts else np.empty((0, 3))

def main():
    radar = RadarConfig.load("configs/radar_config.json")
    submodes = SubmodeLibrary.load("configs/submodes.json")
    sim = RadarSimulator(radar, submodes)

    viz_mode = "targets"  # or "detections"
    det_history = deque(maxlen=20)

    wide_sm = submodes.submodes["coarse_search"]

    r_all = list(
        sm.num_range_gates * C / (2 * sm.fs)
        for sm in submodes.submodes.values()
    )
    max_r_all = max(r_all)
    min_r_all = min(r_all)

    steps = build_scan_steps(
        radar,
        submodes,
        fine_prob=0.1,
        track_prob=0.1,
        confirm_prob=0.10,
        repeat=1,
        seed=42
    )

    cfar_params = CFARParams(
        guard_cells=12,
        training_cells=8,
        threshold=1.5
    )


    targets = generate_targets(radar, n=10, r_min=2000.0, r_max=min_r_all)

    # Figure layout
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Radar Scan Demo", fontsize=14)
    subtitle = fig.text(0.5, 0.94, "", ha="center", va="top", fontsize=10)

    gs = GridSpec(2, 2, figure=fig)

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_rd = fig.add_subplot(gs[0, 1])
    ax_3d = fig.add_subplot(gs[1, 0], projection="3d")
    ax_avg = fig.add_subplot(gs[1, 1])

    img_raw = ax_raw.imshow(np.zeros((wide_sm.num_pulses, wide_sm.num_range_gates)),
                            origin="lower", aspect="auto", cmap="viridis")
    img_rd = ax_rd.imshow(np.zeros((wide_sm.num_pulses, wide_sm.num_range_gates)),
                          origin="lower", aspect="auto", cmap="viridis")
    # img_avg = ax_avg.plot(np.zeros(wide_sm.num_range_gates))[0]
    line_dpc, = ax_avg.plot([], [], "b-", label="DPC power")
    line_thr, = ax_avg.plot([], [], "r--", label="CFAR threshold")
    det_scatter = ax_avg.scatter([], [], c="r", s=35, label="Detections")

    ax_avg.set_title("DPC power with CFAR")
    ax_avg.grid(True, alpha=0.3)
    ax_avg.legend(loc="upper right")

    ax_raw.set_title("Raw power (DBF sum)")
    ax_rd.set_title("Range-Doppler map")
    ax_avg.set_title("Matched filter, avg over PRI")
    ax_avg.grid(True, alpha=0.3)

    dpc_ylim_max = 1.0
    dpc_alpha = 0.1  # smoothing factor (lower = smoother)

    time_elapsed = 0.0

    def update(i):
        nonlocal dpc_ylim_max
        nonlocal time_elapsed


        mode, az, el, level = steps[i]
        sm = submodes.submodes[mode]
        op = OperatingSubmode.from_values(mode, az, el)

        # time is in ms
        update_target_positions(targets, dt=sm.time / 1000.0)

        time_elapsed += sm.time

        target_xyz = np.array([t.position_xyz for t in targets])

        sig = sim.generate(targets, op, noise_sigma=0.0)
        sig = add_noise_snr(sig, snr_db=20, noise_floor=1e-12)

        filtered_targets = filter_target_submode(targets, sm, op)

        detections = psp_process(sig, radar, submodes, op, cfar_params)

        det_history.append(detections)

        all_detections = [d for frame in det_history for d in frame]
        detections_xyz = detections_to_xyz(all_detections)

        print(f"Detections/Targets: {len(detections)}/{len(filtered_targets)}")

        if viz_mode == "targets":
            point_xyz = target_xyz
        else:
            point_xyz = detections_xyz

        beam = sig.sum(axis=0)

        raw_db = db(beam)
        img_raw.set_data(raw_db.T)
        img_raw.set_extent([0, sm.num_range_gates - 1, 0, sm.num_pulses - 1])
        lo, hi = np.percentile(raw_db, [5, 99])
        img_raw.set_clim(lo, hi)

        rd = range_doppler_map(beam)
        rd_db = db(rd)
        img_rd.set_data(rd_db.T)
        img_rd.set_extent([0, sm.num_range_gates - 1, -sm.num_pulses // 2, sm.num_pulses // 2 - 1])
        lo, hi = np.percentile(rd_db, [5, 99])
        img_rd.set_clim(lo, hi)

        # PSP DPC
        x_dbf = dbf_bfcu_mode(sig)  # (n_rg, n_pri)
        x_dpc = dpc(x_dbf[:, :], pulse_width=int(sm.fs * sm.pulse_width))[:, :]
        dpc_pow = np.mean(np.abs(x_dpc) ** 2, axis=1)

        noise_rg, det_rg = cfar_1d(dpc_pow[:, None], cfar_params.guard_cells, cfar_params.training_cells,
                                   cfar_params.threshold)
        noise_rg = noise_rg[:, 0]
        det_rg = det_rg[:, 0]
        thr = noise_rg

        x = np.arange(sm.num_range_gates)

        line_dpc.set_data(x, dpc_pow)
        line_thr.set_data(x, thr)

        det_idx = np.where(det_rg)[0]
        det_y = dpc_pow[det_idx]

        # recreate scatter each frame
        for coll in list(ax_avg.collections):
            coll.remove()
        det_scatter = ax_avg.scatter(det_idx, det_y, c="r", s=35)

        ax_avg.relim()
        ax_avg.autoscale_view(scalex=False, scaley=True)
        ax_avg.set_xlim(0, sm.num_range_gates - 1)

        # let matplotlib autoscale y each frame
        ax_avg.relim()
        ax_avg.autoscale_view(scalex=False, scaley=True)
        ax_avg.set_xlim(0, sm.num_range_gates - 1)

        # 3D beam
        ax_3d.clear()

        ax_3d.set_xlim(-max_r_all, max_r_all)
        ax_3d.set_ylim(-max_r_all, max_r_all)
        ax_3d.set_zlim(0, max_r_all)
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")

        ticks_xy = np.linspace(-max_r_all, max_r_all, 7)
        ticks_z = np.linspace(0, max_r_all, 6)
        ax_3d.set_xticks(ticks_xy)
        ax_3d.set_yticks(ticks_xy)
        ax_3d.set_zticks(ticks_z)
        ax_3d.set_xticklabels([])
        ax_3d.set_yticklabels([])
        ax_3d.set_zticklabels([])

        ax_3d.grid(True, alpha=0.3)
        ax_3d.view_init(elev=20., azim=-35)

        ax_3d.xaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.4)
        ax_3d.yaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.4)
        ax_3d.zaxis._axinfo["grid"]["color"] = (0.7, 0.7, 0.7, 0.4)

        ax_3d.scatter([0], [0], [0], c="blue", s=60, marker="*", label="Radar")

        if point_xyz.size > 0:
            ax_3d.scatter(point_xyz[:, 0], point_xyz[:, 1], point_xyz[:, 2],
                          c="red", s=20, alpha=0.9)

        az_range = (np.deg2rad(az - sm.azimuth_width / 2),
                    np.deg2rad(az + sm.azimuth_width / 2))
        el_range = (np.deg2rad(el - sm.elevation_width / 2),
                    np.deg2rad(el + sm.elevation_width / 2))

        draw_beam_footprint(ax_3d, max_r_all, az_range, el_range)

        subtitle.set_text(
            f"Mode: {mode} | Grid: {level} | Look: az={az:.1f}°, el={el:.1f}° | Step {i + 1}/{len(steps)} | Time {time_elapsed} ms"
        )

        ax_3d.view_init(elev=15, azim=-60)

        return [img_raw, img_rd, line_dpc, line_thr, det_scatter, subtitle]

    ani = FuncAnimation(fig, update, frames=len(steps), interval=1000, blit=False)
    plt.show()


if __name__ == "__main__":
    main()