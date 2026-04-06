"""
Radar debug display — 3 panels:

  Top-left  : Range-Doppler map (dB)
  Top-right : DPC power + CFAR threshold + detection markers
  Bottom    : PPI — true targets, detections, tracks

Requires:
- tracker.radar_manager.RadarManager
- tracker.radar_manager.StepResult
- tracker.track.TrackState
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ""))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from config.radar_config import RadarConfig
from config.submode_config import SubmodeLibrary
from config.target import Target
from radar_signal.generator import RadarSimulator

from psp import psp_process, CFARParams
from psp.dbf import dbf_bfcu_mode
from psp.dpc import dpc
from psp.cfar import cfar_1d

from tracker.radar_manager import RadarManager, StepResult
from tracker.track import TrackState

C0 = 299_792_458.0
EPS = 1e-12


# ── helpers ───────────────────────────────────────────────────────────

def _db(x):
    return 10.0 * np.log10(np.abs(x) ** 2 + EPS)


def _add_noise(signal, snr_db=20.0):
    p = max(np.mean(np.abs(signal) ** 2), EPS)
    sigma = np.sqrt(p / (10 ** (snr_db / 10.0)))
    noise = (sigma / np.sqrt(2)) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


def _spherical_to_cart(range_m, az_rad, el_rad):
    x = range_m * np.cos(el_rad) * np.cos(az_rad)
    y = range_m * np.cos(el_rad) * np.sin(az_rad)
    z = range_m * np.sin(el_rad)
    return x, y, z


def _polar_to_ppi(range_m, az_deg, el_deg=0.0):
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)
    x, y, _ = _spherical_to_cart(range_m, az_rad, el_rad)
    return x, y


def _xyz_to_ppi(xyz):
    return float(xyz[0]), float(xyz[1])


def _detections_to_ppi(dets):
    xs, ys = [], []
    for d in dets:
        x, y = _polar_to_ppi(d.range, d.azimuth, d.elevation)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def _generate_targets(radar, n=40, r_min=3000.0, r_max=12000.0):
    az_min, az_max = radar.azimuth_range
    el_min, el_max = radar.elevation_range
    out = []
    for _ in range(n):
        r = np.random.uniform(r_min, r_max)
        az = np.deg2rad(np.random.uniform(az_min, az_max))
        el = np.deg2rad(np.random.uniform(el_min, el_max))
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        vx, vy, vz = np.random.uniform(-15, 15, 3)
        out.append(Target.new([x, y, z], [vx, vy, vz], np.random.uniform(0.5, 5.0)))
    return out


def _move_targets(targets, dt_s):
    for t in targets:
        x, y, z = t.position_xyz
        vx, vy, vz = t.velocity_xyz
        t.position_xyz = [x + vx * dt_s, y + vy * dt_s, z + vz * dt_s]


# ── colour palette ────────────────────────────────────────────────────

_CMAP = cm._colormaps["tab20"].resampled(20)
_track_color_cache: dict[str, tuple] = {}
_color_idx = 0


def _track_color(tid: str):
    global _color_idx
    if tid not in _track_color_cache:
        _track_color_cache[tid] = _CMAP(_color_idx % 20)
        _color_idx += 1
    return _track_color_cache[tid]


STATE_MARKER = {
    TrackState.CANDIDATE: "o",
    TrackState.CONFIRMING: "^",
    TrackState.TRACKED: "D",
    TrackState.LOST: "s",
    TrackState.DROPPED: "x",
}

BEAM_COLOR = {
    "coarse": "#4fc3f7",
    "fine": "#fff176",
    "confirm": "#ff8a65",
    "track": "#a5d6a7",
    "recovery": "#ce93d8",
}


# ── signal processing helpers ─────────────────────────────────────────

def _compute_rd_and_cfar(sig, sm, cfar_params, fc):
    """
    Returns
    -------
    rd_db       : (n_rg, n_pri) range-Doppler power in dB
    dpc_pow     : (n_rg,)       DPC power (summed over PRI)
    noise_floor : (n_rg,)       CFAR noise estimate
    det_mask    : (n_rg,) bool  CFAR detections
    range_axis  : (n_rg,) m     range per gate
    vel_axis    : (n_pri,) m/s  Doppler velocity axis
    """
    x_dbf = dbf_bfcu_mode(sig)   # (n_rg, n_pri)
    n_rg, n_pri = x_dbf.shape

    x_dbf_3d = x_dbf[np.newaxis, :]  # (1, n_rg, n_pri)

    x_dpc = dpc(x_dbf_3d, fs=sm.fs, pulse_width=sm.pulse_width)
    x_dpc = x_dpc.reshape(n_rg, n_pri)

    rd = np.fft.fftshift(np.fft.fft(x_dpc, axis=1), axes=1)
    rd_db = _db(rd)

    dpc_pow = np.sum(np.abs(x_dpc) ** 2, axis=1)
    dpc_pow_2d = dpc_pow[:, np.newaxis]

    noise_2d, det_2d = cfar_1d(
        dpc_pow_2d,
        cfar_params.guard_cells,
        cfar_params.training_cells,
        cfar_params.threshold,
    )

    noise_floor = noise_2d[:, 0]
    det_mask = det_2d[:, 0].astype(bool)

    range_axis = np.arange(n_rg) * C0 / (2.0 * sm.fs)

    prf = sm.prf
    lam = C0 / fc
    vel_axis = np.fft.fftshift(np.fft.fftfreq(n_pri, d=1.0 / prf)) * lam / 2.0

    return rd_db, dpc_pow, noise_floor, det_mask, range_axis, vel_axis


# ── main ───────────────────────────────────────────────────────────────

def main():
    radar = RadarConfig.load("configs/radar_config.json")
    submodes = SubmodeLibrary.load("configs/submodes.json")
    sim = RadarSimulator(radar, submodes)

    mgr = RadarManager(radar, submodes, max_misses=15, gate_m=500.0)

    targets = _generate_targets(radar, n=10, r_min=500.0, r_max=5000.0)

    max_r = max(
        sm.num_range_gates * C0 / (2 * sm.fs)
        for sm in submodes.submodes.values()
    )

    fig = plt.figure(figsize=(12, 8), facecolor="#0a0a1a")
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1, 1.4],
        hspace=0.35,
        wspace=0.30,
    )

    ax_rd = fig.add_subplot(gs[0, 0])
    ax_cfar = fig.add_subplot(gs[0, 1])
    ax_ppi = fig.add_subplot(gs[1, :])

    for ax in (ax_rd, ax_cfar, ax_ppi):
        ax.set_facecolor("#0d1117")
        for sp in ax.spines.values():
            sp.set_color("#1e3a5f")
        ax.tick_params(colors="#7a9abf", labelsize=7)
        ax.xaxis.label.set_color("#7a9abf")
        ax.yaxis.label.set_color("#7a9abf")
        ax.title.set_color("white")

    fig.suptitle("Radar Debug Display", color="white", fontsize=12)

    # ── Range-Doppler panel ───────────────────────────────────────────
    ax_rd.set_title("Range-Doppler map (dB)")
    ax_rd.set_xlabel("Range gate")
    ax_rd.set_ylabel("Doppler (m/s)")

    wide_sm = submodes.submodes["coarse_search"]
    img_rd = ax_rd.imshow(
        np.zeros((wide_sm.num_pulses, wide_sm.num_range_gates)),
        origin="lower",
        aspect="auto",
        cmap="inferno",
        interpolation="nearest",
    )
    cb_rd = fig.colorbar(img_rd, ax=ax_rd, fraction=0.046, pad=0.04)
    cb_rd.ax.tick_params(colors="#7a9abf", labelsize=6)

    # ── CFAR panel ────────────────────────────────────────────────────
    ax_cfar.set_title("DPC power + CFAR threshold")
    ax_cfar.set_xlabel("Range (km)")
    ax_cfar.set_ylabel("Power (linear)")
    ax_cfar.grid(True, color="#1e3a5f", lw=0.5)

    line_pow, = ax_cfar.plot([], [], color="#4fc3f7", lw=0.9, label="DPC power")
    line_thr, = ax_cfar.plot([], [], color="#ff8a65", lw=0.9, ls="--", label="CFAR thr")
    line_dets = ax_cfar.scatter([], [], c="yellow", s=25, zorder=5, label="Detections")
    ax_cfar.legend(fontsize=7, facecolor="#0d1117", edgecolor="#1e3a5f",
                   labelcolor="white", loc="upper right")

    _cfar_tgt_lines: list = []

    # ── PPI panel ─────────────────────────────────────────────────────
    ax_ppi.set_aspect("equal")
    ax_ppi.set_xlim(-max_r, max_r)
    ax_ppi.set_ylim(-max_r, max_r)
    ax_ppi.axis("off")
    ax_ppi.set_title("PPI", pad=6)

    for r in np.linspace(max_r / 5, max_r, 5):
        ax_ppi.add_patch(plt.Circle((0, 0), r, color="#1e3a5f", fill=False, lw=0.7))
        lx, ly = _polar_to_ppi(r * 0.97, 45.0)
        ax_ppi.text(lx, ly, f"{r/1000:.0f} km", color="#4a7fa5", fontsize=6)

    ax_ppi.axhline(0, color="#1e3a5f", lw=0.4)
    ax_ppi.axvline(0, color="#1e3a5f", lw=0.4)
    ax_ppi.plot(0, 0, "*", color="#00e5ff", ms=10, zorder=10)

    ppi_status = ax_ppi.text(
        0.01, 0.99, "", transform=ax_ppi.transAxes,
        color="white", fontsize=7, va="top", family="monospace",
    )

    legend_items = [
        Line2D([0], [0], marker="P", color="white", ms=7, label="True target", linestyle="None"),
        Line2D([0], [0], marker="s", color="#4fc3f7", ms=5, label="Det (coarse)", linestyle="None"),
        Line2D([0], [0], marker="s", color="#fff176", ms=5, label="Det (fine)", linestyle="None"),
        Line2D([0], [0], marker="s", color="#ff8a65", ms=5, label="Det (confirm)", linestyle="None"),
        Line2D([0], [0], marker="s", color="#a5d6a7", ms=5, label="Det (track)", linestyle="None"),
        Line2D([0], [0], marker="o", color="white", ms=5, label="CANDIDATE", linestyle="None"),
        Line2D([0], [0], marker="^", color="white", ms=5, label="CONFIRMING", linestyle="None"),
        Line2D([0], [0], marker="D", color="white", ms=5, label="TRACKED", linestyle="None"),
        Line2D([0], [0], marker="s", color="white", ms=5, label="LOST", linestyle="None"),
    ]
    ax_ppi.legend(handles=legend_items, loc="lower right",
                  facecolor="#0d1117", edgecolor="#1e3a5f",
                  labelcolor="white", fontsize=6)

    ppi_det_sc = []
    ppi_tgt_sc = [None]
    ppi_sector = [None]

    def update(_frame):
        nonlocal ppi_det_sc, _cfar_tgt_lines

        beam = mgr.next_beam()
        if beam is None:
            return []

        sm = submodes.submodes[beam.submode_name]
        op = beam.op_submode

        _move_targets(targets, dt_s=sm.time / 1000.0)

        sig = sim.generate(targets, op, noise_sigma=0.0)
        cfar = CFARParams(**sm.cfar)
        dets = psp_process(sig, radar, submodes, op, cfar)

        result: StepResult = mgr.process(dets, beam, beam_time_ms=sm.time)

        print(f"t={result.sim_time_ms/1000:.2f}s dets={len(dets)}")

        rd_db, dpc_pow, noise_floor, det_mask, range_axis, vel_axis = _compute_rd_and_cfar(
            sig, sm, cfar, radar.fc
        )

        # ── Range-Doppler ──────────────────────────────────────────────
        img_rd.set_data(rd_db.T)
        img_rd.set_extent([0, sm.num_range_gates - 1, vel_axis[0], vel_axis[-1]])
        lo, hi = np.percentile(rd_db, [2, 99])
        img_rd.set_clim(lo, hi)
        ax_rd.set_ylabel("Doppler (m/s)" if np.max(np.abs(vel_axis)) > 10 else "Doppler bin")
        ax_rd.set_xlabel("Range gate")
        ax_rd.set_title(f"Range-Doppler [{beam.submode_name}] (dB)")

        # ── CFAR plot ─────────────────────────────────────────────────
        rng_km = range_axis / 1000.0
        line_pow.set_data(rng_km, dpc_pow)
        line_thr.set_data(rng_km, noise_floor)

        det_x = rng_km[det_mask]
        det_y = dpc_pow[det_mask]
        offsets = np.column_stack((det_x, det_y)) if det_x.size > 0 else np.empty((0, 2))
        line_dets.set_offsets(offsets)

        for ln in _cfar_tgt_lines:
            ln.remove()
        _cfar_tgt_lines.clear()

        for t in targets:
            r, az, el = t.polar_position
            if (abs(az - beam.az) <= sm.azimuth_width / 2.0 and
                    abs(el - beam.el) <= sm.elevation_width / 2.0 and
                    r / 1000.0 <= rng_km[-1]):
                ln = ax_cfar.axvline(r / 1000.0, color="lime", lw=0.8, ls=":", alpha=0.7)
                _cfar_tgt_lines.append(ln)

        ax_cfar.relim()
        ax_cfar.autoscale_view(scalex=False, scaley=True)
        ax_cfar.set_xlim(rng_km[0], rng_km[-1])
        ax_cfar.set_title(
            f"CFAR [{beam.submode_name}] guard={cfar.guard_cells} "
            f"train={cfar.training_cells} thr={cfar.threshold}"
        )

        # ── PPI clear ────────────────────────────────────────────────
        for sc in ppi_det_sc:
            sc.remove()
        ppi_det_sc.clear()

        if ppi_tgt_sc[0] is not None:
            ppi_tgt_sc[0].remove()
            ppi_tgt_sc[0] = None

        if ppi_sector[0] is not None:
            ppi_sector[0].remove()
            ppi_sector[0] = None

        for artist in getattr(ax_ppi, "_track_artists", []):
            artist.remove()
        ax_ppi._track_artists = []

        # ── PPI beam sector ──────────────────────────────────────────
        half_az = sm.azimuth_width / 2.0
        ppi_sector[0] = mpatches.Wedge(
            (0, 0), max_r,
            beam.az - half_az, beam.az + half_az,
            color=BEAM_COLOR.get(getattr(beam.purpose, "name", str(beam.purpose)).lower(), "white"),
            alpha=0.10, zorder=1,
        )
        ax_ppi.add_patch(ppi_sector[0])

        # ── PPI true targets ────────────────────────────────────────
        txs, tys = [], []
        for t in targets:
            tx, ty = _xyz_to_ppi(t.position_xyz)
            if tx**2 + ty**2 <= max_r**2:
                txs.append(tx)
                tys.append(ty)
        if txs:
            ppi_tgt_sc[0] = ax_ppi.scatter(
                txs, tys, marker="P", s=30, c="white",
                alpha=0.35, linewidths=0.3, zorder=4,
            )

        # ── PPI detections ──────────────────────────────────────────
        for det in result.detections:
            dx, dy = _polar_to_ppi(det.range, det.azimuth, det.elevation)
            sc = ax_ppi.scatter(
                dx, dy, s=20, alpha=0.85, zorder=5,
                c=BEAM_COLOR.get(getattr(beam.purpose, "name", str(beam.purpose)).lower(), "white"),
            )
            ppi_det_sc.append(sc)

        # ── PPI tracks ──────────────────────────────────────────────
        for tr in result.tracks.values():
            if tr.state == TrackState.DROPPED:
                continue

            color = _track_color(tr.id)
            marker = STATE_MARKER.get(tr.state, "o")

            if len(tr.hit_history) >= 2:
                hist = np.array(tr.hit_history)
                ln, = ax_ppi.plot(hist[:, 0], hist[:, 1], "-", color=color, lw=1.1, alpha=0.75, zorder=6)
                ax_ppi._track_artists.append(ln)

            px, py = _xyz_to_ppi(tr.kalman.position_xyz)
            sc = ax_ppi.scatter(
                px, py, c=[color], s=45, marker=marker,
                zorder=7, edgecolors="white", linewidths=0.4,
            )
            ax_ppi._track_artists.append(sc)

            lbl = ax_ppi.text(
                px + max_r * 0.012, py + max_r * 0.012,
                tr.id[:6], color=color, fontsize=6, zorder=8,
            )
            ax_ppi._track_artists.append(lbl)

        # ── status text ─────────────────────────────────────────────
        q = mgr.queue_summary()
        n_tracked = sum(1 for t in result.tracks.values() if t.state == TrackState.TRACKED)
        n_candidate = sum(1 for t in result.tracks.values() if t.state == TrackState.CANDIDATE)
        n_confirming = sum(1 for t in result.tracks.values() if t.state == TrackState.CONFIRMING)
        n_lost = sum(1 for t in result.tracks.values() if t.state == TrackState.LOST)

        ppi_status.set_text(
            f"beam={beam.submode_name} az={beam.az:.1f}° el={beam.el:.1f}°\n"
            f"tracks={len(result.tracks)} tracked={n_tracked} cand={n_candidate} "
            f"conf={n_confirming} lost={n_lost}\n"
            f"queue coarse={q['coarse']} fine={q['fine']} confirm={q['confirm']} "
            f"track={q['track']} recovery={q['recovery']}"
        )

        return []

    anim = FuncAnimation(
        fig, update, interval=250, blit=False, cache_frame_data=False
    )
    plt.show()


if __name__ == "__main__":
    main()