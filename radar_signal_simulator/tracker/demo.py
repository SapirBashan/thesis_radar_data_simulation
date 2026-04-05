"""
PPI (Plan Position Indicator) demo for RadarManager.

Top-down radar display:
  - Concentric range rings
  - Radar at centre
  - Beam sweep shown as a fading sector (colour = submode)
  - Detections as bright dots
  - Real target positions overlaid as white '+' markers
  - Confirmed tracks as coloured lines (hit_history)
  - Track state shown as marker shape:
        FINE    = circle
        CONFIRM = triangle
        TRACKED = diamond
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ""))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from config.radar_config import RadarConfig
from config.submode_config import SubmodeLibrary
from config.target import Target
from radar_signal.generator import RadarSimulator
from psp import psp_process, CFARParams

from tracker.radar_manager import RadarManager, StepResult
from tracker.track import TrackState

C0 = 299_792_458.0


# ── coordinate helpers (original convention) ──────────────────────────

def _polar_to_ppi(range_m: float, az_deg: float, el_deg: float = 0.0):
    """x = R*cos(az), y = R*sin(az)  — matches sim XYZ convention."""
    az_r = np.deg2rad(az_deg)
    el_r = np.deg2rad(el_deg)
    rg   = range_m * np.cos(el_r)
    return rg * np.cos(az_r), rg * np.sin(az_r)


def _xyz_to_ppi(xyz):
    """Sim x=East, y=North maps directly to screen x, y."""
    return float(xyz[0]), float(xyz[1])


# ── simulation helpers ─────────────────────────────────────────────────

def _add_noise(signal, snr_db=20.0):
    P     = max(np.mean(np.abs(signal) ** 2), 1e-12)
    sigma = np.sqrt(P / (10 ** (snr_db / 10.0)))
    noise = (sigma / np.sqrt(2)) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


def _generate_targets(radar, n=40, r_min=3000.0, r_max=12000.0):
    az_min, az_max = radar.azimuth_range
    el_min, el_max = radar.elevation_range
    out = []
    for _ in range(n):
        r   = np.random.uniform(r_min, r_max)
        az  = np.deg2rad(np.random.uniform(az_min, az_max))
        el  = np.deg2rad(np.random.uniform(el_min, el_max))
        x   = r * np.cos(el) * np.cos(az)
        y   = r * np.cos(el) * np.sin(az)
        z   = r * np.sin(el)
        vx, vy, vz = np.random.uniform(-15, 15, 3)
        out.append(Target.new([x, y, z], [vx, vy, vz], np.random.uniform(0.5, 5.0)))
    return out


def _move_targets(targets, dt_s):
    for t in targets:
        x, y, z    = t.position_xyz
        vx, vy, vz = t.velocity_xyz
        t.position_xyz = [x + vx * dt_s, y + vy * dt_s, z + vz * dt_s]


# ── colour palette ────────────────────────────────────────────────────

_CMAP               = cm._colormaps["tab20"].resampled(20)   # no deprecation warning
_track_color_cache: dict[str, tuple] = {}
_color_idx          = 0


def _track_color(tid: str):
    global _color_idx
    if tid not in _track_color_cache:
        _track_color_cache[tid] = _CMAP(_color_idx % 20)
        _color_idx += 1
    return _track_color_cache[tid]


STATE_MARKER = {
    TrackState.FINE:    "o",
    TrackState.CONFIRM: "^",
    TrackState.TRACKED: "D",
    TrackState.DROPPED: "x",
}

BEAM_COLOR = {
    "coarse":  "#4fc3f7",
    "fine":    "#fff176",
    "confirm": "#ff8a65",
    "track":   "#a5d6a7",
}


# ── main ───────────────────────────────────────────────────────────────

def main():
    radar    = RadarConfig.load("configs/radar_config.json")
    submodes = SubmodeLibrary.load("configs/submodes.json")
    sim      = RadarSimulator(radar, submodes)

    mgr  = RadarManager(radar, submodes, max_misses=15, gate_m=500.0)

    targets = _generate_targets(radar, n=100)

    max_r = max(
        sm.num_range_gates * C0 / (2 * sm.fs)
        for sm in submodes.submodes.values()
    )

    # ── figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    ax.set_aspect("equal")
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.axis("off")
    fig.suptitle("Radar PPI Display", color="white", fontsize=13)

    status_text = ax.text(
        0.01, 0.99, "", transform=ax.transAxes,
        color="white", fontsize=8, va="top", family="monospace"
    )

    # range rings
    for r in np.linspace(max_r / 5, max_r, 5):
        ax.add_patch(plt.Circle((0, 0), r, color="#1e3a5f", fill=False, lw=0.8))
        lx, ly = _polar_to_ppi(r * 0.97, 45.0)
        ax.text(lx, ly, f"{r/1000:.0f} km", color="#4a7fa5", fontsize=7)

    # cross-hairs
    ax.axhline(0, color="#1e3a5f", lw=0.5)
    ax.axvline(0, color="#1e3a5f", lw=0.5)

    # radar marker
    ax.plot(0, 0, "*", color="#00e5ff", ms=12, zorder=10)

    # legend
    legend_items = [
        Line2D([0], [0], marker="P", color="white",   ms=7, label="True target",   linestyle="None"),
        Line2D([0], [0], marker="s", color="#4fc3f7", ms=6, label="Det (coarse)",  linestyle="None"),
        Line2D([0], [0], marker="s", color="#fff176", ms=6, label="Det (fine)",    linestyle="None"),
        Line2D([0], [0], marker="s", color="#ff8a65", ms=6, label="Det (confirm)", linestyle="None"),
        Line2D([0], [0], marker="s", color="#a5d6a7", ms=6, label="Det (track)",   linestyle="None"),
        Line2D([0], [0], marker="o", color="white",   ms=6, label="FINE",          linestyle="None"),
        Line2D([0], [0], marker="^", color="white",   ms=6, label="CONFIRM",       linestyle="None"),
        Line2D([0], [0], marker="D", color="white",   ms=6, label="TRACKED",       linestyle="None"),
    ]
    ax.legend(handles=legend_items, loc="lower right",
              facecolor="#0a0a1a", edgecolor="#4a7fa5",
              labelcolor="white", fontsize=7)

    # per-frame artist buckets
    det_scatters = []
    true_tgt_sc  = [None]
    sector_patch = [None]

    # ── animation ─────────────────────────────────────────────────────

    def update(_frame):
        nonlocal det_scatters

        beam = mgr.next_beam()
        sm   = submodes.submodes[beam.submode_name]
        op   = beam.op_submode

        _move_targets(targets, dt_s=sm.time / 1000.0)

        sig  = sim.generate(targets, op, noise_sigma=0.0)
        # sig  = _add_noise(sig, snr_db=20)
        cfar_param: CFARParams = CFARParams(**sm.cfar)
        dets = psp_process(sig, radar, submodes, op, cfar_param)

        result: StepResult = mgr.process(dets, beam, beam_time_ms=sm.time)

        print(
            f"t={result.sim_time_ms/1000:.2f}s  {beam.submode_name:12s} "
            f"az={beam.az:6.1f}°  dets={len(dets):3d}"
        )

        # ── clear previous frame artists ──────────────────────────────
        for sc in det_scatters:
            sc.remove()
        det_scatters.clear()

        if true_tgt_sc[0] is not None:
            true_tgt_sc[0].remove()
            true_tgt_sc[0] = None

        if sector_patch[0] is not None:
            sector_patch[0].remove()
            sector_patch[0] = None

        for artist in getattr(ax, "_track_artists", []):
            artist.remove()
        ax._track_artists = []

        # ── beam sector ───────────────────────────────────────────────
        half_az = sm.azimuth_width / 2.0
        sector_patch[0] = mpatches.Wedge(
            (0, 0), max_r,
            beam.az - half_az, beam.az + half_az,
            color=BEAM_COLOR.get(beam.purpose, "white"),
            alpha=0.10, zorder=1,
        )
        ax.add_patch(sector_patch[0])

        # ── true targets (white '+') ───────────────────────────────────
        txs, tys = [], []
        for t in targets:
            tx, ty = _xyz_to_ppi(t.position_xyz)
            if tx ** 2 + ty ** 2 <= max_r ** 2:
                txs.append(tx)
                tys.append(ty)
        if txs:
            true_tgt_sc[0] = ax.scatter(
                txs, tys, marker="P", s=35,
                c="white", alpha=0.40, linewidths=0.3,
                zorder=4,
            )

        # ── detections ────────────────────────────────────────────────
        for det in result.detections:
            dx, dy = _polar_to_ppi(det.range, det.azimuth, det.elevation)
            sc = ax.scatter(
                dx, dy, s=22, alpha=0.85, zorder=5,
                c=BEAM_COLOR.get(beam.purpose, "white"),
            )
            det_scatters.append(sc)

        # ── tracks ────────────────────────────────────────────────────
        for tr in result.tracks.values():
            if tr.state == TrackState.DROPPED:
                continue

            color  = _track_color(tr.id)
            marker = STATE_MARKER[tr.state]

            if len(tr.hit_history) >= 2:
                hist = np.array(tr.hit_history)
                ln, = ax.plot(
                    hist[:, 0], hist[:, 1],
                    "-", color=color, lw=1.2, alpha=0.75, zorder=6,
                )
                ax._track_artists.append(ln)

            px, py = _xyz_to_ppi(tr.kalman.position_xyz)
            sc = ax.scatter(
                px, py, c=[color], s=50, marker=marker,
                zorder=7, edgecolors="white", linewidths=0.5,
            )
            ax._track_artists.append(sc)

            lbl = ax.text(
                px + max_r * 0.012, py + max_r * 0.012,
                tr.id[:6], color=color, fontsize=6, zorder=8,
            )
            ax._track_artists.append(lbl)

        # ── status ────────────────────────────────────────────────────
        q         = mgr.queue_summary()
        n_tracked = sum(1 for t in result.tracks.values() if t.state == TrackState.TRACKED)
        n_fine    = sum(1 for t in result.tracks.values() if t.state == TrackState.FINE)
        n_confirm = sum(1 for t in result.tracks.values() if t.state == TrackState.CONFIRM)

        status_text.set_text(
            f"t={result.sim_time_ms/1000:.2f}s  "
            f"beam={beam.submode_name}  az={beam.az:.1f}°  el={beam.el:.1f}°\n"
            f"dets={len(result.detections)}  "
            f"tracked={n_tracked}  fine={n_fine}  confirm={n_confirm}\n"
            f"queue  coarse={q['coarse']}  fine={q['fine']}  "
            f"confirm={q['confirm']}  track={q['track']}"
        )

        return []

    anim = FuncAnimation(fig, update, interval=80, blit=False, cache_frame_data=False)  # noqa: F841
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()