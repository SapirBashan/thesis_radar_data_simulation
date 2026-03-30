from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from generate_signal import (
    build_snapshot,
    write_complex_2d_txt,
    write_float_1d_txt,
)


MODES: list[tuple[int, int]] = [
    (1024, 64),
    (256, 256),
    (64, 8),
    (512, 512)
]


def _mode_name(mode_idx: int, nrg: int, npri: int) -> str:
    return f"mode_{mode_idx}_nrg{nrg}_npri{npri}"


def _write_submode_stat_txt(
    path: Path,
    *,
    submode_name: str,
    amount_bytes: int,
    throughput_gbps: float,
    elapsed_ms: float,
    running_throughput_gbps: float,
) -> None:
    """
    Writes submode stats in a compact line-oriented format:
      line1: submode name (str)
      line2: amount bytes (int)
      line3: current throughput (GB/s)
      line4: processing time (ms)
      line5: running throughput average (GB/s)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(f"{submode_name}\n")
        f.write(f"{int(amount_bytes)}\n")
        f.write(f"{float(throughput_gbps):.9e}\n")
        f.write(f"{float(elapsed_ms):.9e}\n")
        f.write(f"{float(running_throughput_gbps):.9e}\n")
    tmp.replace(path)


def run_injector(
    output_dir: Path,
    interval_sec: float,
    mode_hold: int,
    max_steps: int,
    seed: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    throughput_sum_gbps = 0.0

    while True:
        if 0 < max_steps <= step:
            break

        mode_idx = (step // max(1, mode_hold)) % len(MODES)
        nrg, npri = MODES[mode_idx]

        # Deterministic stream if seed is provided.
        rng = np.random.default_rng(None if seed is None else seed + step)

        t0 = time.perf_counter()
        raw, dpc_avg, fft_map, cfar_mask = build_snapshot(nrg=nrg, npri=npri, rng=rng)

        write_complex_2d_txt(output_dir / "raw_signal.txt", raw)
        write_float_1d_txt(output_dir / "dpc_avg.txt", dpc_avg)
        write_complex_2d_txt(output_dir / "fft_signal.txt", fft_map)
        write_float_1d_txt(output_dir / "cfar_mask.txt", cfar_mask)

        elapsed_sec = max(1e-9, time.perf_counter() - t0)

        # Approximate payload size for this submode snapshot (binary payload only).
        amount_bytes = (
            raw.size * 16
            + dpc_avg.size * 8
            + fft_map.size * 16
            + cfar_mask.size * 8
        )
        throughput_gbps = amount_bytes / elapsed_sec / 1e9
        elapsed_ms = elapsed_sec * 1e3

        throughput_sum_gbps += throughput_gbps
        running_avg_gbps = throughput_sum_gbps / float(step + 1)

        _write_submode_stat_txt(
            output_dir / "submode_stat.txt",
            submode_name=_mode_name(mode_idx, nrg, npri),
            amount_bytes=amount_bytes,
            throughput_gbps=throughput_gbps,
            elapsed_ms=elapsed_ms,
            running_throughput_gbps=running_avg_gbps,
        )

        print(
            f"[injector] step={step:06d} mode={mode_idx} nrg={nrg} npri={npri} "
            f"throughput={throughput_gbps:.3f} GB/s written to {output_dir}"
        )
        step += 1
        time.sleep(interval_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and inject random radar snapshots.")
    parser.add_argument("--output-dir", default="output_signal", help="Directory for txt snapshot files.")
    parser.add_argument("--interval-sec", type=float, default=2.5, help="Write period in seconds.")
    parser.add_argument("--mode-hold", type=int, default=1, help="Writes per mode before switching.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means infinite loop.")
    parser.add_argument("--seed", type=int, default=42, help="Optional deterministic seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_injector(
            output_dir=Path(args.output_dir),
            interval_sec=max(0.001, args.interval_sec),
            mode_hold=max(1, args.mode_hold),
            max_steps=max(0, args.max_steps),
            seed=args.seed,
        )
    except KeyboardInterrupt:
        print("\n[injector] stopped by user.")


if __name__ == "__main__":
    main()
