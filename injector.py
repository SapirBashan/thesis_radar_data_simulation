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
]


def run_injector(
    output_dir: Path,
    interval_sec: float,
    mode_hold: int,
    max_steps: int,
    seed: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    step = 0

    while True:
        if max_steps > 0 and step >= max_steps:
            break

        mode_idx = (step // max(1, mode_hold)) % len(MODES)
        nrg, npri = MODES[mode_idx]

        # Deterministic stream if seed is provided.
        rng = np.random.default_rng(None if seed is None else seed + step)

        raw, dpc_avg, fft_map, cfar_mask = build_snapshot(nrg=nrg, npri=npri, rng=rng)

        write_complex_2d_txt(output_dir / "raw_signal.txt", raw)
        write_float_1d_txt(output_dir / "dpc_avg.txt", dpc_avg)
        write_complex_2d_txt(output_dir / "fft_signal.txt", fft_map)
        write_float_1d_txt(output_dir / "cfar_mask.txt", cfar_mask)

        print(
            f"[injector] step={step:06d} mode={mode_idx} nrg={nrg} npri={npri} "
            f"written to {output_dir}"
        )
        step += 1
        time.sleep(interval_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and inject random radar snapshots.")
    parser.add_argument("--output-dir", default="output_signal", help="Directory for txt snapshot files.")
    parser.add_argument("--interval-sec", type=float, default=3, help="Write period in seconds.")
    parser.add_argument("--mode-hold", type=int, default=1, help="Writes per mode before switching.")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means infinite loop.")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed.")
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
