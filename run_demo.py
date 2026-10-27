"""Run the upstream radar demos without manual import-path configuration."""

import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("demo", choices=("scan", "raw"), nargs="?", default="scan")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "radar_signal_simulator"
    sys.path.insert(0, str(source))
    import numpy as np

    np.random.seed(args.seed)
    script = source / ("scan_demo/run_scan.py" if args.demo == "scan" else "tests/test_generator.py")
    previous_directory = Path.cwd()
    try:
        os.chdir(script.parent)
        runpy.run_path(str(script), run_name="__main__")
    finally:
        os.chdir(previous_directory)


if __name__ == "__main__":
    main()
