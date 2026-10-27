# Radar Compression Thesis

Radar simulation workspace for an MSc thesis on radar data compression.
Derived from [dattali18/radar_signal_processing_line](https://github.com/dattali18/radar_signal_processing_line), branch `dev/simulator_v2`.
See [UPSTREAM.md](UPSTREAM.md) for attribution and license status.
Compression algorithms and a compression benchmark are not implemented yet.

## Quick start (Windows)

Install Python 3.12 or newer, including Tcl/Tk for interactive plots.

```powershell
git clone https://github.com/SapirBashan/thesis_radar_data_simulation.git
cd thesis_radar_data_simulation/radar_simulator_v2
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_demo.py scan
```

Linux/macOS: create the environment with `python3 -m venv .venv` and use
`.venv/bin/python` in place of the Windows interpreter path. On some Linux
distributions a separate OS Tk package is needed for interactive plots.

## Demos

```powershell
.\.venv\Scripts\python.exe run_demo.py scan --seed 42
.\.venv\Scripts\python.exe run_demo.py raw
```

`scan` opens an animated scan with signal power, range-Doppler display, a CFAR
profile and a 3D target view. Close the window to stop. `raw` shows per-element
signal-power plots. The launcher handles source imports and configuration paths;
no manual PYTHONPATH or local .pth file is needed. NumPy uses the supplied seed;
target UUID identifiers are generated independently.

## PyCharm

Open this repository folder and select the Python environment containing the
requirements. Run root-level `run_demo.py` for the scan, or set script parameters
to `raw` for the static example. If animations appear static in the Plots panel,
disable **Show plots in tool window** in Python Plots settings and use
`MPLBACKEND=TkAgg` with a Tk-enabled interpreter for a separate window.

## Structure

- `radar_signal_simulator/radar_signal/`: complex-baseband echo generation.
- `radar_signal_simulator/config/`: target, radar and submode configuration.
- `radar_signal_simulator/psp/`: beam sum, filtering, Doppler, CFAR and detections.
- `radar_signal_simulator/scan_demo/`: animated demo and scenario configurations.
- `radar_signal_simulator/tracker/`: upstream scheduling and tracking modules.
- `radar_signal_simulator/tests/test_generator.py`: static plotting example,
  not an assertion-based unit-test suite.
- `olm/` and `cpp_injector/`: upstream monitoring/C++ utilities, retained for
  reference. Building C++ is not required for the Python demos.

Raw simulator data uses `(receiver element, range sample, pulse)` axes with
complex64 samples. The model includes rectangular delayed echoes, radial-velocity
Doppler, array phase, radar-equation amplitudes and optional Gaussian noise.
It remains a simplified synthetic model; running it does not establish scientific
validity or equivalence to measured I/Q.

## Setup scope

`requirements.txt` pins the main runtime dependencies used here. The upstream
`pyproject.toml` and `uv.lock` are retained; requirements do not reproduce every
transitive lockfile version. Virtual environments, caches, IDE configuration,
credentials and generated results are excluded from Git. Local audit folders
and the older checkout are outside this repository and are not uploaded.

The setup changes preserve all upstream radar algorithms. The earlier audit of
the main branch does not cover this branch.

## Upstream TODOs

- Display detections rather than targets in the 3D view.
- Show detection and target counts in the sector.
- Extend radar-manager scheduling based on detections and tracks.

