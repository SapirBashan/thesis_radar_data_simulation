# Running the radar project in PyCharm

Open this folder: C:\Users\Sapir\Desktop\thesis

The repository, report and output folders are included in this project. The old
radar_audit_dependencies folder is retained but excluded from indexing; the project
uses its dedicated .venv instead (Python 3.12, NumPy 2.4.4, Matplotlib 3.10.8).

Choose a run configuration in the top toolbar:

- Radar simulator: generates physical radar data and opens plots. Close each plot
  window to advance to the next stage. This is the unchanged repository example.
- Snapshot injector: generates monitor snapshots continuously with seed 42.
- Online monitor: displays those snapshots. Start the injector first, then run
  the monitor alongside it. Stop both with the red Stop button when finished.

The working directories are already configured so the JSON files and monitor
output paths resolve correctly. Snapshot output is in
radar_signal_processing_line_audit/output_signal/ (ignored by Git).

Run configurations explicitly select .venv/Scripts/python.exe. If PyCharm asks
for a project interpreter for code completion, choose Add Interpreter > Add Local
Interpreter > Existing and select:
C:\Users\Sapir\Desktop\thesis\.venv\Scripts\python.exe

The environment was created from Codex's bundled Python runtime. Recreate it if
that runtime is later removed; it is not a standalone Python installation.

The audit's scientific limitations still apply. No simulator source was changed.

## Simulator v2 branch

The `dev/simulator_v2` branch is cloned separately at `radar_simulator_v2`.
Verified commit: 3418db76439fa3f70610e11d49bf08140bfcc27f.

New run configurations:

- **Simulator v2 - Scan demo**: animated scan, I/Q power, range-Doppler display,
  CFAR profile and 3D target view. Close the plot window to stop.
- **Simulator v2 - Raw IQ example**: static per-element I/Q power plots.

Choose either configuration in PyCharm's Run menu and press Run (Shift+F10).
If PyCharm was already open and the configurations are missing, close/reopen
this thesis project so it reloads the module and run configuration files.

V2 has a separate module/source root because it uses bare imports such as
`config`, `psp`, and `radar_signal`. Its run configurations set the source path
and working directory explicitly. The original checkout remains intact.
SciPy 1.17.1 and compatible SciPy typing stubs were added to the shared .venv.

Verified: static example, animated TkAgg window, sampled animation frames,
and generation plus detection in coarse_search, fine_search, confirm and track.
All four submodes returned complex64 cubes. `pip check` passed. No V2 tracked
source/configuration files were modified. These are runtime checks, not a full
scientific validation of this branch; the previous audit describes main only.
