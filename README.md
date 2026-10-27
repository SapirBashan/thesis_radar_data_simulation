# Thesis radar data simulation

MSc radar compression research workspace. The runnable simulator is in
`radar_simulator_v2`; the earlier checkout is retained in
`radar_signal_processing_line_audit` for comparison.

## Run the simulator

```powershell
git clone https://github.com/SapirBashan/thesis_radar_data_simulation.git
cd thesis_radar_data_simulation
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r radar_simulator_v2/requirements.txt
.\.venv\Scripts\python.exe radar_simulator_v2/run_demo.py scan
```

Use `raw` instead of `scan` for static receiver plots. Open the cloned root
folder in PyCharm, choose `.venv/Scripts/python.exe` as interpreter, and run
`radar_simulator_v2/run_demo.py`. No manual import-path setup is needed.

See `radar_simulator_v2/README.md` for details and `RUN_IN_PYCHARM.md` for the
original local setup notes. Paths in the audit/setup documents describe the
original machine and may need adjusting after cloning.

`radar_simulator_audit.md` audits the older main-branch checkout, not v2.
Compression algorithms have not been implemented. This is a simplified
synthetic simulator, not a validated measured-data benchmark.

## Attribution

Both simulator versions originate from
[dattali18/radar_signal_processing_line](https://github.com/dattali18/radar_signal_processing_line).
See `radar_simulator_v2/UPSTREAM.md` for provenance and upstream license status.
The original commit histories are preserved as parents of the workspace import.
Virtual environments, installed dependencies, IDE state and generated outputs
are excluded from this repository.
