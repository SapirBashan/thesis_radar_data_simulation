# Radar Online Monitor

Monitors text snapshots produced by a C++ signal-processing loop and renders 4 subplots in real-time.

## Expected files in `output_signal/`

- `raw_signal.txt`
  - first line: `nrg npri`
  - then $nrg \times npri$ lines: `real imag`
  - plotted as heatmap (`plot11`): $x=\text{range gate}, y=\text{PRI}$

- `dpc_avg.txt`
  - first line: `nrg`
  - then $nrg$ lines: `value`
  - plotted as line (`plot12`)

- `fft_signal.txt`
  - first line: `nrg npri`
  - then $nrg \times npri$ lines: `re im`
  - plotted as heatmap of $|z|^2$ (`plot21`)

- `cfar_mask.txt`
  - first line: `nrg`
  - then $nrg$ lines: `0/1` (or score)
  - plotted as detection overlay on DPC (`plot22`)
