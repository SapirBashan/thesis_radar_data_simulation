# Classical radar compression baselines

This framework compares actual stored streams and the **same corrected upstream
processing and detector** on original and reconstructed data. It is a first
synthetic baseline, not a validation of the radar model or a real-ADC benchmark.
No neural networks are included. No large study results are claimed.

## Run

From the thesis root, using Python 3.12+:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-baselines.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m radar_baselines.experiment --config experiments/classical_smoke.json --output results/my_smoke --save-dataset
```

Use `.venv/bin/python` on Linux/macOS. Every run requires a **new** output
directory; existing runs are never overwritten. The smoke configuration is small
and useful for checking execution, not drawing performance conclusions. The
larger `experiments/classical_study.json` is a proposed Monte Carlo run, not a
published result. It can take substantially longer. Increase independent trials
and null/calibration exposure for your desired confidence and Pfa.

The CLI runs in PyCharm as module `radar_baselines.experiment`, with the thesis
root as working directory and parameters `--config experiments/classical_smoke.json
--output results/my_smoke`. It uses a noninteractive backend to save figures.

## Reference sample formats

The simulator returns complex64 `[receiver, range sample, pulse]`, with last-axis
(pulse/slow-time) samples contiguous in C order. Each little-endian complex64
sample is two IEEE-754 binary32 components, interleaved I then Q: **8 bytes,
64 bits per complex sample**. These are simulated arbitrary baseband amplitude
units, **not ADC output**. No hidden float64-to-float32 conversion is allowed at
the codec API: unsupported input dtypes/shapes are rejected.

The alternative `adc16` source is explicitly quantized before compression:
little-endian signed int16 `[receiver, range, pulse, IQ]` (I then Q), **4 bytes,
32 bits per complex sample**. It is a simplified ADC representation, not a
hardware model. For one noise operating point the fixed full-scale amplitude is
`adc_full_scale_noise_sigmas * sigma_noise` (default 16 sigma); step is full
scale/32767. Quantization rounds to nearest/even and clips to [-32768,32767].
This gain is fixed across scenes, strengths and codecs at that noise setting,
not fitted separately to each target frame. It assumes known noise power.
The scalar step is stored in **each encoded stream** and used to reconstruct
physical-amplitude complex64 before PSP. ADC clipping counts and ADC-conversion
MSE relative to the floating simulator are reported separately.

Compression distortion is measured relative to the chosen source format.
Lossless ADC compression does not undo ADC quantization. Comparisons between
complex64 and adc16 must account for this separate initial distortion; never
attribute a raw int16/float32 storage difference to a codec.

`SampleBlock` and `Codec` live in `radar_baselines/codecs.py`:

```python
from pathlib import Path
from radar_baselines.codecs import SampleBlock, Codec, decode

# iq must be a little-endian complex64 array with three axes.
block = SampleBlock(iq)
codec = Codec('predictive_zstd', level=3)
encoded = codec.encode(block)            # actual bytes
Path('frame.rbc').write_bytes(encoded)
restored = decode(Path('frame.rbc').read_bytes())
assert block.data.tobytes() == restored.data.tobytes()
```

Every method exposes `encode(SampleBlock) -> bytes` and `decode(bytes) ->
SampleBlock`. The standalone decoder needs no scene manifest, original samples,
encoder object, trained state or external dictionary.

## Codecs and stored representation

* **raw**: uncompressed reference, with the same self-describing container.
* **zstd**: Zstandard on canonical source bytes; configurable level, single
  thread, content size and payload checksum. This preserves source bits.
* **predictive_zstd**: first slow-time sample per channel/range/IQ is stored;
  subsequent float32 component **bit patterns** use XOR against the preceding
  component word. For int16, residuals are unsigned modulo-65536 differences.
  Zstd compresses the residual bytes. Decoder uses cumulative XOR/modular sum.
  Floating arithmetic prediction is intentionally avoided: subtraction/addition
  would not preserve every IEEE bit pattern. This simple predictor may perform
  poorly on independent Gaussian noise; no compression gain is assumed.
* **baq**: basic peak-adaptive uniform scalar quantization with one scale shared
  by I and Q in each contiguous block of B complex samples. For b bits/component,
  qmax=2^(b-1)-1, scale=float32(max(abs(components))/qmax). An all-zero block
  uses scale=1. Quantizer is clip(round(x/scale),-qmax,qmax); stored unsigned
  symbols are code+qmax. The most-negative two's-complement equivalent is unused.
  Symbols are packed into exactly b bits each (little-bit-first), not stored
  as whole-byte estimates. Decoder multiplies by the **stored** float32 scale.
  ADC output is rounded/clipped back to int16. Blocks can cross range boundaries;
  they never change the declared flattening order. Last partial blocks are
  supported. This is a transparent basic BAQ, not a standardized SAR BAQ scheme
  or an entropy-coded/adaptively optimized quantizer.

Container v1: 8-byte `RADARC01` magic, uint32 little-endian JSON length, canonical
UTF-8 JSON, then payload. JSON includes method, dtype/endian, full shape, ADC step
or null, payload length, payload CRC32, and method settings. BAQ payload starts
with one little-endian float32 scale per block, then packed symbols. Zstd payload
contains its own frame header/checksum. Shapes and expected lengths are checked;
the decoded array is limited to 512 MiB. Corruption/truncation is rejected. This
is a research container for trusted experiment outputs, not a hardened general
file parser. All framing, checksums, scales and required metadata are counted.

The uncompressed source-byte numerator is `data.nbytes`; compression ratio is
source bytes / complete encoded stream bytes. Thus even `raw` has ratio slightly
below 1 due to metadata. Bits/complex sample = 8*encoded bytes / complex sample
count. This conservative convention is identical across codecs. CSV file bytes,
filesystem allocation and experiment/truth logs are not counted; they are not
needed to decode samples. Stream metadata alone contains the ADC step.

## Scene and noise model

`radar_baselines/radar.py` reuses `RadarSimulator.generate` with generator noise
disabled. It adds circular Gaussian noise using a local NumPy Generator, so
there is no dependence on global RNG state. A single receiver avoids the current
unsteered array sum's off-axis nulls. The model uses one target at azimuth and
elevation zero, randomized range in 35–65% of the acquisition window, randomized
Doppler in ±0.3 PRF and constant radial velocity. Defaults are 256 range samples,
32 pulses, 2 MHz sample rate, 2 kHz PRF, 2 microsecond rectangular pulse. The
upstream amplitude equation and carrier are retained. Multi-target, clutter,
array-channel and measured-data studies remain future validation work.

The SNR setting is **not whole-cube SNR**: it defines occupied-sample signal power
for an RCS=1 target at the middle range gate divided by per-channel complex-noise
power. `sigma_noise = sqrt(P_reference / 10^(SNR_dB/10))`, with I and Q variance
sigma_noise²/2. Noise level is fixed for all target strengths at that SNR. The
actual isolated occupied-target SNR is nominal SNR + 10log10(RCS) +
40log10(reference range/actual range), recorded per scene. This permits genuinely
weaker targets without renormalizing their noise to cancel the strength change.

Stable SeedSequence namespaces: 1 calibration, 2 independent null test, 3 target
test. Codecs see exactly the same original frame. Scene/noise seeds, truth,
parameters, source Git commit/dirty status, Python/platform and library versions
are saved. With fixed versions/configuration the numeric results reproduce;
wall-clock timing does not. `--save-dataset` saves noisy pre-ADC complex64 target
frames as NPY plus `scenes.jsonl`. Null scenes can be regenerated from recorded
seeds and configuration. No prior standalone dataset tool was found in this
checkout; this adapter supplies one minimal reproducible scene source. It does
not certify the simulator's physical accuracy.

## Downstream processing and matching

Original reference and reconstructed I/Q both use upstream `psp_process`:
receiver sum → rectangular matched filter along range → pulse-integrated power
→ range CA-CFAR → adjacent-range-hit merging → strongest Doppler bin at that
range → velocity conversion. The diagnostic map is the same matched-filter
output's unwindowed, unnormalized Doppler FFT power. There is no new range-Doppler
detector. `SceneGenerator.features` calls the same DBF/DPC/CFAR/FFT primitives
to expose null range-cell scores. Compression occurs **before** all processing.

One upstream correctness fix was necessary: in `psp/pipeline.py` the merger now
receives `det_rg[:,0]` and `x_pri_avg[:,0]`. Previously, multiplying 1D range
indices by (range,1) weights broadcast to a square matrix and gave the wrong
range centroid. This fix applies equally to every original and reconstructed
input; no codec-dependent processing is introduced. Known-target range/Doppler
regression tests protect this behavior.

Matched filtering produces a range offset and, sometimes, a maximum plateau.
Expected detection range is calculated **from ground-truth delay and the known
rectangular waveform and matched filter**, not from noisy or reconstructed data.
It is the mean coordinate of the noiseless maximum plateau. Default matching
gates are 4 range-bin spacings and 1.5 Doppler-bin velocity spacings. Truth Doppler
is kept unambiguous. A Hungarian one-to-one assignment accepts pairs only if
both gates pass; duplicates cannot count twice. Pd = matched truth targets /
eligible truth targets across independent scenes. Default is one target per
scene, so Wilson 95% intervals describe scene-level Bernoulli detections.
Unmatched detection objects in target scenes are reported as **counts**, not
mislabelled as per-cell false-alarm probabilities.

## False-alarm control

Pfa means false **range cells** / eligible range cells in independent **noise-only**
scenes, before adjacent-hit merging. The outer guard+training cells on each side
are excluded by exactly the upstream detector's rule. It is not a per-Doppler
cell or per-frame Pfa. Range cells are correlated by filtering; no independent
exponential-power CA-CFAR threshold formula or iid per-cell confidence interval
is assumed.

For each source format, nominal SNR, codec setting and requested Pfa:

1. Generate noise-only calibration frames and pass them through that codec.
2. Compute range score = integrated power / training-cell mean.
3. Set the threshold to the upper empirical (1-Pfa) quantile (`higher`), using
   the detector's strict greater-than comparison. Calibration uses no targets.
4. Freeze it and measure actual Pfa on **different** null-test seeds.
5. Process original and reconstructed target frames with exactly that threshold.

This measures a separately calibrated operating point per codec; it is not the
claim that codec thresholds remain numerically equal. A second, **fixed-threshold**
protocol uses the raw reference's calibrated threshold for every reconstruction;
its Pd and independently measured Pfa are also reported. The CSV records original
Pd at each method's threshold to make the paired comparison explicit.

Controlled-Pfa plots include only operating points whose held-out Pfa is within
the declared relative tolerance (default ±50%) and whose expected calibration
and null-test false-cell exposures are each at least 20. Every achieved Pfa,
exposure and status is saved, including failed points. This is a descriptive
Monte Carlo acceptance criterion, **not statistical proof of control**. It can
be tightened through configuration. Correlated-cell uncertainty and rare-event
confidence need larger independent-frame studies; low-Pfa estimates must not be
claimed from a handful of frames. Failed/underexposed points are omitted from
the controlled curve with an explanation, not silently relabelled as controlled.

## Metrics and output

`results.csv`: per target trial, codec, format, Pfa and configuration hash. It
contains actual byte counts, ratio, bits/complex sample, median encoding/decoding
milliseconds, MSE, SQNR, phase error/coverage, bitwise equality, ADC conversion
error/clipping, detection and matching counts, thresholds and held-out Pfa.

MSE = mean(|x-y|²) using complex128 arithmetic; SQNR = 10log10(mean(|x|²)/MSE).
Exact reconstruction gives MSE=0, SQNR=+inf. Phase error is wrapped angle(y*conj(x));
RMSE is in radians and only includes samples where **both** magnitudes exceed
0.1*sigma_noise by default. Eligible-original and valid-pair counts are reported
so an erased reconstruction cannot hide low phase coverage. An empty valid set
gives NaN, not zero error. These are reference-relative distortions and include
noise samples, not an isolated-target phase metric.

One untimed warm-up precedes repeated `perf_counter_ns` measurements. Reported
encoding includes transform, codec creation, packing and header serialization;
decoding includes parsing, inverse transform and source-array construction. ADC
conversion, disk I/O and downstream processing are excluded. Median per-frame
time is saved; summary means average those medians. This is serial Python timing,
not a real-time throughput certification. Bitwise lossless equality is checked
on **every measured frame**, including signed zeros/IEEE representations;
lossless detections must also equal the original detector output.

`summary.csv`: aggregate byte-weighted ratios, mean distortions/times, Pd and
Wilson intervals. `calibration.csv` and `null_trials.csv`: thresholds, calibration
counts and independent per-frame null counts for auditing. `config.json`: full
resolved settings, hash and provenance. `scenes.jsonl`: scene truth and seeds.

`streams/`: actual independently decodable `.rbc` files for the first trial of
each method/format/SNR/strength. All trials produce actual byte streams in memory;
only representative streams are persisted to avoid unnecessary storage.
`dataset/` is optional. PNG curves show ratio versus SQNR and Pd at checked Pfa;
exact methods are labelled SQNR=∞ instead of plotting fabricated finite values.
The range-Doppler example uses identical color limits and correctly centered
frequency/range-bin coordinates.

## Tests and limits

Run `python -m pytest -q`. Tests cover byte-exact lossless special floats and ADC
extremes, prediction inverses, BAQ bit packing and partial blocks/error bounds,
zero blocks, corrupted/truncated streams, a fresh-process decoder, reproducible
noise power and disjoint seeds, ADC clipping, phase coverage, known-target
range/Doppler matching, duplicate matching, full repeated experiments, stored
artifacts and prevention of output-directory overwrites.

Gaussian noise can limit lossless compressibility. A BAQ may amplify/suppress
detector statistics or erase phase at low magnitude. Neither improvement nor
degradation should be assumed in advance. Extend scene diversity and use measured
I/Q before drawing claims about real radar, operational Pfa or neural codecs.

Zstandard API references: [compression](https://python-zstandard.readthedocs.io/en/latest/compressor.html)
and [decompression](https://python-zstandard.readthedocs.io/en/latest/decompressor.html).
