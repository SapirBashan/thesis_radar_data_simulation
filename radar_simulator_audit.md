# Technical audit of radar_signal_processing_line

Audit date: 27 October 2026. Repository: https://github.com/dattali18/radar_signal_processing_line. Inspected commit: `5fe4040b60a2a5df5a62f300a768dfc5de50b1d8`. Local inspection copy: `C:/Users/Sapir/Desktop/radar_signal_processing_line_audit`. No source files were edited. Line numbers below refer to this commit. This report distinguishes observations, mathematical consequences, and proposed work.

## 1. Executive summary

The repository contains **two independent generators**:

* `radar_signal_simulator`: a simplified coherent pulsed radar complex-baseband simulator with a planar receive array, rectangular delayed echoes, target Doppler, spatial phase and independent Gaussian I/Q noise (`generator.py:15–77`). It produces actual complex samples before matched filtering. It is not FMCW and does not generate a sampled RF carrier.
* `olm`: a synthetic display-data generator, simple slow-time processing, range-only threshold detection, text snapshot injection and Matplotlib monitoring (`generate_signal.py:17–150`, `injector.py:56–114`, `online_monitor.py:85–143`). Its target positions and Doppler are expressed in bins and cycles per PRI, without a physical radar configuration.

These paths are not wired together. The physical simulator's example performs channel summation, rectangular matched filtering and a Doppler FFT but **has no detector** (`test_generator.py:43–89`). OLM has a detector, but it operates on a coherent slow-time average that can suppress moving targets (`generate_signal.py:59–100`).

**Thesis verdict:** useful as a small, inspectable starting point for a controlled synthetic PoC; insufficient as a validated end-to-end radar compression benchmark. Fixing experimental definitions and establishing a reliable detector matters before implementing compression. The physical model omits range attenuation, propagation carrier phase, intra-pulse Doppler, range migration, clutter, interference and hardware effects. Power and gain settings are unused. Pulse-count configuration is inconsistent. There are no automated assertions/tests, dataset splits, evaluation metrics or real-data validation in the tracked repository.

Execution verified a default physical cube of shape `(16,256,2048)`, dtype `complex128`, occupying 128 MiB. The nominal 128-pulse setting does not control generation. A seeded minimal generation and OLM snapshot both succeeded. Detailed execution notes appear in section 5.

## 2. File-by-file architecture

Complete tracked tree (excluding Git internals):

```text
.
├── .gitignore
├── README.md
├── pyproject.toml
├── uv.lock
├── radar_signal_simulator/
│   ├── __init__.py
│   ├── simulator_config.py
│   ├── radar_config.json
│   ├── target.py
│   ├── targets.json
│   ├── generator.py
│   └── test_generator.py
└── olm/
    ├── __init__.py
    ├── generate_signal.py
    ├── injector.py
    ├── online_monitor.py
    └── main.py
```

| File | Purpose and evidence |
|---|---|
| `.gitignore:1–8` | Ignores `.venv`, IDE metadata, `output_signal/` and `.DS_Store`; does not define processing. |
| `README.md:1–25` | Documents monitor text formats. Mentions a C++ producer, but no C++ source is tracked. Its plot descriptions are stale relative to current monitor code. |
| `pyproject.toml:1–9` | Python >=3.12, NumPy >=2.0.0, Matplotlib >=3.10.8. No console scripts, test tooling or build-system declaration. |
| `uv.lock` | Dependency lock, not simulator logic: NumPy 2.4.4 at lines 262–263, Matplotlib 3.10.8 at 208–209; project dependency record at 422–434. Other records contain plotting dependencies and platform wheel metadata. |
| `radar_signal_simulator/__init__.py:1–3` | Exports `RadarConfig`, `ArrayConfig`, `Target`, `RadarSignalGenerator`, `C`. |
| `simulator_config.py:4–65` | Array/submode/radar dataclasses, derived PRI/wavelength, active-submode lookup and JSON load/save. `SubmodeConfig.npri` aliases `num_doppler_bins` (16–18). No validation. |
| `radar_config.json:1–28` | Default physical configuration and two submodes. |
| `target.py:4–21` | Five-field target dataclass and JSON list load/save. No dynamics or velocity conversion. |
| `targets.json:1–16` | Two explicit target truth records: 12 km / +350 Hz / RCS 5 and 8 km / -120 Hz / RCS 1, with angles. |
| `generator.py:5–77` | Speed of light, dB conversion, planar steering and received I/Q synthesis with noise. |
| `test_generator.py:6–92` | Interactive demonstration called `main`, not an assertion-based test. Loads bare relative JSON paths, generates data, plots each receiver, sums receivers, matched-filters, windows and Doppler-transforms; plots coherent average. |
| `olm/__init__.py` | Empty package marker. |
| `olm/generate_signal.py:10–150` | CFAR parameter dataclass, independent synthetic I/Q generator, Doppler FFT, moving-sum/coherent-average profile, range detector, text writers and snapshot assembly. |
| `olm/injector.py:16–142` | Four size modes; seeded snapshot loop; overwrites five files; estimates throughput; CLI. Imports sibling module by bare name at 9–13. |
| `olm/online_monitor.py:9–357` | File readers, snapshot polling, metadata text and persistent plotting artists. Does not run the physical generator. |
| `olm/main.py:3–21` | Polling monitor entry point, sleeps 1.25 s, closes on interrupt. Bare sibling import and CWD-relative output path. |

No tracked notebooks, external datasets, CI definitions, C++ implementation or additional test suite were found in `git ls-files`.

Dependency/data flows:

```text
radar_config.json → RadarConfig.load ─┐
targets.json → Target.load_list ─────┴→ RadarSignalGenerator.generate
 → [receiver,pulse,range] complex cube
 → test_generator.main: receiver sum → fast-time matched filter
 → slow-time Hann → Doppler FFT → plots

injector CLI → size mode + default_rng(seed+step)
 → build_snapshot → synthetic raw [pulse,range]
   ├→ FFT(axis=0) + shift → fft_signal.txt
   └→ slow-time moving sum → coherent mean magnitude → dpc_avg.txt
       → range threshold detector → cfar_mask.txt
 → raw_signal.txt + submode_stat.txt
 → OnlineMonitor.update → four plots
```

Entry points: `python olm/injector.py` and `python olm/main.py` execute their `main` functions. Their default paths are not mutually aligned from the repository root: injector writes `output_signal`, while monitor main reads `../output_signal` (`injector.py:119`, `main.py:7`). The radar demonstration can run with the package importable and CWD set to `radar_signal_simulator`, because JSON loads use bare filenames (`test_generator.py:9–10`). No standalone CLI exists for `RadarSignalGenerator`; direct Python API invocation is the most reliable current entry point.

## 3. Complete radar signal generation pipeline

### Physical pulsed-baseband simulator

1. **Configuration:** `RadarConfig.load` reconstructs `ArrayConfig` from JSON (`simulator_config.py:60–65`). PRI is reciprocal PRF; wavelength is c/fc (41–48). Active submode supplies both range gates and number of generated pulses (50–53, 16–22).
2. **Sampling grid:** `generate` allocates complex64 zeros `[rows*cols,npri,nrg]`. Fast times are `n/fs`; pulse start times are `p*PRI` (`generator.py:38–44`). Fast time restarts each pulse.
3. **Target representation:** `Target` supplies RCS, range, Doppler frequency, azimuth and elevation (`target.py:4–10`). Range is interpreted as metres by the delay equation, Doppler as Hz by multiplication with seconds, angles as degrees by `deg2rad`. RCS acts as a dimensionless amplitude-control power, conventionally interpretable as square metres only if a calibrated radar equation is added. No velocity field exists.
4. **Waveform/echo envelope:** delay is `2R/c`, then a rectangular indicator selects samples between delay and delay+pulse width, inclusively (`generator.py:48–57`). There is no separate transmitted waveform object or RF oscillator. The implied transmitted baseband waveform is a unit rectangle. `delay_samples` is calculated but unused (49); there is no fractional-delay interpolation. Delay affects sample inclusion, not propagation phase.
5. **Doppler:** each pulse receives a complex rotation `exp(j2πfd p PRI)` (51–53). This is coherent within one generated cube. Every cube starts pulse index at zero. Doppler does not affect fast-time phase or range.
6. **Array:** elements form a centred xy grid; x varies fastest when flattened. Direction components are cos(el)cos(az), cos(el)sin(az); each receiver gets a unit-magnitude plane-wave phase (15–30). This is a far-field receive-array model, not MIMO or a simulated antenna pattern. With this coordinate convention broadside lies along z, at elevation ±90 degrees; elevation 0 lies in the array plane. A reader expecting elevation 0 to mean array broadside would interpret it incorrectly. The sign is a convention; it is not independently proven wrong.
7. **Amplitude/superposition:** each echo has amplitude sqrt(RCS), multiplied by slow-time phase, gate envelope and receiver steering, then added coherently to the cube (62–70). Equal-range/equal-Doppler targets can reinforce or cancel through spatial phase. No per-target random scattering phase or Swerling model exists.
8. **Noise/output:** independent global-NumPy Gaussian arrays are added to I and Q, with standard deviation `10**(noise_level/10)` per component (72–77). Noise is float64, so final addition returns complex128. Target contributions were previously accumulated with complex64 rounding.

All defaults (`radar_config.json:2–26`): fc=9.6 GHz; PRF=2 kHz; pulse width=2 microseconds; fs=20 MHz; `num_pulses`=128; transmit power=1000 and gains=30 each (units not specified or consumed); noise level=-15 (ambiguous dB convention); array 4×4, spacings 0.0156 m. NAV has nrg=2048 and npri=256; TA_TAMAS has nrg=1024 and npri=512. `num_doppler_bins` actually controls pulse count; no independent FFT-length setting is used.

Configurable through dataclass/JSON: those values, submode names/dimensions and active submode. Target fields are configurable through JSON/API. There is no input validation for negative RCS, zero PRF/fs, invalid active submode, nonpositive dimensions or impossible timing. Global `np.random.seed` controls physical noise; `generate` has no seed/RNG argument. The example seeds with 42 (`test_generator.py:7`).

Supported physical stochastic environment: independent additive Gaussian I/Q noise only. No implemented clutter, multipath, jamming, interference, atmospheric loss, receiver response, ADC quantization/saturation, phase noise, mutual coupling or channel mismatch. There is no range attenuation or radar equation, despite power/gain fields. There is no transmit blanking, pulse-to-pulse range evolution, acceleration, cross-frame continuity, waveform coding or chirp modulation (`generator.py:32–77` is the complete synthesis routine).

### Independent OLM synthetic generator

`generate_raw_signal` begins with `(N(0,1)+jN(0,1))*0.25`, hence complex noise power 0.125 (`olm/generate_signal.py:24–27`). It selects 2–5 targets, a start bin uniformly across range, drift in [-0.25,0.25) bins/PRI, frequency in [-0.2,0.2) cycles/PRI, and amplitude in [2,8), with uniform initial phase (30–45). Each target deposits one complex sample per pulse in a rounded/clipped bin. Multiple targets add coherently. The variable `doppler_hz` is misleading: no seconds or PRF appear in its phase equation. Clipping causes tracks to accumulate at boundary bins. Range drift and Doppler are sampled independently and are not kinematically consistent. Truth variables are local and are discarded when the raw array is returned (47). This generator supplies display-friendly data, not a calibrated raw-waveform echo model.

## 4. Important mathematical equations

For receiver e, pulse p and fast-time index n, the physical code implements

\[
x_{e,p,n}=\operatorname{accum}_{64}\!\left[\sum_q\sqrt{\sigma_q}\,a_e(q)e^{j2\pi f_{Dq}pT_r}\,\mathbf1\{\tau_q\le n/f_s\le\tau_q+T_p\}\right]+w_{e,p,n},
\]

where the accumulator is complex64 and final output is complex128. This is a transcription of `generator.py:39–77`, not an assertion that a full radar equation is implemented.

\[
T_r=1/f_{PRF},\quad\lambda=c/f_c,\quad\tau_q=2R_q/c,
\quad a_e=e^{j(2\pi/\lambda)(x_e\cos el_q\cos az_q+y_e\cos el_q\sin az_q)}.
\]

Source: `simulator_config.py:41–48`, `generator.py:15–30,48–53`. Position units are metres; angles are converted from degrees; delay/PRI are seconds.

\[
w=\sigma_n(u+jv),\quad u,v\sim\mathcal N(0,1),\quad
\sigma_n=10^{L/10},\quad E|w|^2=2\,10^{L/5}.
\]

Source: `generator.py:7–8,73–75`. With L=-15, sigma=0.0316227766 and total noise power=0.002. At one occupied sample/receiver, isolated target SNR is RCS/0.002: approximately 33.98 dB for RCS 5 and 26.99 dB for RCS 1. These are arbitrary sample-power ratios, not calibrated received SNRs. Whole-cube SNR is lower because most gates contain only noise. If L were intended to be total complex-noise power dB, the required component standard deviation would instead be sqrt(10^(L/10)/2); that is a proposed interpretation, not the implementation.

The following theoretical quantities are derived from the implemented timing, not additional simulated capabilities:

| Quantity | Equation | Default consequence |
|---|---|---|
| Fast-time interval | 1/fs | 50 ns |
| Range sample spacing | c/(2fs) | 7.49481145 m; not range resolution |
| Unmodulated-pulse resolution scale | c Tp/2 | 299.792458 m, roughly 40 sample spacings |
| Pulse ambiguity distance | c/(2PRF) | 74,948.1145 m; code does not fold distant echoes |
| NAV recorded range coordinate | n c/(2fs), n=0…2047 | 0…15,342 m approximately; late echoes may be truncated |
| NAV observation duration | N/PRF | 0.128 s for N=256 |
| NAV Doppler-bin spacing | PRF/N | 7.8125 Hz |
| Slow-time Nyquist interval | [-PRF/2,PRF/2) | [-1000,1000) Hz |
| Conventional monostatic velocity mapping | v=lambda fd/2 | +350 Hz → +5.465 m/s; -120 Hz → -1.874 m/s; sign convention must be established |
| Conventional unambiguous velocity magnitude | lambda PRF/4 | 15.6142 m/s |
| NAV velocity-bin spacing | lambda PRF/(2N) | 0.121986 m/s |

TA_TAMAS yields 3.90625 Hz Doppler-bin spacing and 0.256 s duration if all generated pulses are used; the current example instead uses 128 pulses, 15.625 Hz bins and 0.064 s. Hann windowing broadens the main lobe: FFT spacing should not be presented as the achieved two-target resolution. Fast-time sample rate is adequate to represent a 40-sample rectangular pulse approximation; an ideal rectangle has unlimited spectral support, and no analog anti-alias model is implemented. A 20 MHz baseband sample rate is not intended to sample the 9.6 GHz RF carrier.

The OLM equations (`generate_signal.py:39–45,55–71,92–98`) are

\[
r_q[p]=\operatorname{clip}(\operatorname{round}(r_{0q}+d_qp),0,G-1),\quad
x[p,r_q[p]]\mathrel{+}=A_qe^{j(2\pi\nu_qp+\phi_q)},
\]
\[
X[k,g]=\operatorname{fftshift}_k\sum_{p=0}^{N-1}x[p,g]e^{-j2\pi kp/N},
\quad d[g]=\left|N^{-1}\sum_p (x[:,g]*\mathbf1_L)[p]_{same}\right|,
\]
\[
m[g]=\mathbf1\left\{d[g]>\frac{K}{2T}\left(\sum_{i=g-G_d-T}^{g-G_d-1}d[i]+\sum_{i=g+G_d+1}^{g+G_d+T}d[i]\right)\right\}.
\]

Here L=min(10,N), guards G_d=2, training cells per side T=12, scale K=3.5. Only interior bins are tested; outer 14 bins on each side remain zero. FFT is unnormalized. The profile is magnitude of a coherent mean, not average power or a conventional delay-line canceller.

## 5. Signal data format and sample inspection

### Exact data interfaces

| Point | Shape and dtype | Meaning |
|---|---|---|
| Physical accumulation (`generator.py:39`) | `(elements,npri,nrg)` complex64 | Internal noiseless target sum, already spatially and Doppler phased |
| `generate` return (77) | same shape, complex128 | Earliest public noisy digital I/Q; raw complex baseband envelope before matched filtering |
| Default NAV | `(16,256,2048)` | Element index=row*4+column; pulse index; fast-time/range sample |
| TA_TAMAS | `(16,512,1024)` | Different pulse/fast-time dimensions; both supplied targets fall outside the recorded fast-time window |
| Example channel sum (`test_generator.py:44`) | `(256,2048)` complex128 | All elements summed with no target steering weights |
| Example matched-filter output (52–55) | `(128,2048)` complex128 by default | Only first cfg.num_pulses rows processed |
| Example Doppler FFT (58–60) | `(128,2048)` complex128 | Windowed slow-time FFT; dB visualization is real |
| OLM raw/FFT (`generate_signal.py:17–56`) | `(npri,nrg)` complex128 | Single synthetic channel; no physical sample rate |
| OLM DPC/mask (59–100) | `(nrg,)` float64 | Amplitude profile and 0/1 detections |

OLM modes are `(nrg,npri)=(1024,64),(256,512),(64,8),(256,512)` (`injector.py:16–21`). Their output shapes reverse this tuple order. Default NAV fast time covers 2048 samples over a nominal 102.4 microsecond acquisition window; last sample time is 102.35 microseconds. PRI is 500 microseconds. This is partial-PRI acquisition.

The physical generator returns complex arrays and the input target list remains available as scene truth. It does not bundle truth with output, export datasets or return explicit support/association labels. OLM returns no target truth. Physical samples have arbitrary amplitude units, not volts or calibrated ADC counts.

Complex binary NumPy serialization could preserve returned samples and phase exactly, but no such exporter exists in the repository. OLM's text writer preserves both real and imaginary components, hence retains approximate phase, but formats each component with `.9e` (10 significant digits), which is **not bit-exact complex128 serialization** (`generate_signal.py:103–112`). The header is `nrg npri`, while flat sample order is row-major pulse then range. The monitor reconstructs complex128 and reshapes `(npri,nrg)` (`online_monitor.py:145–171`). It accepts only a 2D matrix; a physical multi-receiver cube cannot be passed directly to that writer.

### Executed numerical inspection

Direct API generation used bundled Python 3.12.14 and NumPy 2.3.5, `np.random.seed(42)`, repository JSON files, and `RadarSignalGenerator(cfg).generate(targets)`. This NumPy version satisfies pyproject but is not the uv.lock NumPy version. Observed:

```text
shape: (16,256,2048), dtype: complex128, nbytes: 134217728
real min/max: -2.3442933706041704 / 2.3482235788095527
imag min/max: -2.359884711813073 / 2.34588372574821
absolute min/max: 1.359332032715372e-05 / 2.3774405250664867
x[0,0,0:3]:
  0.01570748+0.08059880j
 -0.00437230+0.02305353j
  0.02048171+0.04185590j
x[0,0,1068]: -0.9615792752843801-0.30114379874991704j
x[0,0,1602]:  1.9195542796421439+1.0746623196545546j
noise-only power over gates 0:1000: 0.0020012816594762598
repeat with seed reset: array_equal True
```

The noise measurement agrees with 0.002. An initial inline diagnostic printed an incorrect theoretical comparator because of operator precedence in the diagnostic expression; it was recomputed separately as `2*(10**(-15/10))**2`, yielding 0.0019999999999999996. No source change was involved.

12 km target support: gates 1602–1641 inclusive, 40 samples. 8 km target support: gates 1068–1107, 40 samples. Single-element 256-pulse FFT peak frequencies were respectively 351.5625 and -117.1875 Hz, the nearest bins to 350 and -120 Hz. This validates the discretized model's delay/Doppler implementation, not the missing propagation physics.

OLM `build_snapshot(64,8,default_rng(42))` returned:

| Output | Shape / dtype | Absolute min / max |
|---|---|---|
| raw | `(8,64)` complex128 | 0.00364407434 / 6.41565466914 |
| dpc | `(64,)` float64 | 0.08899403430 / 18.77619298814 |
| fft | `(8,64)` complex128 | 0.02027545285 / 43.13784424174 |
| mask | `(64,)` float64 | 0 / 1 |

First raw sample was `0.07617927+0.02486937j`. A separate deterministic 64-pulse unit-amplitude test gave DPC=9.609375 for a stationary tone and 0.09106917382416008 for a 0.125-cycle/PRI tone, demonstrating strong moving-target suppression by this detector's input statistic. These are audit probes, not repository tests.

### Commands and environment

* `git clone https://github.com/dattali18/radar_signal_processing_line.git radar_signal_processing_line_audit`: initial sandbox network failure; succeeded after network approval.
* `git rev-parse HEAD`, `git ls-files`, line-numbered reads of every source/config/documentation file, and lock package inspection established scope/version.
* `python --version`: failed because Python is not on PATH. Used `C:/Users/Sapir/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe` thereafter.
* Inline Python via standard input, with `-B`, imported the repository API and OLM functions and produced the measurements above. No analysis script was saved into the source tree.
* Initial import check found Matplotlib unavailable. Installation was requested separately into `C:/Users/Sapir/Desktop/radar_audit_dependencies`, using `-m pip install --target ... matplotlib==3.10.8 pytest`; sandbox network access failed and the approved retry downloaded packages. Runtime follow-up is recorded below.
* Initial `-m pytest -p no:cacheprovider -q` and `-m radar_signal_simulator.test_generator` attempts failed with missing pytest/Matplotlib. These failures are environment failures, not test failures. The bundled Python's isolated path configuration requires explicit `sys.path` insertion for external dependencies.

### Final execution results

The separate dependency installation succeeded. Package directories were inaccessible inside the sandbox; approved execution outside it resolved that access issue. Final plotting/test environment was Python 3.12.14, NumPy 2.5.4, Matplotlib 3.10.8 and pytest 9.1.1. This was a compatible environment, **not a full uv.lock reproduction**. Earlier numerical inspection above used bundled NumPy 2.3.5.

Commands were inline Python piped to the explicit Python executable with `-B`, using these exact operations after `sys.path.insert(0,'C:/Users/Sapir/Desktop/radar_audit_dependencies')` and insertion of the repository root:

```python
pytest.main(['-p', 'no:cacheprovider', '-q'])
# no tests ran in 10.74s; exit code 5
runpy.run_module('radar_signal_simulator.test_generator', run_name='__main__')
# from repository root: FileNotFoundError for radar_config.json
os.chdir('radar_signal_simulator')
runpy.run_module('radar_signal_simulator.test_generator', run_name='__main__')
# completed successfully in a fresh process with MPLBACKEND=Agg
```

An intermediate attempt to run the example in the same process after pytest failed because pytest's assertion-rewriting import hook lacked `get_code`; the fresh-process rerun above resolved this audit-harness issue. The successful unchanged example emitted only the expected noninteractive-Agg `plt.show()` warning. No visual correctness claim is made from headless completion alone. There are no collected automated tests to report as passing.

The existing injector and monitor also completed one end-to-end headless cycle, importing from `olm` via an explicit path:

```python
run_injector(Path('C:/Users/Sapir/Desktop/radar_audit_results/snapshot'),
             0.001, 1, 1, 42)
m = OnlineMonitor('C:/Users/Sapir/Desktop/radar_audit_results/snapshot')
m.update(0)
m.close()
```

Injector printed step 0, mode 0, nrg=1024, npri=64, estimated throughput 0.008 GB/s. This exercised existing generation, processing, file writing, file reading and plot update. Snapshot files were written outside the repository. `git status --porcelain` was empty after physical-example execution and again after the injector run. No implementation or compression code was added.

## 6. Radar processing and detection pipeline

### Physical demonstration

`test_generator.main` is the only physical processing implementation:

1. `20 log10(abs(signal)+1e-12)` for per-channel display (15–41), although labelled power. Numerically this is a magnitude-dB representation equivalent to power dB only with a corresponding squared reference.
2. Sum all receivers (43–44). This is a fixed unsteered beam, not target-directed beamforming. No normalization by element count.
3. Build a 40-tap unit rectangle, reverse/conjugate it and convolve along range using `same` (46–55). Output is unnormalized; pulse peak gain is about 40 when fully aligned. `same` centres the response, so the peak corresponds approximately to echo centre, not its leading-edge range. Without a delay calibration, displayed target ranges have about half-pulse (~150 m) offset, subject to discrete alignment.
4. Hann window slow time, FFT axis 0, shift axis 0, magnitude dB (57–60). No range FFT or dechirping is needed for this unmodulated pulsed model. FFT length follows the processed rows.
5. Range and Doppler display coordinates (62–75), then squared magnitude of the **complex mean** over pulses (78–89). The latter suppresses nonzero Doppler rather than estimating noncoherent mean power.

No CFAR, peak extraction, target list, angle estimation, tracking or probability metrics are implemented in this path. Array sum gain magnitude measured for the supplied target angles was only 0.56090 and 0.61762, versus 16 for a fully coherent broadside return; independent receiver noise power increases by 16. These targets are poorly served by the fixed sum. That is a beam choice limitation, not proof that steering synthesis is wrong.

### OLM processing and visualization

`compute_fft_map` performs an unwindowed, unnormalized slow-time FFT and shift (50–56). `compute_dpc_avg` performs a length-min(10,N) moving sum in slow time using zero-padded `same` convolution, complex mean over all pulses, then magnitude (59–71). It is not mean magnitude, mean power, or a standard high-pass Doppler canceller. Edge padding changes profile scale with pulse count.

`compute_cfar_mask` compares each interior range bin with 3.5 times the mean of 24 amplitude training values, excluding two guards on each side (75–100). It returns a float mask; arrays shorter than 29 gates get all zeros. This resembles CA-CFAR structurally but does not accept a desired Pfa or derive a threshold from a specified null distribution. The usual independent exponential-power CA-CFAR Pfa formula cannot simply be applied to this correlated amplitude statistic.

`build_snapshot` sends the same raw matrix into separate FFT and DPC branches; CFAR uses DPC, **not the FFT map** (145–150). Monitor plot 11 shows raw magnitude (237–258), plot 12 DPC (260–273), plot 21 mean FFT power over Doppler (275–297), and plot 22 recomputes a raw-data slow-time FFT power map with one strongest-Doppler marker for each range detection (299–353). Those markers do not constitute independent Doppler-domain detection. By Parseval, mean unnormalized FFT power equals sum input pulse power per range, so plot 21 scale increases with pulse count.

Identical original/decompressed processing is feasible in principle because functions accept NumPy arrays. It is directly accessible in OLM, but physical processing is embedded in `main`, requiring a proposed small extraction into a reusable function. The physical cube must either use per-channel processing/beamforming or be reduced intentionally before a 2D OLM interface. No existing end-to-end physical-data detector can be claimed.

## 7. Confirmed limitations and potential bugs

### Confirmed implementation problems

| Finding | Evidence and consequence |
|---|---|
| Two pulse counts | `generator.py:39,44` uses submode.npri; `test_generator.py:54,58,64–65` uses cfg.num_pulses. Defaults silently discard half the generated data. Setting cfg.num_pulses larger than npri can cause indexing failure. |
| Noise dB ambiguity/misconversion | `generator.py:73–75` uses power conversion as a Gaussian standard deviation. Changing the field by 10 dB changes complex noise power by 20 dB. Cannot treat it as conventional total-noise-power dB. |
| Inactive radar-equation parameters | `simulator_config.py:31–33` defines power/gains, but complete `generate` never reads them; amplitude is sqrt(RCS), independent of range. |
| Output dtype differs from buffer | complex64 allocation at 39, float64 Gaussian noise at 75, addition at 77 returns complex128. Memory/compression baselines must use actual return dtype. |
| Fragile execution paths | Bare JSON paths in physical example; bare sibling imports in OLM; injector/monitor default directory mismatch. |
| Detection blind edges | `generate_signal.py:89–98` never tests outer 14 range bins, although synthetic target generation can place targets there (34). |
| Monitor bundle consistency | `injector.py:80–83,100–107` replaces separate files sequentially. Atomicity is per file, not per complete snapshot; monitor reads independently (101–104). Same-shape mixed generations are undetectable; mode transitions can mix dimensions. No snapshot ID/checksum. |
| Lossy text serialization | `.9e` formatting at `generate_signal.py:112,140` rounds float64 values; unsuitable as an exact floating-point lossless reference. |
| README divergence | README describes plot 21 as heatmap and plot 22 DPC overlay; actual plots are mean FFT power line and range-Doppler overlay (`online_monitor.py:275–353`). |
| Throughput interpretation | `injector.py:85–98` uses estimated binary byte counts divided by generation+processing+text-write time. This is not measured text throughput or compression throughput. Running average is arithmetic mean of per-frame rates, not total bytes/total time. |

### Verified modelling simplifications

No propagation carrier phase `exp(-j4πR/lambda)`, no intra-pulse Doppler, no range-dependent spreading loss, no target motion in physical range gates, no fractional-delay waveform filtering, no antenna patterns or receiver calibration, no clutter/interference/hardware model. Source: complete `generator.py:15–77`. Consequently changing range within a gate can produce identical deterministic data; range phase relationships are not realistic. Across separate calls targets restart at zero slow-time phase, and scene continuity is not represented.

Rectangular inclusive endpoints can yield one more sample when both delay and end align exactly to sample times (`generator.py:57`), while the matched filter always uses rounded width*fs (example 47). The two supplied delays yield 40 samples, so this did not affect the measured default targets. The general mismatch follows from the inequality, not from a failed default run.

PRF aliasing follows sampled slow-time phase; out-of-Nyquist Doppler silently aliases. Range ambiguity is not simulated: echoes beyond the acquisition window simply disappear, rather than wrapping from prior pulses. No checks distinguish out-of-window truth from detectable truth. TA_TAMAS has last range coordinate about 7.667 km, so both supplied target leading edges are beyond that submode's window. The 12 km target is also plainly out of view; the 8 km target does not enter at all.

### Suspected or interpretation-dependent issues

* Steering phase sign is conventional; validate against a chosen coordinate/beamforming convention before calling it defective.
* Missing intra-pulse Doppler may be acceptable for a first PoC at the supplied frequencies and 2-microsecond pulse, but not for all parameter regimes.
* Matched-filter range offset requires an explicit alignment convention and calibration; the default plots do not establish one.
* The lack of physical units for gain/noise means intent cannot be inferred from field names alone.

## 8. Suitability for compression research

| Question | Verified capability | Required extension or validation |
|---|---|---|
| Complex I/Q input? | Yes, pre-filter multi-receiver cube from physical generator. | Explicit dtype/serialization contract; optional ADC quantization model. |
| Reuse directly? | Config/target loaders, simple echo model, receive steering, Gaussian noise, primitive matched filter/FFT and display utilities. | Validate chosen scientific assumptions and processing conventions. |
| Small changes? | APIs accept configurable dimensions/scenes; RNG can be seeded globally. | Unify pulse count, expose RNG, define noise power/SNR, reusable processing function, save metadata and truth, align CLI paths. |
| Major gaps? | No physical end-to-end detection/evaluation. | Calibrated detector, truth association, Pd/Pfa, dataset generation/splitting, diverse physical scene model and external validation. |
| Thousands of samples? | Repeated API calls possible; OLM loops with seed+step. | No batch/dataset tool, labels, manifests or split logic. Stream generation: default physical cube alone is 128 MiB, 1000 cubes ≈125 GiB before intermediates. |
| Weak low-SNR targets? | RCS and Gaussian scale adjustable. | Define per-sample versus post-integration SNR, calibrate noise, avoid unsteered summation/nulls, add trials over target/noise conditions. Defaults are high occupied-sample SNR. |
| Pd and false alarms? | OLM emits range masks. | No metrics, associations, trial accounting or calibrated Pfa. Physical branch has no detector. |
| Reproducible splits? | Seed reset reproduces a physical cube; injector has deterministic seeds. | Scene manifests, fixed versions/configs, disjoint scene seeds/parameters, train-only normalization, avoid near-duplicate scenes across splits. |

Lossless compression of floating I/Q with independent Gaussian noise may have limited compressibility; that must be measured against byte counts, not assumed from visually sparse targets. Complex128 output also contains mixed precision: rounded target values plus full float64 noise. A claim about ADC-radar lossless compression would need a separately defined integer/quantized source baseline.

External validation should include a documented measured complex-I/Q dataset with compatible waveform/channel/timing metadata and independently credible target labels. No such dataset is bundled or validated by this audit. A second independently validated simulator or analytical single-target reference should check range alignment, Doppler sign, steering, noise statistics and detector false-alarm calibration. Real clutter and receiver imperfections are especially important before extending claims beyond this simple synthetic model. Selecting a specific external dataset requires waveform and access requirements; no compatibility claim is made here.

## 9. Proposed minimal integration points

These are proposals only; no implementation was performed.

1. **Primary thesis insertion:** immediately after `RadarSignalGenerator.generate(targets)` and before receiver sum (`test_generator.py:13` → 44). Preserve the entire `(receiver,pulse,range)` complex cube. This measures compression before beamforming, filtering and detection.
2. Establish a single processing function used by original and decompressed data. Extract the existing sum/filter/window/FFT stages, resolve pulse count and range calibration, and add a scientifically specified detector. Original and reconstructed inputs must use identical parameters and noise realizations.
3. Keep data/truth/config/seed/version metadata together. Save a binary baseline with exact dtype and layout before any text/plot conversion. Treat codec metadata and side information as part of compressed size.
4. For a quick software plumbing check only, insert a reconstruction between `raw=generate_raw_signal(...)` and subsequent processing (`olm/generate_signal.py:146–149`). This tests equality/differences through existing functions but does not establish physically credible radar detection performance.
5. If choosing a post-beamforming compression experiment, insert after example line 44 and state that receiver information has already been reduced. That is a different research question from compressing raw multichannel I/Q.

## 10. Recommended first PoC experiment

Before codecs, validate a noise-only scene and one target with an on-grid Doppler, then off-grid Doppler, two separated targets and weak-target-plus-strong-target cases. Use one receiver initially to avoid the current fixed-beam attenuation. Establish pulse count, waveform support, range alignment, SNR and detector false-alarm behavior. Archive inputs and analytical expectations. Do not tune detector thresholds separately for each codec.

Then run a paired experiment: one fixed original cube and a byte-exact identity round trip through the agreed storage format, followed by the exact same processing and detector. Require bit equality for the lossless path and identical detections. Only after this baseline works introduce one classical lossless codec and a simple controlled lossy baseline before deep learning.

Record compressed bytes including metadata, ratio relative to the actual source dtype, encode/decode time, memory, complex NMSE, phase error where amplitude is meaningful, range-Doppler error, target range/Doppler bias and amplitude changes. Estimate Pd over target trials and false alarms over explicitly defined eligible cells/no-target trials; exclude only predetermined blind/guard regions. Include confidence intervals and keep detection thresholds fixed at a declared operating point.

For learning, split independent scenes before making patches or frames, use distinct reproducible scene/noise seeds and held-out SNR/scene conditions, and retain an untouched final test set. Treat training on this simplified simulator and testing on measured I/Q as separate validation stages. The repository alone cannot substantiate general real-radar compression claims.
