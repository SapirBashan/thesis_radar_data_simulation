"""Seeded scene adapter and reuse of the unchanged upstream PSP implementation."""
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from io import StringIO
from pathlib import Path
import sys

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'radar_simulator_v2' / 'radar_signal_simulator'
sys.path.insert(0, str(SOURCE))
from config.radar_config import RadarConfig, ArrayConfig
from config.submode_config import SubmodeLibrary, SubmodeTemplate
from config.operating_submode import OperatingSubmode
from config.target import Target
from radar_signal.generator import RadarSimulator
from psp import psp_process, CFARParams, dbf_bfcu_mode, dpc, cfar_1d, range_doppler_power

from .codecs import SampleBlock

C = 299792458.0


@dataclass
class Settings:
    seed: int = 20261027
    trials: int = 24
    calibration_frames: int = 32
    null_test_frames: int = 32
    snr_db: tuple = (-20.0, -10.0, 0.0)
    rcs: tuple = (0.25, 1.0, 4.0)
    target_pfa: tuple = (0.01, 0.05)
    formats: tuple = ('complex64', 'adc16')
    zstd_levels: tuple = (3,)
    baq_bits: tuple = (2, 4, 6, 8)
    baq_blocks: tuple = (64, 128)
    range_gates: int = 256
    pulses: int = 32
    fs: float = 2e6
    prf: float = 2000.0
    pulse_width: float = 2e-6
    guard_cells: int = 6
    training_cells: int = 12
    range_tolerance_bins: float = 4.0
    doppler_tolerance_bins: float = 1.5
    adc_full_scale_noise_sigmas: float = 16.0
    phase_floor_noise_sigmas: float = 0.1
    timing_repeats: int = 3
    pfa_relative_tolerance: float = 0.5

    def validate(self):
        for key in ('trials', 'calibration_frames', 'null_test_frames', 'range_gates', 'pulses', 'timing_repeats'):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if self.trials >= 10000 or max(self.calibration_frames,self.null_test_frames) >= 1000000 or len(self.rcs) >= 100:
            raise ValueError('Trial grid exceeds disjoint seed index namespaces')
        if self.seed < 0 or self.seed >= 2**32 or not self.snr_db or not self.rcs or not self.target_pfa:
            raise ValueError('Invalid seed or empty experiment grid')
        if any(not np.isfinite(x) for x in (*self.snr_db, *self.rcs, *self.target_pfa)) or min(self.rcs) <= 0:
            raise ValueError('Invalid strengths/SNRs')
        if any(not 0 < p < 1 for p in self.target_pfa):
            raise ValueError('Pfa must lie in (0,1)')
        if not self.formats or any(f not in ('complex64', 'adc16') for f in self.formats):
            raise ValueError('Unsupported sample format')
        if self.guard_cells < 0 or self.training_cells < 1 or self.range_gates <= 2 * (self.guard_cells + self.training_cells) + 16:
            raise ValueError('Invalid CFAR geometry')
        if min(self.fs, self.prf, self.pulse_width, self.adc_full_scale_noise_sigmas, self.range_tolerance_bins, self.doppler_tolerance_bins) <= 0:
            raise ValueError('Timing/scales must be positive')
        if not 0 < self.phase_floor_noise_sigmas or not 0 < self.pfa_relative_tolerance < 1:
            raise ValueError('Invalid reporting tolerances')
        if self.range_gates / self.fs > 1 / self.prf or not 1 <= self.fs * self.pulse_width < self.range_gates / 8:
            raise ValueError('Acquisition must fit PRI and pulse must fit range window')

    def dictionary(self):
        return asdict(self)


class SceneGenerator:
    def __init__(self, settings):
        self.s = settings
        settings.validate()
        self.radar = RadarConfig.load(str(SOURCE / 'scan_demo/configs/radar_config.json'))
        # One receiver avoids the upstream unsteered array sum's target-angle nulls.
        self.radar.array = ArrayConfig(1, 1, self.radar.array.dx, self.radar.array.dy)
        self.params = CFARParams(settings.guard_cells, settings.training_cells, 1.0)
        self.submode = SubmodeTemplate(settings.fs, settings.prf, settings.pulse_width, settings.pulses,
                                       settings.range_gates, 60, 60, 1000 * settings.pulses / settings.prf, self.params)
        self.library = SubmodeLibrary({'baseline': self.submode})
        self.beam = OperatingSubmode.from_values('baseline', 0, 0)
        self.sim = RadarSimulator(self.radar, self.library)

    def seed(self, split, index):
        # Disjoint, stable seed namespaces independent of codec iteration/order.
        return int(np.random.SeedSequence([self.s.seed, split, index]).generate_state(1)[0])

    def noise_sigma(self, snr_db):
        r = self.s.range_gates / 2 * C / (2 * self.s.fs)
        cfg = self.radar
        ref_power = 10**(cfg.tx_power/10) * 10**(cfg.tx_gain/10) * 10**(cfg.rx_gain/10) * cfg.wavelength**2 / ((4*np.pi)**3*r**4)
        return float(np.sqrt(ref_power / 10**(snr_db / 10)))

    def scene(self, snr_db, rcs, split, index, target=True):
        seed = self.seed(split, index)
        rng = np.random.default_rng(seed)
        sigma = self.noise_sigma(snr_db)
        targets, truth = [], []
        if target:
            # Keep the complete pulse and its match gate away from CFAR edges.
            gate = rng.uniform(0.35, 0.65) * self.s.range_gates
            distance = gate * C / (2 * self.s.fs)
            fd = rng.uniform(-0.3, 0.3) * self.s.prf
            velocity = fd * self.radar.wavelength / 2
            t = Target(f'{seed:08x}', [distance, 0, 0], [velocity, 0, 0], float(rcs))
            targets = [t]
            env = ((np.arange(self.s.range_gates)/self.s.fs >= 2*distance/C) &
                   (np.arange(self.s.range_gates)/self.s.fs <= 2*distance/C+self.s.pulse_width)).astype(float)
            expected = np.convolve(env, np.ones(int(self.s.fs*self.s.pulse_width)), mode='same')
            peak_bin = float(np.flatnonzero(expected == expected.max()).mean())
            truth = [dict(id=t.id, range_m=distance, velocity_mps=velocity, doppler_hz=fd, rcs=rcs,
                          expected_detection_range_m=peak_bin*C/(2*self.s.fs))]
        clean = self.sim.generate(targets, self.beam, noise_sigma=0.0)
        noise = (rng.standard_normal(clean.shape) + 1j*rng.standard_normal(clean.shape)) * (sigma/np.sqrt(2))
        x = (clean + noise).astype('<c8')
        return x, dict(seed=seed, split=split, index=index, truth=truth, noise_sigma=sigma,
                       nominal_snr_db=snr_db, target_present=target,
                       actual_occupied_snr_db=(snr_db+10*np.log10(rcs)+40*np.log10((self.s.range_gates/2)/gate)) if target else None)

    def reference(self, x, format_name, sigma):
        if format_name == 'complex64':
            return SampleBlock(x), dict(adc_clipped_components=0, adc_conversion_mse=0.0)
        step = self.s.adc_full_scale_noise_sigmas * sigma / 32767
        components = np.stack([x.real, x.imag], axis=-1).astype(np.float64)
        codes = np.rint(components/step)
        clipping = int(((codes < -32768) | (codes > 32767)).sum())
        block = SampleBlock(np.clip(codes, -32768, 32767).astype('<i2'), step)
        mse = float(np.mean(np.abs(x.astype(np.complex128)-block.iq())**2))
        return block, dict(adc_clipped_components=clipping, adc_conversion_mse=mse)

    def features(self, iq):
        filtered = dpc(dbf_bfcu_mode(iq), int(self.s.fs*self.s.pulse_width))
        power = np.sum(np.abs(filtered)**2, axis=1)[:, None]
        noise, _ = cfar_1d(power, self.s.guard_cells, self.s.training_cells, 1.0)
        score = np.divide(power, noise, out=np.zeros_like(power,dtype=np.float64), where=noise > 0)[:, 0]
        eligible = np.zeros(self.s.range_gates, dtype=bool)
        pad = self.s.guard_cells+self.s.training_cells
        eligible[pad:-pad] = True
        return score, eligible, range_doppler_power(filtered)

    def detect(self, iq, threshold):
        params = CFARParams(self.s.guard_cells, self.s.training_cells, float(threshold))
        with redirect_stdout(StringIO()):
            return psp_process(iq, self.radar, self.library, self.beam, params)

    def match(self, detections, truth):
        # General one-to-one assignment for the gated range/velocity metric.
        from scipy.optimize import linear_sum_assignment
        if not truth or not detections:
            return 0, len(detections)
        dr = C/(2*self.s.fs)*self.s.range_tolerance_bins
        dv = self.radar.wavelength*self.s.prf/(2*self.s.pulses)*self.s.doppler_tolerance_bins
        costs = np.full((len(truth), len(detections)), 1e6)
        for i, t in enumerate(truth):
            for j, d in enumerate(detections):
                a = abs(d.range-t['expected_detection_range_m'])/dr
                b = abs(d.velocity-t['velocity_mps'])/dv
                if a <= 1 and b <= 1:
                    costs[i,j] = a+b
        rows, cols = linear_sum_assignment(costs)
        matches = int((costs[rows,cols] < 1e6).sum())
        return matches, len(detections)-matches
