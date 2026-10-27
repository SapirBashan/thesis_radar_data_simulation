"""Paired Monte Carlo experiments with disjoint calibration/null-test/target seeds."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter_ns

import numpy as np

from .codecs import Codec, decode
from .metrics import reconstruction_metrics
from .radar import Settings, SceneGenerator


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def timed_roundtrip(codec, block, repeats):
    # Warm-up excluded. Timings include serialization, header and codec setup,
    # not disk I/O, source conversion or downstream radar processing.
    codec.decode(codec.encode(block))
    enc, dec = [], []
    for _ in range(repeats):
        start = perf_counter_ns()
        blob = codec.encode(block)
        middle = perf_counter_ns()
        restored = decode(blob)
        end = perf_counter_ns()
        enc.append((middle-start)/1e6)
        dec.append((end-middle)/1e6)
    if codec.lossless and (block.data.tobytes() != restored.data.tobytes() or block.adc_step != restored.adc_step):
        raise AssertionError(f'Lossless reconstruction failed: {codec.name}')
    return blob, restored, float(np.median(enc)), float(np.median(dec))


def make_codecs(s):
    codecs = [Codec('raw')]
    for level in s.zstd_levels:
        codecs.extend([Codec('zstd', level=level), Codec('predictive_zstd', level=level)])
    codecs.extend(Codec('baq', bits=b, block_complex_samples=n) for b in s.baq_bits for n in s.baq_blocks)
    if len({c.name for c in codecs}) != len(codecs):
        raise ValueError('Duplicate codec configurations')
    return codecs


def wilson(hits, trials):
    z = 1.95996398454
    p = hits/trials
    center = (p+z*z/(2*trials))/(1+z*z/trials)
    width = z*np.sqrt(p*(1-p)/trials+z*z/(4*trials**2))/(1+z*z/trials)
    return center-width, center+width


def run(s, output, save_dataset=False):
    s.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'streams').mkdir()
    if save_dataset:
        (output/'dataset').mkdir()
    config_bytes = json.dumps(s.dictionary(), sort_keys=True).encode()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    repo = Path(__file__).resolve().parents[1]
    radar, codecs = SceneGenerator(s), make_codecs(s)
    def git(*args):
        result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else 'unavailable'
    manifest = dict(settings=s.dictionary(), config_sha256=config_hash, seed_namespaces={'calibration':1,'null_test':2,'target_test':3},
                    python=platform.python_version(), platform=platform.platform(), git_commit=git('rev-parse','HEAD'),
                    git_status=git('status','--porcelain'), versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','matplotlib','zstandard')},
                    scene_model='single_receiver_single_target_v1', source_axis_order=['receiver','range','pulse'],
                    radar_config=asdict(radar.radar),submode_config=asdict(radar.submode),
                    source_sha256={str(p.relative_to(repo)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for root in (repo/'radar_baselines',repo/'radar_simulator_v2'/'radar_signal_simulator')
                                   for p in root.rglob('*.py')})
    (output/'config.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    calibration, null_rows, results, scenes = [], [], [], []
    calibration_by_key = {}
    example = None
    for si, snr in enumerate(s.snr_db):
        for fmt in s.formats:
            # Same null realizations across methods; independent from target trials.
            cal_scores = {c.name:[] for c in codecs}
            test_scores = {c.name:[] for c in codecs}
            adc_cal_clips = {c.name:0 for c in codecs}
            for split, frames, scores in ((1,s.calibration_frames,cal_scores),(2,s.null_test_frames,test_scores)):
                for i in range(frames):
                    x, meta = radar.scene(snr,1.0,split,si*1000000+i,target=False)
                    block, adc_info = radar.reference(x,fmt,meta['noise_sigma'])
                    for codec in codecs:
                        y = decode(codec.encode(block))
                        score, eligible, _ = radar.features(y.iq())
                        scores[codec.name].append(score[eligible])
                        if split == 1:
                            adc_cal_clips[codec.name] += adc_info['adc_clipped_components']
            for pfa in s.target_pfa:
                base_threshold = float(np.quantile(np.concatenate(cal_scores['raw']),1-pfa,method='higher'))
                for codec in codecs:
                    samples = np.concatenate(cal_scores[codec.name])
                    threshold = float(np.quantile(samples,1-pfa,method='higher'))
                    n = sum(len(v) for v in test_scores[codec.name])
                    false = sum(int((v>threshold).sum()) for v in test_scores[codec.name])
                    fixed_false = sum(int((v>base_threshold).sum()) for v in test_scores[codec.name])
                    achieved = false/n
                    # Exposure and tolerance are declared descriptive criteria,
                    # not a claim that correlated range cells are iid Bernoulli.
                    status = ('insufficient_null_exposure' if min(len(samples),n)*pfa < 20 else
                              'within_tolerance' if abs(achieved-pfa) <= s.pfa_relative_tolerance*pfa else 'outside_tolerance')
                    row = dict(format=fmt,snr_db=snr,codec=codec.name,target_pfa=pfa,threshold=threshold,
                               reference_threshold=base_threshold,calibration_cells=len(samples),
                               calibration_false_cells=int((samples>threshold).sum()),null_test_cells=n,
                               null_test_false_cells=false,achieved_pfa=achieved,fixed_threshold_pfa=fixed_false/n,
                               null_test_frames=s.null_test_frames,pfa_control_status=status,
                               adc_calibration_clipped_components=adc_cal_clips[codec.name])
                    calibration.append(row)
                    calibration_by_key[(fmt,snr,codec.name,pfa)] = row
                    for i, v in enumerate(test_scores[codec.name]):
                        null_rows.append(dict(format=fmt,snr_db=snr,codec=codec.name,target_pfa=pfa,
                                              seed=radar.seed(2,si*1000000+i),eligible_cells=len(v),
                                              false_cells=int((v>threshold).sum()),fixed_threshold_false_cells=int((v>base_threshold).sum())))
            for ri, rcs in enumerate(s.rcs):
                for i in range(s.trials):
                    index = si*1000000+ri*10000+i
                    x, meta = radar.scene(snr,rcs,3,index)
                    block, adc_info = radar.reference(x,fmt,meta['noise_sigma'])
                    scene_id = f's{si}_r{ri}_t{i}'
                    if fmt == s.formats[0]:
                        scenes.append(dict(scene_id=scene_id,**meta))
                        if save_dataset:
                            np.save(output/'dataset'/f'{scene_id}.npy',x,allow_pickle=False)
                    for codec in codecs:
                        blob,y,enc_ms,dec_ms = timed_roundtrip(codec,block,s.timing_repeats)
                        metrics = reconstruction_metrics(block,y,s.phase_floor_noise_sigmas*meta['noise_sigma'])
                        if i == 0:
                            (output/'streams'/f'{scene_id}_{fmt}_{codec.name}.rbc').write_bytes(blob)
                            if codec.decode((output/'streams'/f'{scene_id}_{fmt}_{codec.name}.rbc').read_bytes()).data.tobytes() != y.data.tobytes():
                                raise AssertionError('Stored stream differs from measured reconstruction')
                        preferred_bits = 4 if 4 in s.baq_bits else min(s.baq_bits,default=4)
                        if codec.method == 'baq' and (example is None or
                            (snr == s.snr_db[-1] and rcs == s.rcs[-1] and i == 0 and codec.bits == preferred_bits)):
                            example = (block.iq(),y.iq(),fmt,codec.name)
                        for pfa in s.target_pfa:
                            cal = calibration_by_key[(fmt,snr,codec.name,pfa)]
                            original_dets = radar.detect(block.iq(),cal['threshold'])
                            restored_dets = radar.detect(y.iq(),cal['threshold'])
                            fixed_dets = radar.detect(y.iq(),cal['reference_threshold'])
                            hits, unmatched = radar.match(restored_dets,meta['truth'])
                            orig_hits, _ = radar.match(original_dets,meta['truth'])
                            fixed_hits, _ = radar.match(fixed_dets,meta['truth'])
                            if codec.lossless and original_dets != restored_dets:
                                raise AssertionError('Lossless downstream output differs')
                            results.append(dict(config_sha256=config_hash,scene_id=scene_id,seed=meta['seed'],format=fmt,snr_db=snr,
                                                actual_occupied_snr_db=meta['actual_occupied_snr_db'],rcs=rcs,codec=codec.name,
                                                target_pfa=pfa,threshold=cal['threshold'],raw_sample_bytes=block.data.nbytes,
                                                encoded_bytes=len(blob),compression_ratio=block.data.nbytes/len(blob),
                                                bits_per_complex_sample=8*len(blob)/block.count,encode_ms=enc_ms,decode_ms=dec_ms,
                                                **metrics,**adc_info,targets=len(meta['truth']),matched_targets=hits,
                                                original_matched_targets_same_threshold=orig_hits,fixed_threshold_matched_targets=fixed_hits,
                                                detections=len(restored_dets),unmatched_detections=unmatched,
                                                achieved_pfa=cal['achieved_pfa'],pfa_control_status=cal['pfa_control_status']))
                print(f'Completed format={fmt} SNR={snr:g} RCS={rcs:g}',flush=True)
    summary = summarize(results)
    write_csv(output/'results.csv',results)
    write_csv(output/'summary.csv',summary)
    write_csv(output/'calibration.csv',calibration)
    write_csv(output/'null_trials.csv',null_rows)
    (output/'scenes.jsonl').write_text(''.join(json.dumps(m,allow_nan=False)+'\n' for m in scenes),encoding='utf8')
    from .plots import plot_results
    plot_results(summary,radar,example,output)
    return results, summary


def summarize(results):
    groups = {}
    for row in results:
        key = tuple(row[k] for k in ('format','snr_db','rcs','codec','target_pfa'))
        groups.setdefault(key,[]).append(row)
    summary = []
    for key,rows in groups.items():
        n=sum(r['targets'] for r in rows)
        hits=sum(r['matched_targets'] for r in rows)
        low,high=wilson(hits,n)
        summary.append(dict(zip(('format','snr_db','rcs','codec','target_pfa'),key))|
                       dict(trials=len(rows),matched_targets=hits,pd=hits/n,pd_ci95_low=low,pd_ci95_high=high,
                            original_pd_same_threshold=sum(r['original_matched_targets_same_threshold'] for r in rows)/n,
                            fixed_threshold_pd=sum(r['fixed_threshold_matched_targets'] for r in rows)/n,
                            compression_ratio=sum(r['raw_sample_bytes'] for r in rows)/sum(r['encoded_bytes'] for r in rows),
                            bits_per_complex_sample=float(np.mean([r['bits_per_complex_sample'] for r in rows])),
                            mean_sqnr_db=float(np.mean([r['sqnr_db'] for r in rows])),mean_mse=float(np.mean([r['mse'] for r in rows])),
                            mean_encode_ms=float(np.mean([r['encode_ms'] for r in rows])),mean_decode_ms=float(np.mean([r['decode_ms'] for r in rows])),
                            achieved_pfa=rows[0]['achieved_pfa'],pfa_control_status=rows[0]['pfa_control_status']))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New output directory (never overwritten)')
    parser.add_argument('--save-dataset',action='store_true')
    args=parser.parse_args()
    run(Settings(**json.loads(args.config.read_text(encoding='utf8'))),args.output,args.save_dataset)


if __name__ == '__main__':
    main()
