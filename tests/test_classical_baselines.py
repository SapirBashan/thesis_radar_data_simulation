from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from radar_baselines.codecs import SampleBlock, Codec, decode
from radar_baselines.metrics import reconstruction_metrics
from radar_baselines.radar import Settings, SceneGenerator
from radar_baselines.experiment import run


@pytest.mark.parametrize('method',['raw','zstd','predictive_zstd'])
@pytest.mark.parametrize('adc',[False,True])
def test_lossless_preserves_every_bit_including_special_floats(method,adc):
    if adc:
        data=np.array([-32768,-1,0,1,32767,2,9,-11],dtype='<i2').reshape(1,2,2,2)
        block=SampleBlock(data,1e-8)
    else:
        words=np.array([0,0x80000000,0x7f800000,0xff800000,0x7fc01234,0xffc05678,0x3f800000,1],dtype='<u4')
        block=SampleBlock(words.view('<c8').reshape(1,2,2))
    restored=decode(Codec(method).encode(block))
    assert restored.data.tobytes()==block.data.tobytes()
    assert restored.adc_step==block.adc_step


@pytest.mark.parametrize('bits',[2,3,4,6,8,12])
@pytest.mark.parametrize('adc',[False,True])
def test_baq_error_bound_partial_block_and_stored_metadata(bits,adc):
    rng=np.random.default_rng(4)
    x=(rng.normal(size=(2,3,7))+1j*rng.normal(size=(2,3,7))).astype('<c8')
    block=SampleBlock(np.stack([np.rint(x.real*100),np.rint(x.imag*100)],axis=-1).astype('<i2'),.01) if adc else SampleBlock(x)
    codec=Codec('baq',bits=bits,block_complex_samples=5)
    blob=codec.encode(block)
    y=decode(blob)
    a=block.data.reshape(-1) if adc else block.data.view('<f4').reshape(-1)
    b=y.data.reshape(-1) if adc else y.data.view('<f4').reshape(-1)
    for i in range(0,len(a),10):
        scale=np.max(np.abs(a[i:i+10].astype(float)))/((1<<(bits-1))-1)
        assert np.max(np.abs(a[i:i+10].astype(float)-b[i:i+10]))<=scale/2+(.51 if adc else 1e-6)
    # Serialized size must include scales, framing and all packed sample bits.
    assert len(blob)>int(np.ceil(2*block.count*bits/8))


def test_zero_blocks_and_corruption_and_signed_zero():
    block=SampleBlock(np.zeros((1,5,3),dtype='<c8'))
    for method in ('raw','zstd','predictive_zstd','baq'):
        blob=Codec(method,block_complex_samples=4).encode(block)
        assert np.all(decode(blob).data==0)
        for damaged in (blob[:-1],blob[:8]+b'\xff'*4+blob[12:],blob[:-1]+bytes([blob[-1]^1])):
            with pytest.raises((ValueError,KeyError)):
                decode(damaged)


def test_decoder_works_in_fresh_process_without_codec_state(tmp_path):
    block=SampleBlock(np.ones((1,8,4),dtype='<c8'))
    path=tmp_path/'block.rbc'
    path.write_bytes(Codec('baq',bits=3).encode(block))
    code="from radar_baselines.codecs import decode; from pathlib import Path; import sys; x=decode(Path(sys.argv[1]).read_bytes()); assert x.data.shape==(1,8,4)"
    subprocess.run([sys.executable,'-B','-c',code,str(path)],check=True,cwd=Path(__file__).resolve().parents[1])


def test_noise_power_reproducibility_and_seed_separation():
    radar=SceneGenerator(Settings())
    a,meta=radar.scene(-10,1,2,0,target=False)
    b,_=radar.scene(-10,1,2,0,target=False)
    assert np.array_equal(a,b)
    assert len({radar.seed(split,0) for split in (1,2,3)})==3
    assert np.isclose(np.mean(np.abs(a)**2),meta['noise_sigma']**2,rtol=.06)
    assert np.isclose(np.var(a.real),np.var(a.imag),rtol=.08)


def test_calibration_scores_reproduce_upstream_cfar_cell_decisions():
    from psp import cfar_1d, dpc, dbf_bfcu_mode
    radar=SceneGenerator(Settings())
    x,_=radar.scene(-10,1,2,5,target=False)
    scores,eligible,_=radar.features(x)
    filtered=dpc(dbf_bfcu_mode(x),int(radar.s.fs*radar.s.pulse_width))
    power=np.sum(np.abs(filtered)**2,axis=1)[:,None]
    for threshold in (0.5,1.0,1.3,2):
        _,mask=cfar_1d(power,radar.s.guard_cells,radar.s.training_cells,threshold)
        assert np.array_equal(mask[:,0],(scores>threshold)&eligible)


def test_known_target_detected_at_expected_range_and_doppler():
    radar=SceneGenerator(Settings())
    x,meta=radar.scene(35,1,3,0)
    dets=radar.detect(x,2)
    hits,_=radar.match(dets,meta['truth'])
    assert hits==1
    # Check the actual closest detection, not just a permissive gate.
    expected=meta['truth'][0]['expected_detection_range_m']
    assert min(abs(d.range-expected) for d in dets)<299792458/(2*radar.s.fs)*2


def test_one_to_one_matching_does_not_count_duplicate_detections():
    from psp import Detection
    radar=SceneGenerator(Settings())
    truth=[dict(expected_detection_range_m=5000,velocity_mps=1)]
    det=Detection(5000,1,10,0,0)
    assert radar.match([det,det],truth)==(1,1)
    assert radar.match([Detection(5000,10,10,0,0)],truth)==(0,1)


def test_adc_clipping_and_distortion_not_hidden():
    radar=SceneGenerator(Settings())
    x=np.ones((1,2,2),dtype='<c8')*1e5
    block,info=radar.reference(x,'adc16',1)
    assert info['adc_clipped_components']==4
    assert info['adc_conversion_mse']>0
    exact=reconstruction_metrics(block,decode(Codec('zstd').encode(block)),.1)
    assert exact['mse']==0 and exact['sqnr_db']==float('inf')


def test_phase_mask_reports_zero_reconstruction_coverage():
    x=SampleBlock(np.ones((1,2,2),dtype='<c8'))
    y=SampleBlock(np.zeros((1,2,2),dtype='<c8'))
    metrics=reconstruction_metrics(x,y,.1)
    assert metrics['phase_valid_samples']==0
    assert metrics['phase_eligible_samples']==4
    assert np.isnan(metrics['phase_rmse_rad'])


def test_full_experiment_emits_real_streams_metrics_and_deterministic_values(tmp_path):
    settings=Settings(trials=2,calibration_frames=2,null_test_frames=2,snr_db=(0,),rcs=(1,),
                      target_pfa=(.01,),formats=('complex64',),baq_bits=(3,),baq_blocks=(8,),timing_repeats=1)
    first,summary=run(settings,tmp_path/'first',save_dataset=True)
    second,_=run(settings,tmp_path/'second')
    for a,b in zip(first,second):
        for key in ('seed','mse','encoded_bytes','matched_targets','threshold','bitwise_equal'):
            assert a[key]==b[key]
    for row in first:
        if not row['codec'].startswith('baq'):
            assert row['mse']==0 and row['bitwise_equal']
            assert row['matched_targets']==row['original_matched_targets_same_threshold']
    for path in (tmp_path/'first'/'streams').glob('*.rbc'):
        assert decode(path.read_bytes()).data.shape==(1,256,32)
    assert len(list((tmp_path/'first'/'dataset').glob('*.npy')))==2
    assert (tmp_path/'first'/'range_doppler_example.png').stat().st_size>1000
    assert all(r['pfa_control_status']=='insufficient_null_exposure' for r in summary)
    with pytest.raises(FileExistsError):
        run(settings,tmp_path/'first')
