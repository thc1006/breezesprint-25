"""CPU tests for DSP, seek continuity, journaling, export, and TPU gating.

These are not TPU/pretrained-ASR certification tests.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import soundfile as sf
import breezesprint25.frontend as r

class FakeCodec:
    eos=90
    ts_begin=100
    def text(self,tokens):
        return ''.join({1:' A',2:' B',3:' C'}.get(int(x),'') for x in tokens)

@pytest.mark.parametrize('mels',[80,128])
def test_frontend_against_torch_stft(mels):
    rng=np.random.default_rng(91)
    audio=(rng.normal(size=21000)*0.1).astype(np.float32)
    filters=r.mel_filters(mels)
    got=r.log_mel(audio,mels,filters)
    wave=torch.from_numpy(np.pad(audio,(0,r.N_SAMPLES-len(audio))))
    stft=torch.stft(wave,r.N_FFT,r.HOP,window=torch.hann_window(r.N_FFT),return_complex=True)
    magnitude=stft[:,:-1].abs().square()
    mel=torch.from_numpy(filters) @ magnitude
    expected=torch.clamp(mel,min=1e-10).log10()
    expected=torch.maximum(expected,expected.max()-8)
    expected=((expected+4)/4).numpy()
    error=float(np.max(np.abs(got-expected)))
    print(f'mel_{mels}_max_abs_error={error:.12g}')
    np.testing.assert_allclose(got,expected,atol=2e-5,rtol=2e-5)
    assert got.shape==(mels,3000)

@pytest.mark.parametrize('mels',[80,128])
def test_silence_features(mels):
    x=r.log_mel(np.zeros(123,np.float32),mels)
    np.testing.assert_allclose(x,np.full((mels,3000),-1.5,np.float32),atol=3e-7,rtol=0)
    assert np.all(r.mel_filters(mels)>=0)

@pytest.mark.parametrize('audio',[np.array([],np.float32),np.zeros((2,10)),np.full(20,np.nan),np.zeros(r.N_SAMPLES+1)])
def test_invalid_audio(audio):
    with pytest.raises(ValueError):r.log_mel(audio,80)


def test_real_ffmpeg_resample_and_cleanup(tmp_path):
    sr=48000;t=np.arange(sr*2)/sr
    wave=(np.sin(2*np.pi*440*t)*.2).astype(np.float32)
    path=tmp_path/'stereo.wav';sf.write(path,np.stack([wave,wave],axis=1),sr,subtype='FLOAT')
    with r.decoded_audio(path,tmp_path/'temp') as audio:
        assert isinstance(audio,np.memmap)
        assert len(audio)==32000 and np.isfinite(audio).all()
        assert .13<float(np.sqrt(np.mean(audio**2)))<.15
    assert not list((tmp_path/'temp').iterdir())

@pytest.mark.parametrize('tokens,advance,expectedflags',[
    ([100,1,150,90],r.SR*10,[]),
    ([100,1,150,150,2,90],r.SR,['tail_redecoded_from_last_closed_timestamp']),
    ([1,2,90],r.SR*10,['missing_closing_timestamp_coarse_timing']),
    ([100,90],r.SR*10,['empty_transcript']),
])
def test_timestamp_seek(tokens,advance,expectedflags):
    segments,got,flags=r.split_timestamp_segments(tokens,FakeCodec(),r.SR*10)
    assert got==advance and flags==expectedflags
    for s in segments:assert 0<=s['start']<s['end']<=10
    if 'tail_redecoded_from_last_closed_timestamp' in flags:
        assert len(segments)==1 and segments[0]['text']==' A'
    if 'missing_closing_timestamp_coarse_timing' in flags:
        assert segments[0]['timing']=='coarse_window_fallback'


def test_truncated_with_completed_segment():
    segments,advance,flags=r.split_timestamp_segments([100,1,150,150,2],FakeCodec(),r.SR*10,True)
    assert advance==r.SR and segments[0]['text']==' A'
    assert 'token_budget_reached_tail_redecoded' in flags

@pytest.mark.parametrize('tokens,truncated',[([100,1],True),([100,1,100,90],False)])
def test_timestamp_failure(tokens,truncated):
    with pytest.raises(RuntimeError):r.split_timestamp_segments(tokens,FakeCodec(),r.SR,truncated)


def test_timecode_rounding():
    assert r.timecode(59.9999)=='00:01:00,000'
    assert r.timecode(3600.001,True)=='01:00:00.001'


def test_journal_recovers_only_unfinished_line(tmp_path):
    path=tmp_path/'progress.jsonl'
    row={'signature':'s','seek_samples':0,'advance_samples':16000}
    path.write_bytes((json.dumps(row)+'\n').encode()+b'{"unfinished":')
    got=r.load_journal(path,'s',32000)
    assert got==[row] and path.read_bytes().endswith(b'\n')
    assert path.with_suffix('.interrupted-tail').read_bytes()==b'{"unfinished":'

@pytest.mark.parametrize('row',[
    {'signature':'other','seek_samples':0,'advance_samples':2},
    {'signature':'s','seek_samples':2,'advance_samples':2},
    {'signature':'s','seek_samples':0,'advance_samples':0},
    {'signature':'s','seek_samples':0,'advance_samples':33},
])
def test_bad_journal_refused(tmp_path,row):
    path=tmp_path/'progress.jsonl';path.write_text(json.dumps(row)+'\n')
    with pytest.raises(RuntimeError):r.load_journal(path,'s',32)


def test_malformed_complete_journal_not_ignored(tmp_path):
    path=tmp_path/'progress.jsonl';path.write_bytes(b'bad-json\n')
    with pytest.raises(json.JSONDecodeError):r.load_journal(path,'s',32)


def test_partial_export_is_not_final(tmp_path):
    rows=[{'segments':[{'start':0.,'end':1.,'text':' 中文測試','timing':'model_timestamp'}],
           'index':0,'seek_samples':0,'advance_samples':16000,'flags':[]}]
    result=r.render_outputs(tmp_path,rows,{'model':'synthetic'},False)
    assert result['complete'] is False and (tmp_path/'transcript.partial.txt').is_file()
    assert not (tmp_path/'transcript.txt').exists()
    result=r.render_outputs(tmp_path,rows,{'model':'synthetic'},True)
    assert result['complete'] is True and (tmp_path/'transcript.txt').is_file()
    assert not list(tmp_path.glob('*.partial.*'))
    assert '00:00:00,000 --> 00:00:01,000' in (tmp_path/'transcript.txt').read_text()
    assert not list(tmp_path.glob('*.srt')) and not list(tmp_path.glob('*.vtt'))


def test_strict_gate_rejects_non_v5(monkeypatch):
    monkeypatch.setattr(r.jax,'devices',lambda backend:[SimpleNamespace(platform='tpu',device_kind='TPU v6e')])
    monkeypatch.setattr(r.jax,'process_count',lambda:1)
    with pytest.raises(RuntimeError,match='v5'):r.select_tpu(strict_v5=True)


def test_gate_does_not_cpu_fallback(monkeypatch):
    def fail(backend):raise RuntimeError('not available')
    monkeypatch.setattr(r.jax,'devices',fail)
    with pytest.raises(RuntimeError,match='fallback'):r.select_tpu(strict_v5=True)

