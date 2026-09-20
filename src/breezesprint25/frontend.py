"""Colab TPU-v5 Whisper application. See README and validation report.

Standard-library + NumPy/SciPy frontend, SafeTensors loader, JAX inference.
TPU selection is fail-closed; this program never falls back to CPU ASR.
"""
from __future__ import annotations
import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zlib
from typing import Any

import numpy as np
from scipy.fft import rfft
from .core import ModelSpec, DecodeSpec, JaxWhisper, load_hf_weights
from .text_output import atomic_text, timecode, timestamped_text
import jax
import jax.numpy as jnp

SR = 16000
N_SAMPLES = 30 * SR
HOP = 160
N_FFT = 400
from .assets import profile
REGISTRY = {profile()['preset']: (profile()['model_id'], profile()['revision'])}



def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def package_versions():
    result={}
    for name in ['jax','jaxlib','libtpu','numpy','scipy','ml_dtypes','opt_einsum','safetensors','tokenizers','huggingface-hub']:
        try:result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result[name]=None
    return result


def select_tpu(strict_v5=False):
    try:
        devices=jax.devices('tpu')
    except Exception as exc:
        raise RuntimeError('TPU initialization failed. Select a Colab TPU runtime. No CPU fallback.\n'+str(exc)) from exc
    if not devices or any(d.platform!='tpu' for d in devices):
        raise RuntimeError('No TPU available. CPU ASR fallback is disabled.')
    if jax.process_count()!=1:
        raise RuntimeError('Only one local TPU host is supported.')
    target=devices[0]
    if strict_v5 and 'v5' not in target.device_kind.lower():
        raise RuntimeError(f'Expected v5, received {target.device_kind}. Select TPU v5e-1.')
    if len(devices) != 1 or not re.search(r'v[56]', target.device_kind.lower()):
        raise RuntimeError('This release requires one v5/v6 TPU device; no CPU fallback.')
    print(f'TPU: {target.device_kind}; visible={len(devices)}; selected={target}',flush=True)
    return target


def hardware_probe(device):
    @jax.jit
    def operation(a):return a @ a.T
    a=jax.device_put(np.eye(128,dtype=np.float32),device)
    start=time.perf_counter();y=operation(a);y.block_until_ready();elapsed=time.perf_counter()-start
    np.testing.assert_array_equal(jax.device_get(y),np.eye(128,dtype=np.float32))
    if any(d.platform!='tpu' for d in y.devices()):raise RuntimeError('TPU probe escaped to CPU.')
    return {'device_kind':device.device_kind,'platform':device.platform,
            'device_id':int(device.id),'local_device_count':jax.local_device_count(),
            'process_count':jax.process_count(),'probe_passed':True,'probe_cold_seconds':elapsed,
            'python':sys.version,'packages':package_versions()}


def mel_filters(n_mels):
    """Librosa-compatible Slaney mel scale + area normalization (no librosa dep)."""
    def hz_to_mel(f):
        f=np.asarray(f,dtype=np.float64)
        return np.where(f>=1000,15+np.log(np.maximum(f,1e-30)/1000)/(np.log(6.4)/27),f/(200/3))
    def mel_to_hz(m):
        return np.where(m>=15,1000*np.exp((m-15)*(np.log(6.4)/27)),m*(200/3))
    edges=mel_to_hz(np.linspace(hz_to_mel(0),hz_to_mel(SR/2),n_mels+2))
    fft_freqs=np.linspace(0,SR/2,1+N_FFT//2)
    widths=np.diff(edges)
    ramps=edges[:,None]-fft_freqs[None,:]
    lower=-ramps[:-2]/widths[:-1,None]
    upper=ramps[2:]/widths[1:,None]
    weights=np.maximum(0,np.minimum(lower,upper)).astype(np.float32)
    weights *= (2/(edges[2:]-edges[:-2]))[:,None]
    return weights


def log_mel(audio: np.ndarray, n_mels: int, filters=None):
    """16k mono, zero-pad to 30 s; 400-point periodic-Hann STFT, hop 160."""
    a=np.asarray(audio,dtype=np.float32)
    if a.ndim!=1 or not (0 < len(a) <= N_SAMPLES):
        raise ValueError('Audio window must be nonempty mono and <=30 seconds.')
    if not np.isfinite(a).all():raise ValueError('Nonfinite audio samples; stopping.')
    padded=np.zeros(N_SAMPLES,dtype=np.float32);padded[:len(a)]=a
    padded=np.pad(padded,(N_FFT//2,N_FFT//2),mode='reflect')
    frames=np.lib.stride_tricks.sliding_window_view(padded,N_FFT)[::HOP][:-1]
    window=np.hanning(N_FFT+1)[:-1].astype(np.float32)
    spectrum=rfft(frames*window,n=N_FFT,axis=-1,workers=1)
    power=(spectrum.real*spectrum.real+spectrum.imag*spectrum.imag).T
    filters=mel_filters(n_mels) if filters is None else filters
    mel=filters @ power
    logged=np.log10(np.maximum(mel,np.float32(1e-10)))
    logged=np.maximum(logged,logged.max()-8.0)
    out=((logged+4.0)/4.0).astype(np.float32)
    if out.shape!=(n_mels,3000) or not np.isfinite(out).all():raise RuntimeError('Invalid mel output.')
    return out


@contextlib.contextmanager
def decoded_audio(path: Path, work: Path):
    """Decode to a disk-backed float32 file, never an entire long audio in RAM."""
    if not path.is_file():raise FileNotFoundError(path)
    if not shutil.which('ffmpeg'):raise RuntimeError('FFmpeg is not installed.')
    work.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(work).free < 512*1024**2:raise RuntimeError('Less than 512 MiB of scratch disk remains.')
    with tempfile.TemporaryDirectory(prefix='pcm_',dir=work) as tmp:
        dest=Path(tmp)/'audio.f32';log=Path(tmp)/'ffmpeg.log'
        # Match OpenAI Whisper's PCM16 -> float32 / 32768 decoding convention.
        # Stream to disk instead of capture_output() on a multi-hour recording.
        command=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-protocol_whitelist','file,pipe','-i',str(path),
                 '-map','0:a:0','-vn','-ac','1','-ar',str(SR),'-c:a','pcm_s16le','-f','s16le','-']
        with log.open('w') as errors, dest.open('wb') as output:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=errors)
            carry=b''
            try:
                while True:
                    block=process.stdout.read(1024*1024)
                    if not block:break
                    data=carry+block;even=len(data)-(len(data)%2)
                    carry=data[even:]
                    values=np.frombuffer(data[:even],dtype='<i2').astype('<f4')/32768.0
                    output.write(values.tobytes())
                returncode=process.wait()
                if carry:raise RuntimeError('Incomplete PCM16 sample from FFmpeg.')
            except BaseException:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
            finally:process.stdout.close()
        if returncode:
            raise RuntimeError('FFmpeg decoding failed: '+log.read_text(errors='replace')[-5000:])
        if not dest.exists() or dest.stat().st_size==0 or dest.stat().st_size%4:
            raise ValueError('Audio decoder returned empty or incomplete PCM.')
        audio=np.memmap(dest,dtype='<f4',mode='r')
        try:yield audio
        finally:
            # Avoid leaving the mmap open while TemporaryDirectory removes its file.
            audio._mmap.close()


from .assets import fetch_snapshot


class Codec:
    def __init__(self, tokenizer_path: Path, generation: dict, spec: ModelSpec, added_tokens_path=None):
        from .vocabulary import DecoderVocabulary, resolve_no_speech_token
        self.tok=DecoderVocabulary(tokenizer_path, added_tokens_path);self.gen=generation;self.spec=spec
        if set(self.tok.inverse) != set(range(spec.vocab)):
            raise ValueError('Vocabulary IDs must cover exactly the model output vocabulary.')
        def token(s):
            t=self.tok.token_to_id(s)
            if t is None or not 0<=t<spec.vocab:raise ValueError('Missing special token: '+s)
            return t
        self.sot=token('<|startoftranscript|>');self.eos=token('<|endoftext|>')
        self.task=token('<|transcribe|>')
        self.no_speech_token,self.no_speech=resolve_no_speech_token(self.tok.token_to_id,spec.vocab)
        self.no_ts=token('<|notimestamps|>');self.ts_begin=token('<|0.00|>')
        if token('<|30.00|>')!=self.ts_begin+1500 or self.ts_begin+1500>=spec.vocab:
            raise ValueError('Unexpected timestamp vocabulary.')
        for i in range(1501):
            if token(f'<|{i*0.02:.2f}|>') != self.ts_begin+i:
                raise ValueError('Noncontiguous timestamp vocabulary.')
        if not self.eos < self.no_speech < self.ts_begin:
            raise ValueError('No-speech token must be a control token, not text, EOS or a timestamp.')
        # Cross-check configured IDs when supplied; the vocabulary remains authoritative.
        expected_ids = {'decoder_start_token_id': self.sot, 'eos_token_id': self.eos,
                        'no_timestamps_token_id': self.no_ts, 'no_speech_token_id': self.no_speech}
        for key, expected_id in expected_ids.items():
            actual = generation.get(key)
            if actual is not None and (type(actual) is not int or actual != expected_id):
                raise ValueError('Generation config conflicts with vocabulary: ' + key)
        task_ids = generation.get('task_to_id') or {}
        if 'transcribe' in task_ids and (type(task_ids['transcribe']) is not int or task_ids['transcribe'] != self.task):
            raise ValueError('Generation config conflicts with vocabulary: transcribe')
        for name, ident in (generation.get('lang_to_id') or {}).items():
            if type(ident) is not int or self.tok.token_to_id(name) != ident:
                raise ValueError('Generation config conflicts with vocabulary: language ' + name)
        self.ds=DecodeSpec(self.eos,self.no_speech,self.ts_begin,self.no_ts,True)
        self.suppress=np.zeros(spec.vocab,bool)
        for item in generation.get('suppress_tokens',[]) or []:
            if type(item) is not int or not 0<=item<spec.vocab:raise ValueError('Invalid suppression config.')
            self.suppress[item]=True
        # All control tokens other than EOS are disallowed after the fixed prefix.
        self.suppress[self.eos+1:self.ts_begin]=True
        self.suppress[self.eos]=False
        self.begin=np.zeros(spec.vocab,bool)
        self.begin[self.eos]=True
        for item in self.tok.encode(' ',add_special_tokens=False).ids:
            self.begin[item]=True
    def prefix(self,language):
        if not re.fullmatch('[a-z]{2,3}',language):raise ValueError('An explicit ISO language code is required.')
        t=self.tok.token_to_id('<|'+language+'|>')
        lang_values=set((self.gen.get('lang_to_id') or {}).values())
        if t is None or (lang_values and t not in lang_values):raise ValueError('Unknown model language: '+language)
        return np.array([[self.sot,t,self.task]],np.int32)
    def text(self,tokens):
        ordinary=[int(t) for t in tokens if 0<=int(t)<self.eos]
        return self.tok.decode(ordinary,skip_special_tokens=True)


def split_timestamp_segments(tokens, codec, valid_samples: int, truncated=False):
    """Return closed segments and the next seek offset without losing an open tail.

    Closed-single-timestamp ending => consume the window (Whisper convention).
    An unfinished tail => commit closed spans, re-decode from their last end.
    No closing timestamps => coarse fallback, explicitly flagged; never fake words.
    """
    tokens=[int(t) for t in tokens]
    if codec.eos in tokens:tokens=tokens[:tokens.index(codec.eos)]
    duration=valid_samples/SR;start=None;pending=[];segments=[];flags=[]
    for token in tokens:
        if token>=codec.ts_begin:
            stamp=min(duration,max(0,(token-codec.ts_begin)*0.02))
            if start is not None and pending:
                if stamp<=start:
                    raise RuntimeError('Nonpositive timestamp span; refusing an invalid subtitle.')
                segments.append({'start':start,'end':stamp,'text':codec.text(pending),'tokens':list(pending),
                                 'timing':'model_timestamp'})
            start=stamp;pending=[]
        elif token<codec.eos:
            pending.append(token)
    single_end=len(tokens)>=2 and tokens[-1]>=codec.ts_begin and tokens[-2]<codec.ts_begin
    if single_end and not truncated:
        advance=valid_samples
    elif segments:
        advance=min(valid_samples,int(round(segments[-1]['end']*SR)))
        flags.append('tail_redecoded_from_last_closed_timestamp')
    else:
        if truncated:
            raise RuntimeError('Token budget exhausted without a closed segment; no text was silently discarded.')
        text=codec.text(tokens)
        if text.strip():
            segments=[{'start':0.0,'end':duration,'text':text,
                       'tokens':[t for t in tokens if t<codec.eos],'timing':'coarse_window_fallback'}]
            flags.append('missing_closing_timestamp_coarse_timing')
        else:flags.append('empty_transcript')
        advance=valid_samples
    if truncated:flags.append('token_budget_reached_tail_redecoded')
    if not 0<advance<=valid_samples:raise RuntimeError('Invalid/nonprogressing audio seek.')
    return segments,advance,flags


def load_journal(path: Path, signature: str, total_samples: int):
    if not path.exists():return []
    raw=path.read_bytes();cut=raw.rfind(b'\n')+1
    # An interrupted final append is recoverable; a malformed complete line is not.
    if cut<len(raw):
        path.with_suffix('.interrupted-tail').write_bytes(raw[cut:])
        with path.open('r+b') as f:f.truncate(cut);f.flush();os.fsync(f.fileno())
        raw=raw[:cut]
    rows=[];seek=0
    for line in raw.splitlines():
        row=json.loads(line)
        if row.get('signature')!=signature or row.get('seek_samples')!=seek:
            raise RuntimeError('Journal signature/order mismatch; not resuming stale output.')
        advance=row.get('advance_samples')
        if not isinstance(advance,int) or advance<=0 or seek+advance>total_samples:
            raise RuntimeError('Invalid journal seek; refusing to skip audio.')
        seek+=advance;rows.append(row)
    return rows


def render_outputs(outdir: Path, rows, manifest, complete: bool):
    """Write one user-facing timestamped TXT; retain JSON only for local recovery."""
    outdir.mkdir(parents=True, exist_ok=True)
    basename = 'transcript' if complete else 'transcript.partial'
    segments = [s for row in rows for s in row['segments']]
    # Validate before writing; keep raw model text separately for audits.
    formatted = timestamped_text(segments)
    text = ''.join(s['text'] for s in segments)
    result = {'complete': complete, 'text': text, 'segments': segments,
              'text_file_format': 'srt-style-timestamped-txt-v1',
              'review_windows': [{'window': r['index'], 'start': r['seek_samples']/SR,
                                  'flags': r['flags']} for r in rows if r['flags']],
              'manifest': manifest,
              'processed_samples': sum(r['advance_samples'] for r in rows)}
    atomic_json(outdir/(basename+'.json'), result)
    atomic_text(outdir/(basename+'.txt'), formatted)
    atomic_json(outdir/'manifest.json', dict(manifest, complete=complete))
    # Remove any old subtitle exports in a reused folder; they are not this API.
    for suffix in ('.srt', '.vtt', '.partial.srt', '.partial.vtt'):
        (outdir/('transcript'+suffix)).unlink(missing_ok=True)
    opposite = 'transcript.partial' if complete else 'transcript'
    for suffix in ('.txt', '.json'):
        (outdir/(opposite+suffix)).unlink(missing_ok=True)
    return result


def run_window(engine,codec,filters,audio,language):
    start=time.perf_counter();features=log_mel(audio,engine.spec.mel_bins,filters)[None,:,:]
    feature_time=time.perf_counter()-start
    start=time.perf_counter();encoded=engine.encode(features);encoder_time=time.perf_counter()-start
    max_index=min(1500,max(1,math.ceil(len(audio)/(SR*.02))))
    start=time.perf_counter()
    result=engine.generate(encoded,codec.prefix(language),codec.suppress,codec.begin,
                           np.array([codec.ts_begin+max_index],np.int32),codec.ds)
    decoder_time=time.perf_counter()-start
    prefix_len=int(result['prefix_len']);length=int(result['lengths'][0])
    tokens=result['sequences'][0,prefix_len:prefix_len+length].tolist()
    segments,advance,flags=split_timestamp_segments(tokens,codec,len(audio),bool(result['truncated'][0]))
    avg=float(result['avg_logprob'][0]);no_speech=float(result['no_speech_prob'][0])
    text=codec.text(tokens);raw=text.encode('utf-8')
    compression=len(raw)/max(1,len(zlib.compress(raw)))
    if avg < -1.0:flags.append('low_logprob_review')
    if no_speech>.6:flags.append('high_no_speech_probability_review_not_automatically_deleted')
    if compression>2.4:flags.append('repetitive_text_review')
    return {'raw_tokens':tokens,'raw_window_text':text,'segments':segments,'advance_samples':advance,
            'flags':flags,'avg_logprob':avg,'no_speech_prob':no_speech,'compression_ratio':compression,
            'truncated':bool(result['truncated'][0]),'feature_seconds':feature_time,
            'encoder_seconds':encoder_time,'decoder_seconds':decoder_time}


def checked_window(engine, codec, filters, audio, language):
    """First nonzero input runs twice: execution/repeatability, NOT a gold transcript.

    No English JFK requirement is imposed on a Taigi-specialized model. The first
    pass is returned unchanged; the second pass is not inserted into the output.
    """
    first = run_window(engine, codec, filters, audio, language)
    if getattr(engine, 'first_window_checked', True): return first
    second = run_window(engine, codec, filters, audio, language)
    equal = first['raw_tokens'] == second['raw_tokens']
    report = dict(passed=equal, repeated_tokens_equal=equal,
                  input_samples=len(audio), input_pcm_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
                  nonempty_text=bool(first['raw_window_text'].strip()),
                  first_truncated=first['truncated'], second_truncated=second['truncated'],
                  reference_verbatim_checked=False, model_accuracy_validated=False,
                  scope='First nonzero user window run twice on this worker. Same-token check only; '
                        'not an independently transcribed Mandarin/Taigi reference.',
                  first=first, second=second)
    atomic_json(engine.acceptance_report_path, report)
    if not equal: raise RuntimeError('First-window token repeatability failed; see first_window_check.json. No complete TXT is offered.')
    engine.first_window_checked = True
    engine.first_window_report = report
    return first
