"""Offline release and real subprocess IPC tests; fake worker NEVER simulates a TPU pass."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import pytest
from breezesprint25 import colab

ROOT = Path(__file__).resolve().parents[1]


def test_release_integrity():
    spec=importlib.util.spec_from_file_location('release_checker',ROOT/'tools/check_release.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.check()


def test_metadata_helper_is_dry_run():
    result=subprocess.run([sys.executable,ROOT/'tools/apply_github_metadata.py'],
                          capture_output=True,text=True,check=True)
    assert 'DRY RUN: no requests sent' in result.stdout
    assert '"apply": false' in result.stdout
    assert 'breeze-asr-' in result.stdout


def test_local_rebrand_rebuilds_links_and_payload(tmp_path):
    dest=tmp_path/'copy'
    shutil.copytree(ROOT,dest,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    (dest/'SHA256SUMS.txt').write_text('stale distribution checksums')
    subprocess.run([sys.executable,dest/'tools/rebrand.py','--owner','test-owner','--repo','test-repo'],
                   capture_output=True,text=True,check=True)
    project=json.loads((dest/'project.json').read_text())
    assert project['repository']=='test-owner/test-repo'
    assert 'test-owner/test-repo' in (dest/'README.md').read_text()
    assert 'test-owner.github.io/test-repo' in (dest/'docs/index.html').read_text()
    assert not (dest/'SHA256SUMS.txt').exists()
    subprocess.run([sys.executable,dest/'tools/check_release.py'],capture_output=True,check=True)


def test_one_process_serves_two_jobs_and_closes_cleanly(tmp_path,monkeypatch):
    """Exercise real Session/stdin/atomic response/TXT staging/close with a stdlib fake worker."""
    bundle=tmp_path/'bundle';package=bundle/'src/breezesprint25';package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'worker.py').write_text('''import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--session');a=p.parse_args()
s=Path(a.session)
def put(path,data):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(data));os.replace(t,path)
put(s/'status.json',dict(state='ready',stage='FAKE IPC TEST ONLY'))
count=0
for line in sys.stdin:
 r=json.loads(line)
 if r['op']=='close':break
 count+=1
 d=Path(r['output'])/('fake_'+str(count));d.mkdir(parents=True)
 (d/'transcript.txt').write_text('1\\n00:00:00,000 --> 00:00:01,000\\nSynthetic IPC fixture.\\n')
 output=dict(directory=str(d),review_windows=0,input_name='test.wav',segments=1,processing_complete=True)
 result=dict(outputs=[output],audio_seconds=1.,batch_wall_seconds=.1,request_count=count,processing_complete=True)
 put(s/(r['id']+'.json'),dict(ok=True,result=result))
''')
    monkeypatch.setattr(colab.LiveStatus,'show',lambda *a,**k:None)
    root=tmp_path/'runtime';session=colab.Session(root,bundle,sys.executable)
    audio=tmp_path/'input.wav';audio.write_bytes(b'Not real audio: fake worker does no ASR.')
    try:
        pid=session.process.pid
        first=session.transcribe([audio],tmp_path/'outputs')
        assert session.last_result['request_count']==1
        second=session.transcribe([audio],tmp_path/'outputs')
        assert session.last_result['request_count']==2 and session.process.pid==pid
        assert first!=second and session.process.poll() is None
        for paths in [first,second]:
            assert len(paths)==1 and paths[0].name=='test_transcript.txt'
            assert '00:00:00,000 --> 00:00:01,000' in paths[0].read_text()
        assert not list(root.rglob('*.zip'))
        assert (session.directory/'notebook_timing.json').exists()
    finally:session.close()
    assert session.process.poll()==0 and session.process.stdin.closed


def test_log_failure_does_not_hide_primary_failure(tmp_path,monkeypatch):
    session=colab.Session.__new__(colab.Session)
    session.root=tmp_path;session.log=tmp_path/'worker.log'
    def primary(ui):raise RuntimeError('PRIMARY TPU FAILURE')
    def secondary(*a,**kw):raise OSError('LOG DISK FAILURE')
    session._wait_ready=primary
    monkeypatch.setattr(colab,'tail',secondary)
    monkeypatch.setattr(colab.LiveStatus,'show',lambda *a,**k:None)
    path=tmp_path/'input';path.write_bytes(b'fixture')
    with pytest.raises(RuntimeError,match='PRIMARY TPU FAILURE'):
        session.transcribe([path])


def test_exact_duplicate_input_path_is_processed_once(tmp_path):
    import numpy as np
    import soundfile as sf
    from breezesprint25 import pipeline as p
    from test_release import manifest
    path=tmp_path/'audio.wav';sf.write(path,np.zeros(1600,dtype=np.float32),16000)
    result=p.transcribe_many([path,path],None,None,None,tmp_path/'work',tmp_path/'out',manifest())
    assert len(result['outputs'])==1
    assert result['outputs'][0]['timing']['resumed_samples']==0
