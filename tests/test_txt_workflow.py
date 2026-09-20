"""Upload-only, automatic TXT download regressions. No browser or TPU is simulated as real."""
import json
from pathlib import Path
from types import ModuleType
import sys
import pytest
from breezesprint25 import colab
from breezesprint25 import frontend as f
from breezesprint25.text_output import atomic_text, download_name, timecode, timestamped_text


def segments():
    return [dict(start=0.,end=1.25,text=' Order 12 nuggets for $3.14, please.'),
            dict(start=1.25,end=3.125,text='Order 12 nuggets for $3.14, please.'),
            dict(start=3600.,end=3601.001,text=' A --> B; Café, £5.00 — 2026!')]


def test_timed_txt_preserves_repetitions_digits_unicode_and_arrows():
    result=timestamped_text(segments())
    assert result=='1\n00:00:00,000 --> 00:00:01,250\nOrder 12 nuggets for $3.14, please.\n\n2\n00:00:01,250 --> 00:00:03,125\nOrder 12 nuggets for $3.14, please.\n\n3\n01:00:00,000 --> 01:00:01,001\nA --> B; Café, £5.00 — 2026!\n'
    assert result.count('Order 12 nuggets')==2


@pytest.mark.parametrize('start,end',[(float('nan'),1.),(0.,float('inf')),(-1.,1.),(1.,1.),(2.,1.)])
def test_invalid_timestamps_cannot_produce_a_final_txt(tmp_path,start,end):
    rows=[dict(segments=[dict(start=start,end=end,text=' text')],index=0,seek_samples=0,advance_samples=16000,flags=[])]
    with pytest.raises(RuntimeError): f.render_outputs(tmp_path,rows,{},True)
    assert not (tmp_path/'transcript.txt').exists()


def test_overlapping_segments_rejected():
    with pytest.raises(RuntimeError):timestamped_text([dict(start=0,end=2,text='a'),dict(start=1,end=3,text='b')])


def test_empty_and_whitespace_segments_do_not_invent_text():
    assert timestamped_text([])==''
    assert timestamped_text([dict(start=0,end=1,text='  ')])==''
    assert timestamped_text([dict(start=0,end=1,text='  '),dict(start=1,end=2,text='x')]).startswith('1\n')


def test_timecode_carries_and_long_recordings():
    assert timecode(3599.9999)=='01:00:00,000'
    assert timecode(25*3600+.123)=='25:00:00,123'
    with pytest.raises(ValueError):timecode(float('nan'))


def test_render_only_timed_txt_and_local_json_no_subtitles(tmp_path):
    rows=[dict(segments=segments(),index=0,seek_samples=0,advance_samples=60000000,flags=['review'])]
    for name in ['transcript.srt','transcript.vtt','transcript.partial.srt','transcript.partial.vtt']:
        (tmp_path/name).write_text('old output')
    f.render_outputs(tmp_path,rows,{},True)
    assert (tmp_path/'transcript.txt').read_text()==timestamped_text(segments())
    assert not list(tmp_path.glob('*.srt')) and not list(tmp_path.glob('*.vtt'))
    assert json.loads((tmp_path/'transcript.json').read_text())['text']==''.join(s['text'] for s in segments())


def test_atomic_text_failure_keeps_previous_file_and_cleans_temp(tmp_path,monkeypatch):
    dest=tmp_path/'transcript.txt';dest.write_text('last complete result')
    def fail(*args):raise OSError('full disk')
    from breezesprint25 import text_output
    monkeypatch.setattr(text_output.os,'replace',fail)
    with pytest.raises(OSError):atomic_text(dest,'new result')
    assert dest.read_text()=='last complete result'
    assert not (tmp_path/'transcript.txt.tmp').exists()


def test_names_are_readable_unique_and_bounded():
    used=set()
    assert download_name('Lecture.mp3',used)=='Lecture_transcript.txt'
    assert download_name('lecture.wav',used)=='lecture_transcript_2.txt'
    assert download_name('lecture.mp4',used)=='lecture_transcript_3.txt'
    name=download_name('臺'*400+'.mp3',used)
    assert len(name.encode())<255 and name.endswith('.txt')
    name=download_name('bad<>:?*|".mp3',used)
    assert all(ch not in name for ch in '<>:?*|"')


def session_fixture(tmp_path):
    session=colab.Session.__new__(colab.Session)
    session.root=tmp_path/'runtime';session.root.mkdir()
    session.directory=session.root/'session';session.directory.mkdir()
    session.log=session.directory/'worker.log';session.log.write_text('test log')
    session._ensure_alive=lambda:dict(state='ready',stage='FAKE UI TEST ONLY')
    return session


def test_stage_only_complete_txt_and_disambiguate_names(tmp_path):
    session=session_fixture(tmp_path);out=session.root/'results';items=[]
    for i,name in enumerate(['Recording.mp3','recording.wav']):
        d=out/str(i);d.mkdir(parents=True)
        (d/'transcript.txt').write_text(timestamped_text(segments()))
        (d/'transcript.json').write_text('{"private":"diagnostics stay local"}')
        items.append(dict(directory=str(d),input_name=name,processing_complete=True))
    paths=session.stage_text_downloads(dict(processing_complete=True,outputs=items),'id',out)
    assert [p.name for p in paths]==['Recording_transcript.txt','recording_transcript_2.txt']
    assert sorted(p.name for p in paths[0].parent.iterdir())==sorted(p.name for p in paths)
    assert all(p.suffix=='.txt' for p in paths)
    assert not list(session.root.rglob('*.zip'))


@pytest.mark.parametrize('kind',['incomplete-batch','incomplete-file','missing-file','escaped-directory','symlink'])
def test_download_staging_rejects_nonfinal_or_unsafe_outputs(tmp_path,kind):
    session=session_fixture(tmp_path);out=session.root/'results';out.mkdir()
    d=out/'one';d.mkdir();(d/'transcript.txt').write_text('complete text')
    result=dict(processing_complete=True,outputs=[dict(directory=str(d),input_name='test.wav',processing_complete=True)])
    if kind=='incomplete-batch':result['processing_complete']=False
    elif kind=='incomplete-file':result['outputs'][0]['processing_complete']=False
    elif kind=='missing-file':(d/'transcript.txt').unlink()
    elif kind=='escaped-directory':result['outputs'][0]['directory']=str(tmp_path)
    elif kind=='symlink':
        (d/'transcript.txt').unlink();target=tmp_path/'secret';target.write_text('not an output')
        (d/'transcript.txt').symlink_to(target)
    with pytest.raises(RuntimeError):session.stage_text_downloads(result,'id',out)


def mock_colab(monkeypatch,upload,download):
    google=ModuleType('google');google.__path__=[]
    package=ModuleType('google.colab');package.__path__=[]
    file_module=ModuleType('google.colab.files');file_module.upload=upload;file_module.download=download
    package.files=file_module;google.colab=package
    monkeypatch.setitem(sys.modules,'google',google)
    monkeypatch.setitem(sys.modules,'google.colab',package)
    monkeypatch.setitem(sys.modules,'google.colab.files',file_module)
    monkeypatch.setattr(colab.LiveStatus,'show',lambda *a,**kw:None)


def test_upload_transcribes_then_automatically_downloads_txts(tmp_path,monkeypatch):
    session=session_fixture(tmp_path);saved={};received=[];downloads=[]
    def upload():
        for name in ['one.mp3','two.wav']:Path(name).write_bytes(b'fixture')
        saved.update({'one.mp3':b'fixture','two.wav':b'fixture'})
        return saved
    results=[tmp_path/'one_transcript.txt',tmp_path/'two_transcript.txt']
    for p in results:p.write_text('1\n00:00:00,000 --> 00:00:01,000\nTest.\n')
    def transcribe(paths):
        assert all(p.is_absolute() and p.exists() for p in paths)
        received.extend(paths);return results
    session.transcribe=transcribe
    mock_colab(monkeypatch,upload,downloads.append)
    cwd=Path.cwd();got=colab.upload_and_transcribe(session)
    assert got==results and downloads==list(map(str,results))
    assert len(received)==2 and not saved and Path.cwd()==cwd
    assert not list(session.root.rglob('*.zip'))


def test_cancel_does_not_reuse_or_download_previous_audio(tmp_path,monkeypatch):
    session=session_fixture(tmp_path);session.last_text_files=[tmp_path/'old.txt']
    def never(*args):raise AssertionError('Cancellation must not transcribe or download')
    session.transcribe=never;mock_colab(monkeypatch,lambda:{},never)
    cwd=Path.cwd()
    assert colab.upload_and_transcribe(session)==[] and Path.cwd()==cwd


@pytest.mark.parametrize('name',['../escape.mp3','/tmp/escape.mp3','nested/audio.mp3','nested\\audio.mp3'])
def test_upload_rejects_paths_not_just_names(tmp_path,monkeypatch,name):
    session=session_fixture(tmp_path)
    def never(*args):raise AssertionError('Unsafe uploads must not be used')
    session.transcribe=never;mock_colab(monkeypatch,lambda:{name:b'x'},never)
    cwd=Path.cwd()
    with pytest.raises(ValueError,match='Unsafe'):colab.upload_and_transcribe(session)
    assert Path.cwd()==cwd


def test_upload_interruption_restores_working_directory(tmp_path,monkeypatch):
    session=session_fixture(tmp_path)
    def interrupt():raise KeyboardInterrupt()
    mock_colab(monkeypatch,interrupt,lambda p:pytest.fail('No download'))
    cwd=Path.cwd()
    with pytest.raises(KeyboardInterrupt):colab.upload_and_transcribe(session)
    assert Path.cwd()==cwd


def test_transcription_error_does_not_download_diagnostics(tmp_path,monkeypatch):
    session=session_fixture(tmp_path)
    def upload():Path('input.mp3').write_bytes(b'x');return {'input.mp3':b'x'}
    def fail(paths):raise RuntimeError('ACTUAL MODEL ERROR')
    session.transcribe=fail
    mock_colab(monkeypatch,upload,lambda p:pytest.fail('No failed result or log downloads'))
    with pytest.raises(RuntimeError,match='ACTUAL MODEL ERROR'):colab.upload_and_transcribe(session)
    assert not list(session.root.rglob('*.zip'))


def test_browser_download_failure_keeps_successful_txt_and_visible_error(tmp_path,monkeypatch,capsys):
    session=session_fixture(tmp_path)
    def upload():Path('input.mp3').write_bytes(b'x');return {'input.mp3':b'x'}
    result=tmp_path/'input_transcript.txt';result.write_text('valid transcript')
    session.transcribe=lambda paths:[result]
    def blocked(path):raise RuntimeError('browser rejected download')
    mock_colab(monkeypatch,upload,blocked)
    assert colab.upload_and_transcribe(session)==[result]
    assert result.read_text()=='valid transcript'
    assert 'TXT saved:' in capsys.readouterr().err
    assert 'browser rejected download' in (session.directory/'download-errors.log').read_text()


def test_upload_cell_before_setup_shows_clear_instruction():
    import json
    root=Path(__file__).resolve().parents[1]
    code=''.join(json.loads((root/'BreezeSprint25.ipynb').read_text())['cells'][1]['source'])
    with pytest.raises(RuntimeError,match='Run setup cell 1 first'):
        exec(compile(code,'upload-cell','exec'),{})


def test_runtime_and_project_version_are_consistent():
    import tomllib
    from breezesprint25 import __version__
    root=Path(__file__).resolve().parents[1]
    assert __version__==json.loads((root/'project.json').read_text())['version']=='0.1.1'
    assert tomllib.loads((root/'pyproject.toml').read_text())['project']['version']==__version__
    assert "release=__version__" in (root/'src/breezesprint25/worker.py').read_text()


def test_partial_export_failure_does_not_mask_original_model_error(tmp_path,monkeypatch):
    import numpy as np
    import soundfile as sf
    from breezesprint25 import pipeline as p
    from test_release import manifest
    path=tmp_path/'input.wav';sf.write(path,np.full(1600,.1,np.float32),16000)
    original=f.render_outputs
    calls=[]
    def render(*args,**kwargs):
        calls.append(1)
        if len(calls)>1:raise OSError('SECONDARY EXPORT DISK ERROR')
        return original(*args,**kwargs)
    def fail(*args,**kwargs):raise RuntimeError('PRIMARY GENERATION ERROR')
    monkeypatch.setattr(f,'render_outputs',render)
    monkeypatch.setattr(f,'run_window',fail)
    with pytest.raises(RuntimeError,match='PRIMARY GENERATION ERROR'):
        p.transcribe_many([path],None,None,None,tmp_path/'work',tmp_path/'out',manifest())
    assert len(calls)==2
