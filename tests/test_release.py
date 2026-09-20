"""Release-specific regressions, on CPU/synthetic data unless explicitly stated."""
from pathlib import Path
from types import SimpleNamespace
import ast
import base64
import hashlib
import io
import json
import zipfile
import numpy as np
import pytest
import jax
import jax.numpy as jnp
from safetensors.numpy import save_file
import soundfile as sf
from test_core import SPEC, fixture_weights
from breezesprint25.core import load_hf_weights, JaxWhisper, DecodeSpec
from breezesprint25.loading import load_weights
from breezesprint25.engine import ResidentWhisper
from breezesprint25 import frontend as f
from breezesprint25 import pipeline as p
from breezesprint25.colab import Session, environment_matches, tail

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('source_dtype',[np.float16,np.float32])
@pytest.mark.parametrize('target_dtype',['float32','bfloat16'])
def test_host_packed_weights_are_exactly_baseline(source_dtype,target_dtype,tmp_path):
    weights={k:v.astype(source_dtype) for k,v in fixture_weights().items()}
    path=tmp_path/'weights.safetensors';save_file(weights,path)
    device=jax.devices('cpu')[0]
    baseline=load_hf_weights(str(path),SPEC,device,target_dtype)
    packed=load_weights(str(path),SPEC,device,target_dtype)
    assert jax.tree.structure(baseline)==jax.tree.structure(packed)
    for left,right in zip(jax.tree.leaves(baseline),jax.tree.leaves(packed)):
        assert left.dtype==right.dtype and left.shape==right.shape
        np.testing.assert_array_equal(np.asarray(left).astype(np.float32),np.asarray(right).astype(np.float32))

@pytest.mark.parametrize('corruption',['missing','extra','shape','nan','untied'])
def test_fast_loader_fails_closed(corruption,tmp_path):
    weights=fixture_weights();key='model.encoder.conv1.weight'
    if corruption=='missing':weights.pop(key)
    elif corruption=='extra':weights['surprise.weight']=np.ones(1,np.float32)
    elif corruption=='shape':weights[key]=weights[key][1:]
    elif corruption=='nan':weights[key][0,0,0]=np.nan
    else:weights['proj_out.weight']=weights['model.decoder.embed_tokens.weight']+1
    path=tmp_path/'bad.safetensors';save_file(weights,path)
    with pytest.raises(ValueError):load_weights(str(path),SPEC,jax.devices('cpu')[0])


def codec_fixture():
    ds=DecodeSpec(eos=40,no_speech=41,timestamp_begin=50,no_timestamps=49)
    return SimpleNamespace(prefix=lambda lang:np.array([[42,43,44]],np.int32),
                           ds=ds,suppress=np.zeros(SPEC.vocab,bool),
                           begin=np.zeros(SPEC.vocab,bool))


def test_resident_controls_and_host_mel_cast_match_baseline(tmp_path):
    path=tmp_path/'model.safetensors';save_file(fixture_weights(),path)
    params=load_weights(str(path),SPEC,jax.devices('cpu')[0])
    codec=codec_fixture()
    old=JaxWhisper(params,SPEC,jax.devices('cpu')[0])
    new=ResidentWhisper(params,SPEC,jax.devices('cpu')[0],codec)
    mel=np.random.default_rng(5).normal(size=(1,SPEC.mel_bins,2*SPEC.audio_ctx)).astype(np.float32)
    with jax.default_matmul_precision('default'):
        a=old.encode(mel);b=new.encode(mel)
        np.testing.assert_array_equal(np.asarray(a).astype(np.float32),np.asarray(b).astype(np.float32))
        args=(codec.prefix('en'),codec.suppress,codec.begin,np.array([63],np.int32),codec.ds)
        x=old.generate(a,*args);y=new.generate(b,*args)
        for key in x:np.testing.assert_array_equal(x[key],y[key])
    with pytest.raises(ValueError,match='Chinese'):
        new.generate(b,np.array([[42,42,44]],np.int32),*args[1:])

@pytest.mark.parametrize('kind',['TPU v5 lite','TPU v5p','TPU v6e'])
def test_single_supported_device_gate(kind,monkeypatch):
    device=SimpleNamespace(platform='tpu',device_kind=kind)
    monkeypatch.setattr(f.jax,'devices',lambda backend:[device])
    monkeypatch.setattr(f.jax,'process_count',lambda:1)
    assert f.select_tpu() is device

@pytest.mark.parametrize('kind,platform,count',[('TPU v2','tpu',1),('TPU v5 lite','cpu',1),('TPU v5 lite','tpu',2)])
def test_unsupported_topology_rejected(kind,platform,count,monkeypatch):
    monkeypatch.setattr(f.jax,'devices',lambda backend:[SimpleNamespace(platform=platform,device_kind=kind)]*count)
    monkeypatch.setattr(f.jax,'process_count',lambda:1)
    with pytest.raises(RuntimeError):f.select_tpu()


def test_adjacent_repetition_preserves_words_and_numbers():
    segments=[dict(text=' Order 12 chicken nuggets, please.'),dict(text='Order 12 chicken nuggets, please.')]
    before=[s['text'] for s in segments]
    previous,flag=p.mark_repetitions(segments)
    assert flag and [s['text'] for s in segments]==before
    assert '12' in previous
    assert segments[1]['review_flags']==['adjacent_repetition_review']


def test_cross_window_repeat_is_only_a_flag():
    previous=p.normalized_words('That is all. Thank you.')
    segments=[dict(text='That is all. Thank you.')]
    assert p.mark_repetitions(segments,previous)[1]
    assert len(segments)==1


def manifest():
    return dict(model={'id':'synthetic'},dtype='bfloat16',source_sha256={},packages={},
                ffmpeg='test',device_kind='CPU tests only',decode={})


def test_silence_full_file_resume_and_final_export_timing(tmp_path):
    path=tmp_path/'silence.wav';sf.write(path,np.zeros(31*f.SR,np.float32),f.SR)
    work=tmp_path/'work';out=tmp_path/'results'
    first=p.transcribe_many([path],None,None,None,work,out,manifest())
    dest=Path(first['outputs'][0]['directory'])
    result=json.loads((dest/'transcript.json').read_text())
    assert result['complete'] and result['processed_samples']==31*f.SR
    assert result['text']=='' and not result['manifest']['reference_verbatim_checked']
    assert first['batch_rtfx']>0
    stat=json.loads((dest/'timing.json').read_text())
    assert stat['file_processing_seconds']>=stat['inference_and_artifact_export_seconds']>0
    second=p.transcribe_many([path],None,None,None,work,out,manifest())
    assert second['outputs'][0]['directory']==str(dest)
    assert second['batch_rtfx'] is None
    assert second['outputs'][0]['timing']['rtfx'] is None
    assert len((dest/'progress.jsonl').read_text().splitlines())==2


def test_prefetch_cleans_current_and_next_on_failure(tmp_path,monkeypatch):
    paths=[]
    for i in range(2):
        path=tmp_path/f'{i}.wav';sf.write(path,np.ones(f.SR,np.float32)*.1,f.SR);paths.append(path)
    def fail(*args,**kwargs):raise RuntimeError('synthetic decode failure')
    monkeypatch.setattr(f,'run_window',fail)
    with pytest.raises(RuntimeError,match='synthetic'):
        p.transcribe_many(paths,None,None,None,tmp_path/'work',tmp_path/'out',manifest())
    assert not list((tmp_path/'work/temp').iterdir())
    assert list((tmp_path/'out').glob('*/transcript.partial.json'))
    assert not list((tmp_path/'out').glob('*/transcript.json'))


def test_controller_surfaces_worker_failure(tmp_path):
    session=Session.__new__(Session);session.directory=tmp_path;session.log=tmp_path/'worker.log'
    session.log.write_text('Traceback: TPU initialization failed')
    (tmp_path/'status.json').write_text(json.dumps(dict(state='failed',stage='TPU init',message='no device')))
    session.process=SimpleNamespace(poll=lambda:1)
    with pytest.raises(RuntimeError,match='TPU init'):session._ensure_alive()


def test_diagnostic_tail_is_bounded(tmp_path):
    log=tmp_path/'log';log.write_text('x'*100000+'\nFINAL FAILURE\n')
    result=tail(log)
    assert len(result)<=7000 and result.endswith('FINAL FAILURE')


def test_no_accidental_runtime_cpu_install(tmp_path):
    assert not environment_matches(tmp_path/'absent-python',ROOT/'requirements-tpu.lock')


def test_notebook_has_exactly_two_clean_cells_and_matching_sources():
    import nbformat
    nb=nbformat.read(ROOT/'BreezeSprint25.ipynb',as_version=4)
    nbformat.validate(nb)
    assert len(nb.cells)==2 and all(c.cell_type=='code' for c in nb.cells)
    assert all(c.outputs==[] and c.execution_count is None for c in nb.cells)
    values={}
    for c in nb.cells:ast.parse(c.source)
    for node in ast.parse(nb.cells[0].source).body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in {'PAYLOAD_B64','PAYLOAD_SHA256','PAYLOAD_FILES'}:
                values[node.targets[0].id]=ast.literal_eval(node.value)
    raw=base64.b64decode(values['PAYLOAD_B64'],validate=True)
    assert hashlib.sha256(raw).hexdigest()==values['PAYLOAD_SHA256']
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert sorted(z.namelist())==sorted(values['PAYLOAD_FILES'])
        for name in z.namelist():assert z.read(name)==(ROOT/name).read_bytes()


def test_core_is_identical_to_successful_user_runtime():
    actual=hashlib.sha256((ROOT/'src/breezesprint25/core.py').read_bytes()).hexdigest()
    assert actual=='8f1838a6457ebf952fdf6cbf6260662120beaf26c2db6d119a69630ed0332769'


def test_only_selected_checkpoint_exposed():
    model=json.loads((ROOT/'src/breezesprint25/model.json').read_text())
    assert list(f.REGISTRY)==[model['preset']]
    assert model['model_id'].startswith('MediaTek-Research/Breeze-ASR-')
    assert len(model['revision'])==40 and model['language_token']=='zh'
    assert all(name.endswith('.safetensors') for name in model['weight_files'])


def test_notebook_controller_has_no_heavy_imports():
    tree=ast.parse((ROOT/'src/breezesprint25/colab.py').read_text())
    modules=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):modules.extend(x.name.split('.')[0] for x in node.names)
        elif isinstance(node,ast.ImportFrom):modules.append((node.module or '').split('.')[0])
    assert not set(modules)&{'jax','torch','flax','transformers','tensorflow','numpy'}
