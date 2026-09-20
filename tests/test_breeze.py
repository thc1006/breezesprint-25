"""Breeze integration regressions. Synthetic fixtures; no model/TPU claim."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import Mock
import hashlib, json, sys
import numpy as np
import ml_dtypes
import pytest
import jax
from safetensors.numpy import save_file
from test_core import SPEC, fixture_weights
from breezesprint25.loading import load_weights
from breezesprint25.checkpoint import TensorStore
from breezesprint25.vocabulary import DecoderVocabulary, byte_alphabet
from breezesprint25 import assets, frontend as f, pipeline as p


def assert_tree_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for l, r in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
        assert l.dtype == r.dtype and l.shape == r.shape
        np.testing.assert_array_equal(np.asarray(l).astype('float32'), np.asarray(r).astype('float32'))


@pytest.mark.parametrize('target', ['bfloat16', 'float32'])
def test_native_bf16_safetensors_matches_exact_f32_expansion(tmp_path, target):
    # BF16 has numpy dtype.kind == 'V'; it must not be treated as non-floating.
    source = {k:v.astype(ml_dtypes.bfloat16) for k,v in fixture_weights().items()}
    assert next(iter(source.values())).dtype.kind == 'V'
    bf, fp = tmp_path/'bf.safetensors', tmp_path/'fp.safetensors'
    save_file(source, bf)
    save_file({k:v.astype(np.float32) for k,v in source.items()}, fp)
    device = jax.devices('cpu')[0]
    assert_tree_equal(load_weights(bf,SPEC,device,target), load_weights(fp,SPEC,device,target))


@pytest.mark.parametrize('source_dtype', [np.float32, ml_dtypes.bfloat16])
@pytest.mark.parametrize('target', ['bfloat16', 'float32'])
def test_two_shards_equal_single_checkpoint(tmp_path, source_dtype, target):
    source={k:v.astype(source_dtype) for k,v in fixture_weights().items()}
    full=tmp_path/'full.safetensors'; save_file(source,full)
    keys=list(source); paths=[]; ownership={}
    for i in range(2):
        path=tmp_path/f'model-0000{i+1}-of-00002.safetensors'; paths.append(path)
        shard={k:source[k] for k in keys[i::2]};save_file(shard,path)
        ownership.update({k:path.name for k in shard})
    index=tmp_path/'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map':ownership}))
    device=jax.devices('cpu')[0]
    assert_tree_equal(load_weights(paths,SPEC,device,target,index),load_weights(full,SPEC,device,target))


@pytest.mark.parametrize('bad', ['missing', 'extra', 'wrong_owner', 'traversal_owner', 'nonobject'])
def test_shard_index_must_match_exact_ownership(tmp_path,bad):
    path=tmp_path/'one.safetensors';save_file({'a':np.ones(1,np.float32)},path)
    wm={'a':path.name}
    if bad=='missing':wm={}
    elif bad=='extra':wm['b']=path.name
    elif bad=='wrong_owner':wm['a']='two.safetensors'
    elif bad=='traversal_owner':wm['a']='../one.safetensors'
    else:wm=[]
    index=tmp_path/'index.json';index.write_text(json.dumps({'weight_map':wm}))
    with pytest.raises(ValueError,match='ownership'):
        with TensorStore([path],index):pass


def test_duplicate_tensor_across_shards_is_fatal(tmp_path):
    paths=[]
    for i in range(2):
        path=tmp_path/f'{i}.safetensors';save_file({'same':np.ones(1,np.float32)},path);paths.append(path)
    with pytest.raises(ValueError,match='Duplicate tensor'):
        with TensorStore(paths):pass


def test_empty_or_duplicate_shard_names_rejected(tmp_path):
    with pytest.raises(ValueError):TensorStore([])
    with pytest.raises(ValueError):TensorStore([tmp_path/'a.safetensors',tmp_path/'a.safetensors'])


def vocab_fixture(tmp_path, timestamps=False):
    # One token per byte: tests UTF-8 sequences crossing BPE token boundaries.
    inverse={byte:char for char,byte in byte_alphabet().items()}
    vocab={inverse[i]:i for i in range(256)}
    specials=['<|endoftext|>','<|startoftranscript|>','<|zh|>','<|transcribe|>',
              '<|nospeech|>','<|notimestamps|>']
    added={token:256+i for i,token in enumerate(specials)}
    if timestamps:added.update({f'<|{i*.02:.2f}|>':262+i for i in range(1501)})
    vp=tmp_path/'vocab.json'; ap=tmp_path/'added_tokens.json'
    vp.write_text(json.dumps(vocab,ensure_ascii=False),encoding='utf8')
    ap.write_text(json.dumps(added,ensure_ascii=False),encoding='utf8')
    return vp,ap,vocab,added


def test_byte_alphabet_covers_every_byte_exactly():
    mapping=byte_alphabet()
    assert len(mapping)==256 and set(mapping.values())==set(range(256))


@pytest.mark.parametrize('text', ['台灣語音辨識。','今仔日 2026/09/21，有 12 個 projects。','a é 中 😀 0.627s', '重複。重複。'])
def test_unicode_decodes_across_token_boundaries_without_rewriting(tmp_path,text):
    vp,ap,_,_=vocab_fixture(tmp_path);v=DecoderVocabulary(vp,ap)
    assert v.decode(list(text.encode('utf8')))==text
    assert v.decode([257,*text.encode('utf8'),256])==text
    assert v.decode([257,65,256],skip_special_tokens=False)=='<|startoftranscript|>A<|endoftext|>'


def test_space_only_encoder_and_unknown_ids(tmp_path):
    vp,ap,_,_=vocab_fixture(tmp_path);v=DecoderVocabulary(vp,ap)
    assert v.encode(' ').ids==[32]
    for text in ['中文','prompt','  ']:
        with pytest.raises(ValueError,match='literal space'):v.encode(text)
    with pytest.raises(ValueError):v.encode(' ',add_special_tokens=True)
    with pytest.raises(ValueError,match='Unknown'):v.decode([99999])


@pytest.mark.parametrize('bad', ['duplicate_id','bool_id','negative','added_conflict','no_eos'])
def test_corrupt_vocab_rejected(tmp_path,bad):
    vp,ap,vocab,added=vocab_fixture(tmp_path)
    if bad=='duplicate_id':added['another']=1
    elif bad=='bool_id':added['another']=True
    elif bad=='negative':added['another']=-1
    elif bad=='added_conflict':added['A']=123
    else:added.pop('<|endoftext|>')
    ap.write_text(json.dumps(added))
    with pytest.raises(ValueError):DecoderVocabulary(vp,ap)


def test_codec_full_timestamp_range_and_fixed_zh_prefix(tmp_path):
    vp,ap,_,_=vocab_fixture(tmp_path,True)
    spec=replace(SPEC,vocab=1763)
    codec=f.Codec(vp,{'lang_to_id':{'<|zh|>':258},'suppress_tokens':[]},spec,ap)
    assert codec.prefix('zh').tolist()==[[257,258,259]]
    assert codec.text(list('台語 26'.encode()))=='台語 26'
    assert codec.ts_begin==262 and codec.begin[32]
    with pytest.raises(ValueError,match='Unknown'):codec.prefix('nan')


def test_noncontiguous_timestamp_vocabulary_is_fatal(tmp_path):
    vp,ap,_,added=vocab_fixture(tmp_path,True)
    a,b='<|5.00|>','<|5.02|>';added[a],added[b]=added[b],added[a]
    ap.write_text(json.dumps(added))
    with pytest.raises(ValueError,match='Noncontiguous'):
        f.Codec(vp,{},replace(SPEC,vocab=1763),ap)


def test_noncontiguous_model_vocabulary_is_fatal(tmp_path):
    vp,ap,_,_=vocab_fixture(tmp_path,True)
    with pytest.raises(ValueError,match='cover exactly'):
        f.Codec(vp,{},replace(SPEC,vocab=1764),ap)


@pytest.mark.parametrize('style', ['sha256','git_blob','quoted'])
def test_publisher_etag_integrity(tmp_path,style):
    data='台語 26\n'.encode();path=tmp_path/'config.json';path.write_bytes(data)
    sha=hashlib.sha256(data).hexdigest()
    etag=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() if style=='git_blob' else sha
    if style=='quoted':etag='"'+etag+'"'
    assert assets.verify_file(path,etag)==sha
    path.write_bytes(data+b'!')
    with pytest.raises(RuntimeError,match='verification failed'):assets.verify_file(path,etag)


@pytest.mark.parametrize('bad',[None,'W/"etag"','garbage','0'*39])
def test_unknown_etag_rejected(tmp_path,bad):
    path=tmp_path/'x';path.write_bytes(b'x')
    with pytest.raises(RuntimeError,match='unsupported'):assets.verify_file(path,bad)


def fake_hub(tmp_path,monkeypatch,wrong_commit=False):
    cfg=assets.profile();calls=[];source=tmp_path/'hub';source.mkdir()
    for name in cfg['files']:(source/name).write_bytes(('fixture:'+name).encode())
    def url(repo,name,revision):
        assert repo==cfg['model_id'] and revision==cfg['revision'];return name
    def metadata(name,token,timeout):
        assert token is False
        data=(source/name).read_bytes()
        return SimpleNamespace(commit_hash='f'*40 if wrong_commit else cfg['revision'],
                               etag=hashlib.sha256(data).hexdigest(),size=len(data))
    def download(repo,name,revision,cache_dir,token):
        assert repo==cfg['model_id'] and revision==cfg['revision'] and token is False
        calls.append(name);return str(source/name)
    module=SimpleNamespace(hf_hub_url=url,get_hf_file_metadata=metadata,hf_hub_download=download)
    monkeypatch.setitem(sys.modules,'huggingface_hub',module)
    return cfg,calls


def test_allowlisted_pinned_assets_only(tmp_path,monkeypatch):
    cfg,calls=fake_hub(tmp_path,monkeypatch)
    paths,identity=assets.fetch_snapshot(cfg['preset'],tmp_path/'cache')
    assert set(calls)==set(cfg['files'])==set(paths)==set(identity['files_sha256'])
    assert not any(x in calls for x in ['optimizer.bin','training_args.bin','pytorch_model.bin','tokenizer.json'])
    assert identity['revision']==cfg['revision']
    assert all(len(x)==64 for x in identity['files_sha256'].values())


def test_wrong_publisher_commit_rejected(tmp_path,monkeypatch):
    cfg,calls=fake_hub(tmp_path,monkeypatch,True)
    with pytest.raises(RuntimeError,match='commit'):assets.fetch_snapshot(cfg['preset'],tmp_path/'cache')
    assert calls==[]


def test_wrong_model_preset_rejected(tmp_path,monkeypatch):
    fake_hub(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='only its bundled'):assets.fetch_snapshot('turbo',tmp_path/'cache')


def test_profile_matches_whisper_v2_model_not_turbo():
    cfg=assets.profile();a=cfg['expected_architecture']
    assert len(cfg['revision'])==40 and cfg['language_token']=='zh'
    assert a['num_mel_bins']==80 and a['decoder_layers']==32 and a['vocab_size']==51865
    assert cfg['model_license']=='Apache-2.0' and cfg['model_id'].startswith('MediaTek-Research/Breeze-ASR-')
    assert 'tokenizer.json' not in cfg['files']
    if cfg['preset']=='breeze26':
        assert len(cfg['weight_files'])==2 and 'model.safetensors.index.json' in cfg['files']
        assert 'Taigi' in cfg['input_language'] and 'Mandarin' in cfg['output_description']
    else:assert cfg['weight_files']==['model.safetensors'] and cfg['source_dtype']=='BF16'


def synthetic_window(tokens=None):
    return dict(raw_tokens=[1,2,3] if tokens is None else tokens,raw_window_text='台灣 25',truncated=False)


def test_first_actual_window_twice_but_output_returned_once(tmp_path,monkeypatch):
    first=synthetic_window();second=synthetic_window();calls=[]
    def run(*args):calls.append(1);return first if len(calls)==1 else second
    monkeypatch.setattr(f,'run_window',run)
    engine=SimpleNamespace(first_window_checked=False,acceptance_report_path=tmp_path/'check.json')
    audio=np.ones(100,np.float32)
    assert f.checked_window(engine,None,None,audio,'zh') is first
    assert len(calls)==2 and engine.first_window_checked
    report=json.loads(engine.acceptance_report_path.read_text())
    assert report['passed'] and not report['reference_verbatim_checked'] and not report['model_accuracy_validated']
    f.checked_window(engine,None,None,audio,'zh')
    assert len(calls)==3


def test_repeatability_failure_stops_instead_of_claiming_success(tmp_path,monkeypatch):
    seq=iter([synthetic_window([1,2]),synthetic_window([1,3])])
    monkeypatch.setattr(f,'run_window',lambda *a:next(seq))
    engine=SimpleNamespace(first_window_checked=False,acceptance_report_path=tmp_path/'check.json')
    with pytest.raises(RuntimeError,match='repeatability failed'):
        f.checked_window(engine,None,None,np.ones(10,np.float32),'zh')
    report=json.loads(engine.acceptance_report_path.read_text())
    assert not report['passed'] and not engine.first_window_checked


def test_chinese_repeat_warning_does_not_delete_text():
    text='這個模型支援台灣華語，總共 25 個測試。'
    segments=[dict(text=text),dict(text=text)];before=json.dumps(segments,ensure_ascii=False)
    _,flag=p.mark_repetitions(segments)
    assert flag and [s['text'] for s in segments]==[text,text]
    assert len(segments)==2


def test_application_license_does_not_relicense_model():
    root=Path(__file__).resolve().parents[1]
    assert 'MIT License' in (root/'LICENSE').read_text()
    assert 'Apache License' in (root/'licenses/Apache-2.0.txt').read_text()
    assert 'Model weights are not redistributed' in (root/'THIRD_PARTY_NOTICES.md').read_text()
