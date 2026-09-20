"""Token alias + metadata-first regressions.

The fixture uses publisher-documented special IDs and SYNTHETIC ordinary tokens.
It is NOT a downloaded full Breeze vocabulary. The separate online check validates
real immutable publisher files and does not skip when the network is unavailable.
"""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import numpy as np
import pytest
import jax
from breezesprint25 import assets, frontend as f, loading, engine
from breezesprint25.preflight import METADATA_FILES, validate_model_assets, merge_asset_identities
from breezesprint25.vocabulary import resolve_no_speech_token, byte_alphabet
from breezesprint25.worker import Worker
from test_core import SPEC
from test_breeze import vocab_fixture, fake_hub

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('tokens,expected', [
    ({'<|nospeech|>':260},260),
    ({'<|nocaptions|>':260},260),
    ({'<|nospeech|>':50362},50362),
    ({'<|nocaptions|>':50362},50362),
    ({'<|nospeech|>':50362,'<|nocaptions|>':50362},50362),
    ({'<|nocaptions|>':0},0),
])
def test_lookup_uses_actual_alias_not_a_hardcoded_id(tokens,expected):
    name,ident=resolve_no_speech_token(tokens.get,51865)
    assert name in tokens and ident==expected


@pytest.mark.parametrize('tokens', [
    {}, {'<|unrelated|>':50362}, {'<|nospeech|>':-1},
    {'<|nocaptions|>':51865}, {'<|nospeech|>':True},
    {'<|nocaptions|>':'50362'}, {'<|nocaptions|>':50362.0},
    {'<|nocaptions|>':50362,'<|nospeech|>':50363},
])
def test_bad_aliases_never_get_a_guessed_fallback(tokens):
    with pytest.raises(ValueError):resolve_no_speech_token(tokens.get,51865)


@pytest.mark.parametrize('name',['<|nospeech|>','<|nocaptions|>'])
def test_full_codec_supports_both_names_without_mutating_vocab(tmp_path,name):
    vp,ap,_,added=vocab_fixture(tmp_path,True)
    ident=added.pop('<|nospeech|>');added[name]=ident
    ap.write_text(json.dumps(added));before=ap.read_bytes()
    codec=f.Codec(vp,{},replace(SPEC,vocab=1763),ap)
    assert codec.no_speech==ident and codec.no_speech_token==name
    assert codec.ds.no_speech==ident and codec.suppress[ident]
    assert ap.read_bytes()==before
    assert codec.prefix('zh').tolist()==[[257,258,259]]


@pytest.mark.parametrize('key,value',[
    ('no_speech_token_id',261),('no_speech_token_id',True),
    ('no_speech_token_id','260'),('decoder_start_token_id',256),
    ('eos_token_id',257),('no_timestamps_token_id',260),
])
def test_generation_special_ids_must_match_vocabulary(tmp_path,key,value):
    vp,ap,_,_=vocab_fixture(tmp_path,True)
    with pytest.raises(ValueError,match='conflicts'):
        f.Codec(vp,{key:value},replace(SPEC,vocab=1763),ap)


@pytest.mark.parametrize('generation',[
    {'task_to_id':{'transcribe':258}},
    {'task_to_id':{'transcribe':True}},
    {'lang_to_id':{'<|zh|>':259}},
    {'lang_to_id':{'<|en|>':258}},
    {'suppress_tokens':[True]},
])
def test_bad_task_language_suppression_rejected(tmp_path,generation):
    vp,ap,_,_=vocab_fixture(tmp_path,True)
    with pytest.raises(ValueError):f.Codec(vp,generation,replace(SPEC,vocab=1763),ap)


def test_matching_config_ids_are_accepted(tmp_path):
    vp,ap,_,_=vocab_fixture(tmp_path,True)
    gen={'decoder_start_token_id':257,'eos_token_id':256,'no_timestamps_token_id':261,
         'no_speech_token_id':260,'task_to_id':{'transcribe':259},'lang_to_id':{'<|zh|>':258}}
    assert f.Codec(vp,gen,replace(SPEC,vocab=1763),ap).no_speech==260


def published_contract_fixture(tmp_path):
    """Only specials are publisher facts; ordinary byte strings are synthetic."""
    contract=json.loads((ROOT/'tests/fixtures/publisher-token-contract.json').read_text())
    p=assets.profile();inverse={v:k for k,v in byte_alphabet().items()}
    vocab={inverse[i]:i for i in range(256)}
    # Fill the real ID range, but do not misrepresent these as pretrained BPE merges.
    vocab.update({'fixture_'+str(i):i for i in range(256,50257)})
    vocab['<|endoftext|>']=50257
    added=dict(contract['special_tokens'])
    used=set(added.values())
    for i in range(50258,50364):
        if i not in used:added[f'<|fixture_control_{i}|>']=i
    added.update({f'<|{i*.02:.2f}|>':50364+i for i in range(1501)})
    config=dict(p['expected_architecture'],model_type='whisper',activation_function='gelu',
                tie_word_embeddings=True,decoder_start_token_id=50258,eos_token_id=50257)
    prep=dict(sampling_rate=16000,n_fft=400,hop_length=160,feature_size=80,chunk_length=30,padding_value=0.)
    generation=dict(contract['generation_config_excerpt'])
    data={'vocab.json':vocab,'added_tokens.json':added,'config.json':config,
          'generation_config.json':generation,'preprocessor_config.json':prep}
    paths={}
    for name,value in data.items():
        path=tmp_path/name;path.write_text(json.dumps(value,ensure_ascii=False));paths[name]=path
    return paths,p


def test_publisher_special_token_layout_initializes_complete_codec(tmp_path):
    paths,p=published_contract_fixture(tmp_path)
    spec,codec,report=validate_model_assets(paths,p)
    assert codec.tok.token_to_id('<|nospeech|>') is None
    assert report['no_speech_token']=='<|nocaptions|>'
    assert codec.no_speech==50362 and codec.no_ts==50363 and codec.ts_begin==50364
    assert codec.prefix('zh').tolist()==[[50258,50260,50359]]
    assert codec.ts_begin+1500==51864 and spec.vocab==51865
    assert codec.text(list('微積分 11.1 Sequences，2026。'.encode('utf8')))=='微積分 11.1 Sequences，2026。'
    assert not report['weights_loaded'] and not report['tpu_inference_tested']


@pytest.mark.parametrize('kind',['architecture','preprocessor','missing_metadata','config_eos'])
def test_preflight_fails_before_weights_for_bad_metadata(tmp_path,kind):
    paths,p=published_contract_fixture(tmp_path)
    if kind=='missing_metadata':paths.pop('vocab.json')
    else:
        file=paths['preprocessor_config.json' if kind=='preprocessor' else 'config.json']
        data=json.loads(file.read_text())
        data[{'architecture':'decoder_layers','preprocessor':'feature_size','config_eos':'eos_token_id'}[kind]]=1
        file.write_text(json.dumps(data))
    with pytest.raises(ValueError):validate_model_assets(paths,p)


def test_metadata_download_never_requests_weight_files(tmp_path,monkeypatch):
    cfg,calls=fake_hub(tmp_path,monkeypatch)
    paths,identity=assets.fetch_snapshot(cfg['preset'],tmp_path/'cache',METADATA_FILES)
    assert set(paths)==set(calls)==set(METADATA_FILES)
    assert not any(name.endswith('.safetensors') for name in calls)
    assert set(identity['files_sha256'])==set(METADATA_FILES)


@pytest.mark.parametrize('names',[[],['vocab.json','vocab.json'],['../vocab.json'],['training_args.bin'],[True]])
def test_invalid_asset_subsets_are_rejected_without_a_download(tmp_path,monkeypatch,names):
    cfg,calls=fake_hub(tmp_path,monkeypatch)
    with pytest.raises(ValueError):assets.fetch_snapshot(cfg['preset'],tmp_path/'cache',names)
    assert not calls


def identity(p,names):
    return dict(model_id=p['model_id'],revision=p['revision'],
                files_sha256={n:'0'*64 for n in names},files_integrity={n:{'fixture':True} for n in names})


@pytest.mark.parametrize('bad',['revision','overlap','missing'])
def test_two_phase_identity_merge_is_strict(bad):
    p=assets.profile();remaining=[n for n in p['files'] if n not in METADATA_FILES]
    a=identity(p,METADATA_FILES);b=identity(p,remaining)
    if bad=='revision':b['revision']='0'*40
    elif bad=='overlap':b=identity(p,list(remaining)+['vocab.json'])
    else:b=identity(p,[])
    with pytest.raises(ValueError):merge_asset_identities(a,b,p['files'])


def test_worker_checks_codec_before_requesting_any_weight(tmp_path,monkeypatch):
    paths,p=published_contract_fixture(tmp_path)
    data=json.loads(paths['added_tokens.json'].read_text())
    data['<|unknown-no-speech|>']=data.pop('<|nocaptions|>')
    paths['added_tokens.json'].write_text(json.dumps(data))
    calls=[]
    def fetch(preset,cache,names=None):
        calls.append(list(names));assert set(names)==set(METADATA_FILES)
        return paths,identity(p,names)
    monkeypatch.setattr(f,'fetch_snapshot',fetch)
    monkeypatch.setattr(f,'select_tpu',lambda **kw:SimpleNamespace(device_kind='MOCK device, not a TPU test'))
    monkeypatch.setattr(f,'hardware_probe',lambda d:{'probe_passed':False})
    loader=Mock(side_effect=AssertionError('weights must not load'))
    monkeypatch.setattr(loading,'load_weights',loader)
    worker=Worker(tmp_path,tmp_path/'session')
    with pytest.raises(ValueError,match='no-speech token'):worker.initialize()
    assert calls==[list(METADATA_FILES)]
    assert worker.status['stage']=='tokenizer/configuration validation'
    loader.assert_not_called()


def test_worker_two_phase_success_records_resolved_token(tmp_path,monkeypatch):
    paths,p=published_contract_fixture(tmp_path);calls=[]
    def fetch(preset,cache,names=None):
        calls.append(list(names))
        if len(calls)==1:return paths,identity(p,names)
        assert (tmp_path/'session/tokenizer_validation.json').exists()
        return {name:tmp_path/name for name in names},identity(p,names)
    monkeypatch.setattr(f,'fetch_snapshot',fetch)
    monkeypatch.setattr(f,'select_tpu',lambda **kw:SimpleNamespace(device_kind='MOCK device, not a TPU test'))
    monkeypatch.setattr(f,'hardware_probe',lambda d:{'probe_passed':False})
    monkeypatch.setattr(loading,'load_weights',lambda *a,**k:{'fixture':np.ones(1,np.float32)})
    monkeypatch.setattr(engine,'ResidentWhisper',lambda *a:SimpleNamespace())
    worker=Worker(tmp_path,tmp_path/'session');worker.initialize()
    assert worker.status['state']=='ready'
    assert set(calls[0])==set(METADATA_FILES)
    assert set(calls[1])==set(p['files'])-set(METADATA_FILES)
    assert worker.manifest['tokenizer_validation']['no_speech_token_id']==50362
    assert worker.manifest['model_accuracy_validated'] is False
    assert worker.manifest['hardware_probe_passed'] is False  # explicitly fake probe, no device claim


def test_online_ci_check_is_separate_and_downloads_metadata_only():
    text=(ROOT/'tools/check_model_assets.py').read_text()
    assert 'fetch_snapshot(p[\'preset\'], args.cache_dir, METADATA_FILES)' in text
    assert 'return 1' in text and 'pytest.skip' not in text
    assert 'tools/check_model_assets.py' in (ROOT/'.github/workflows/ci.yml').read_text()
