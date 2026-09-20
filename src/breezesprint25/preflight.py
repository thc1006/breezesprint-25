"""Validate small model assets before downloading any multi-GB weight shards."""
from pathlib import Path
import json
from .core import ModelSpec
from .frontend import Codec

METADATA_FILES = ('config.json', 'generation_config.json', 'preprocessor_config.json',
                  'vocab.json', 'added_tokens.json')


def validate_model_assets(files, expected):
    missing = sorted(set(METADATA_FILES) - set(files))
    if missing:
        raise ValueError('Missing model metadata files: ' + ', '.join(missing))
    config = json.loads(Path(files['config.json']).read_text(encoding='utf-8'))
    for key, value in expected['expected_architecture'].items():
        if type(config.get(key)) is not int or config[key] != value:
            raise ValueError('Unexpected Breeze architecture field: ' + key)
    spec = ModelSpec.from_config(config)
    prep = json.loads(Path(files['preprocessor_config.json']).read_text(encoding='utf-8'))
    for key, value in dict(sampling_rate=16000, n_fft=400, hop_length=160,
                           feature_size=spec.mel_bins, chunk_length=30, padding_value=0.).items():
        if isinstance(prep.get(key), bool) or prep.get(key) != value:
            raise ValueError('Unexpected audio frontend configuration: ' + key)
    codec = Codec(Path(files['vocab.json']),
                  json.loads(Path(files['generation_config.json']).read_text(encoding='utf-8')),
                  spec, added_tokens_path=Path(files['added_tokens.json']))
    prefix = codec.prefix(expected['language_token']).tolist()
    for key, value in {'decoder_start_token_id':codec.sot, 'eos_token_id':codec.eos}.items():
        actual = config.get(key)
        if actual is not None and (type(actual) is not int or actual != value):
            raise ValueError('Model config conflicts with vocabulary: ' + key)
    report = dict(passed=True, model_id=expected['model_id'], revision=expected['revision'],
                  vocabulary_size=spec.vocab, no_speech_token=codec.no_speech_token,
                  no_speech_token_id=codec.no_speech, no_timestamps_token_id=codec.no_ts,
                  timestamp_begin=codec.ts_begin, timestamp_count=1501,
                  decoder_prefix=prefix, weights_loaded=False, tpu_inference_tested=False,
                  scope='Model metadata and full vocabulary validation only; not ASR accuracy or TPU inference.')
    return spec, codec, report


def merge_asset_identities(first, second, expected_names):
    for key in ('model_id', 'revision'):
        if first.get(key) != second.get(key):
            raise ValueError('Conflicting asset snapshot identity: ' + key)
    result = {key:first[key] for key in ('model_id','revision')}
    for key in ('files_sha256','files_integrity'):
        left, right = first[key], second[key]
        if set(left) & set(right):
            raise ValueError('Duplicate asset in metadata/weight phases.')
        result[key] = dict(left, **right)
        if set(result[key]) != set(expected_names):
            raise ValueError('Incomplete model snapshot after metadata/weight phases.')
    return result
