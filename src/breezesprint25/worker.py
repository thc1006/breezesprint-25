"""One isolated, persistent TPU process. JSON commands over stdin; no HTTP server."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from . import __version__


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temporary, path)


class Worker:
    def __init__(self, root, session):
        self.root, self.session = Path(root), Path(session)
        self.started = time.perf_counter()
        self.status = dict(state='starting', stage='imports')
        self.last_update = 0.
        self.emit(force=True)

    def emit(self, force=False, **fields):
        old_stage = self.status.get('stage')
        self.status.update(fields)
        now = time.perf_counter()
        if force or fields.get('stage', old_stage) != old_stage or now-self.last_update >= 1:
            self.status['updated_at'] = time.time()
            atomic(self.session/'status.json', self.status)
            self.last_update = now

    def initialize(self):
        import jax
        from . import frontend as f
        from .core import ModelSpec
        from .engine import ResidentWhisper
        from .loading import load_weights
        from .preflight import METADATA_FILES, validate_model_assets, merge_asset_identities
        self.f = f
        cache = self.root/'jit-cache'
        cache.mkdir(mode=0o700, exist_ok=True)
        os.chmod(cache, 0o700)
        jax.config.update('jax_compilation_cache_dir', str(cache))
        jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
        jax.config.update('jax_default_matmul_precision', 'default')
        # The download and TPU initialization are independent. No second probe process.
        self.emit(stage='TPU initialization + metadata download', force=True)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='cold-start') as pool:
            expected = f.profile()
            model_future = pool.submit(f.fetch_snapshot, expected['preset'], self.root/'hf-cache', METADATA_FILES)
            target = f.select_tpu(strict_v5=False)
            probe = f.hardware_probe(target)
            f.atomic_json(self.session/'hardware.json', probe)
            self.emit(stage='metadata download + verification', force=True,
                      device=target.device_kind)
            files, model = model_future.result()
        if model['model_id'] != expected['model_id'] or model['revision'] != expected['revision']:
            raise RuntimeError('Unexpected model identity.')
        self.emit(stage='tokenizer/configuration validation', force=True)
        spec, codec, tokenizer_report = validate_model_assets(files, expected)
        f.atomic_json(self.session/'tokenizer_validation.json', tokenizer_report)
        self.emit(stage='weight download + verification', force=True)
        remaining = [name for name in expected['files'] if name not in METADATA_FILES]
        weight_paths, weight_identity = f.fetch_snapshot(expected['preset'], self.root/'hf-cache', remaining)
        model = merge_asset_identities(model, weight_identity, expected['files'])
        files.update(weight_paths)
        self.emit(stage='loading BF16 model', force=True)
        started = time.perf_counter()
        params = load_weights([files[n] for n in expected['weight_files']], spec, target, 'bfloat16',
                              index_path=files.get('model.safetensors.index.json'))
        load_time = time.perf_counter()-started
        self.engine = ResidentWhisper(params, spec, target, codec)
        self.codec, self.filters = codec, f.mel_filters(80)
        self.engine.first_window_checked = False
        self.engine.acceptance_report_path = self.session/'first_window_check.json'
        source_hashes = {p.name: f.sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
        versions = f.package_versions()
        ffmpeg = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
        self.manifest = dict(
            project='BreezeSprint25', release=__version__, model=model, dtype='bfloat16',
            language=expected['language_token'], input_language=expected['input_language'],
            output_description=expected['output_description'], packages=versions, ffmpeg=ffmpeg,
            device_kind=target.device_kind, source_sha256=source_hashes,
            model_load_seconds=load_time,
            worker_setup_seconds=time.perf_counter()-self.started,
            parameter_bytes=sum(x.size*x.dtype.itemsize for x in jax.tree.leaves(params)),
            hardware_probe_passed=probe['probe_passed'],
            tokenizer_validation=tokenizer_report,
            first_window_repeatability_passed=None, model_accuracy_validated=False,
            decode=dict(task='transcribe', strategy='greedy', sampling=False,
                        batch_size=1, max_context=448, audio_speed_factor=1.,
                        timestamps='segment', condition_on_previous_text=False),
            validation_scope='Hardware probe + pinned weights/schema. First nonzero user window checked twice. No independent Mandarin/Taigi accuracy reference.')
        f.atomic_json(self.session/'runtime_validation.json', self.manifest)
        self.emit(state='ready', stage='ready', force=True,
                  setup_seconds=self.manifest['worker_setup_seconds'])

    def transcribe(self, request):
        from .pipeline import transcribe_many
        ident = request['id']
        if not isinstance(ident, str) or len(ident) != 32 or any(c not in '0123456789abcdef' for c in ident):
            raise ValueError('Invalid request identifier.')
        output = Path(request['output']).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        self.emit(state='busy', stage='preparing audio', force=True, request=ident)
        try:
            result = transcribe_many(request['paths'], self.engine, self.codec, self.filters,
                                     self.root, output, self.manifest, self.emit)
            self.f.atomic_json(self.session/'runtime_validation.json', self.manifest)
            result['request_id'] = ident
            result['runtime_validation'] = str(self.session/'runtime_validation.json')
            self.f.atomic_json(output/('batch_'+ident+'.json'), result)
            atomic(self.session/(ident+'.json'), dict(ok=True, result=result))
            self.emit(state='ready', stage='ready', force=True)
        except Exception as exc:
            traceback.print_exc()
            report = dict(ok=False, stage=self.status.get('stage'),
                          error_type=type(exc).__name__, message=str(exc),
                          recovery='Fix the input or environment, then rerun upload cell 2. '
                                   'For device/OOM errors restart the runtime. Partial files remain on disk.')
            atomic(self.session/(ident+'.json'), report)
            self.emit(state='ready', stage='previous request failed', force=True,
                      last_error=report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    root, session = Path(args.root), Path(args.session)
    root.mkdir(exist_ok=True, parents=True)
    session.mkdir(exist_ok=True, parents=True)
    # Only this project's worker; never kill or modify another Colab process.
    with (root/'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another BreezeSprint25 worker is active. Restart the runtime before installing another copy.')
        worker = Worker(root, session)
        try:
            worker.initialize()
            for line in sys.stdin:
                request = json.loads(line)
                if request.get('op') == 'close':
                    return
                if request.get('op') != 'transcribe':
                    raise ValueError('Unknown worker operation.')
                worker.transcribe(request)
        except BaseException as exc:
            traceback.print_exc()
            worker.emit(state='failed', stage=worker.status.get('stage'), force=True,
                        error_type=type(exc).__name__, message=str(exc))
            raise


if __name__ == '__main__':
    main()
