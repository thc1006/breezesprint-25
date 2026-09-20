"""Breeze long-form transcription, with bounded prefetch and honest timing."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import re
import time
import traceback
import numpy as np
from . import frontend as f


@dataclass
class PreparedAudio:
    path: Path
    digest: str
    audio: np.ndarray
    context: object
    seconds: float

    def close(self):
        self.context.__exit__(None, None, None)


def prepare_audio(path: Path, work: Path) -> PreparedAudio:
    start = time.perf_counter()
    path = Path(path).expanduser().resolve(strict=True)
    before = path.stat()
    digest = f.sha256_file(path)
    context = f.decoded_audio(path, work / 'temp')
    audio = context.__enter__()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        context.__exit__(None, None, None)
        raise RuntimeError('Input changed during hashing/decoding. Save a stable copy and retry.')
    return PreparedAudio(path, digest, audio, context, time.perf_counter()-start)


def normalized_words(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+(?:'[a-z]+)?|[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0003134f]", text.casefold()))


def mark_repetitions(segments, previous=()):
    """Flag, never delete: a repeated teaching phrase can be legitimate audio."""
    flagged = False
    for segment in segments:
        words = normalized_words(segment['text'])
        if len(words) >= 4 and words == previous:
            segment.setdefault('review_flags', []).append('adjacent_repetition_review')
            flagged = True
        previous = words
    return previous, flagged


def transcribe_prepared(prepared, engine, codec, filters, work, output_root,
                        base_manifest, event=lambda **kw: None):
    audio = prepared.audio
    total = len(audio)
    language = base_manifest.get('language', 'zh')
    identity = dict(base_manifest, audio_sha256=prepared.digest, language=language)
    stable_keys = ['model', 'dtype', 'language', 'audio_sha256', 'source_sha256',
                   'packages', 'ffmpeg', 'device_kind', 'decode']
    stable = {key: identity[key] for key in stable_keys}
    signature = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()
    stem = re.sub(r'[^\w.-]+', '_', prepared.path.stem)[:70] or 'audio'
    outdir = Path(output_root)/(stem+'_'+signature[:16])
    outdir.mkdir(parents=True, exist_ok=True)
    identity.update(signature=signature, input_filename=prepared.path.name,
                    duration_seconds=total/f.SR,
                    reference_verbatim_checked=False)
    journal = outdir/'progress.jsonl'
    rows = f.load_journal(journal, signature, total)
    seek = sum(row['advance_samples'] for row in rows)
    resumed_samples = seek
    last_segments = [s for row in rows for s in row['segments']]
    previous = normalized_words(last_segments[-1]['text']) if last_segments else ()
    start = time.perf_counter()
    f.render_outputs(outdir, rows, identity, complete=seek==total)
    event(stage='transcribing', file=prepared.path.name, processed=seek/f.SR, total=total/f.SR)
    slow = 0
    last_checkpoint = last_export = time.perf_counter()
    try:
        with journal.open('a', encoding='utf-8') as log:
            while seek < total:
                valid = min(f.N_SAMPLES, total-seek)
                samples = np.array(audio[seek:seek+valid], dtype=np.float32, copy=True)
                if not np.isfinite(samples).all():
                    raise ValueError('Nonfinite decoded PCM.')
                if np.count_nonzero(samples) == 0:
                    row = dict(raw_tokens=[], raw_window_text='', segments=[],
                               advance_samples=valid, flags=['exact_digital_silence'],
                               feature_seconds=0., encoder_seconds=0., decoder_seconds=0.)
                else:
                    row = f.checked_window(engine, codec, filters, samples, language)
                    if getattr(engine, 'first_window_checked', False):
                        identity['first_window_repeatability_passed'] = True
                        base_manifest['first_window_repeatability_passed'] = True
                for segment in row['segments']:
                    segment['start'] += seek/f.SR
                    segment['end'] += seek/f.SR
                    segment['window'] = len(rows)
                previous, repeat = mark_repetitions(row['segments'], previous)
                if repeat:
                    row['flags'].append('adjacent_repetition_review_not_deleted')
                row.update(index=len(rows), signature=signature,
                           seek_samples=seek, valid_samples=valid)
                slow = slow+1 if row['advance_samples'] < f.SR//2 and valid > f.SR else 0
                if slow >= 5:
                    f.atomic_json(outdir/'failed_window.json', row)
                    raise RuntimeError('Five tiny timestamp advances: stopping a likely loop.')
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                log.flush()  # process-crash recovery at every committed window
                now = time.perf_counter()
                if now-last_checkpoint >= 2:
                    os.fsync(log.fileno())  # fsync between windows after >=2 seconds have elapsed
                    last_checkpoint = now
                rows.append(row)
                seek += row['advance_samples']
                event(stage='transcribing', file=prepared.path.name,
                      processed=seek/f.SR, total=total/f.SR)
                if now-last_export >= 60:
                    f.render_outputs(outdir, rows, identity, complete=False)
                    last_export = now
            log.flush()
            os.fsync(log.fileno())
        result = f.render_outputs(outdir, rows, identity, complete=True)
    except BaseException:
        try:
            f.render_outputs(outdir, rows, identity, complete=False)
        except Exception:
            # A full disk or bad timestamp must not hide the original failure.
            traceback.print_exc()
        raise
    export_seconds = time.perf_counter()-start
    # This timer ends after final timestamped TXT + local JSON/manifest export, not before it.
    # timing.json, mmap cleanup, download staging and browser download are separate.
    work_seconds = prepared.seconds + export_seconds
    timing = dict(
        audio_preparation_seconds=prepared.seconds,
        inference_and_artifact_export_seconds=export_seconds,
        file_processing_seconds=work_seconds,
        audio_seconds=total/f.SR,
        resumed_samples=resumed_samples,
        full_file_rtf=work_seconds/(total/f.SR) if not resumed_samples else None,
        rtfx=(total/f.SR)/work_seconds if not resumed_samples else None,
        scope='Hash + FFmpeg + inference + final transcript export. Excludes model setup, '
              'queue waits, timing.json write, mmap cleanup, TXT download staging and browser transfer. '
              'Per-file work intervals can overlap across files; use batch_wall_seconds for a batch.',
        feature_seconds=sum(r['feature_seconds'] for r in rows),
        encoder_seconds=sum(r['encoder_seconds'] for r in rows),
        decoder_seconds=sum(r['decoder_seconds'] for r in rows),
        component_scope='Journal components include previously committed work when resumed.')
    f.atomic_json(outdir/'timing.json', timing)
    return dict(directory=str(outdir), text_file=str(outdir/'transcript.txt'),
                input_name=prepared.path.name,
                segments=len(result['segments']), review_windows=len(result['review_windows']),
                timing=timing, processing_complete=True, reference_verbatim_checked=False)


def transcribe_many(paths, engine, codec, filters, work, output_root, manifest,
                    event=lambda **kw: None):
    """Keep at most the current and next disk-backed decoded files open.

    Decode/hash the next independent recording during current TPU inference.
    Do not split one recording into unrelated chunks just to fill a batch.
    """
    paths = list(dict.fromkeys(Path(p).expanduser().resolve(strict=True) for p in paths))
    if not paths:
        raise ValueError('No audio selected.')
    start = time.perf_counter()
    outputs = []
    pending = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='audio-prefetch') as pool:
        try:
            pending = pool.submit(prepare_audio, paths[0], work)
            for index, path in enumerate(paths):
                event(stage='decoding', file=path.name, file_index=index+1, files=len(paths))
                current = pending.result()
                pending = None
                try:
                    if index+1 < len(paths):
                        pending = pool.submit(prepare_audio, paths[index+1], work)
                    outputs.append(transcribe_prepared(current, engine, codec, filters,
                                   work, output_root, manifest, event))
                finally:
                    current.close()
        finally:
            if pending is not None:
                # Drain and close prepared mmap even when the current file fails.
                try:
                    pending.result().close()
                except Exception:
                    pass  # the active exception remains authoritative; worker logs it
    elapsed = time.perf_counter()-start
    audio_seconds = sum(item['timing']['audio_seconds'] for item in outputs)
    resumed = any(item['timing']['resumed_samples'] for item in outputs)
    return dict(outputs=outputs, batch_wall_seconds=elapsed,
                audio_seconds=audio_seconds,
                batch_rtfx=None if resumed else audio_seconds/elapsed,
                processing_complete=True, reference_verbatim_checked=False,
                scope='Batch wall includes prefetch waits, hashing, decoding, inference, '
                      'transcript exports, timing sidecars and decoded-file cleanup. '
                      'Excludes model setup, final batch report, TXT download staging and browser transfer.')
