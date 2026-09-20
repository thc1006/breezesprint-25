"""SRT-style timestamped plain text, with no extra download formats.

This module uses only the standard library. It does not modify decoded words,
remove repetitions or invent speaker names. JSON remains local diagnostic data.
"""
from __future__ import annotations
import math
import os
from pathlib import Path
import re
from typing import Iterable, Mapping


def timecode(seconds: float, vtt: bool = False) -> str:
    """Format milliseconds with integer carry, including recordings over 24 h."""
    if not math.isfinite(seconds):
        raise ValueError('A timestamp must be finite.')
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    separator = '.' if vtt else ','
    return f'{hours:02d}:{minutes:02d}:{secs:02d}{separator}{ms:03d}'


def timestamped_text(segments: Iterable[Mapping]) -> str:
    """A numbered cue, timestamp range and original text for every nonempty segment.

    Validate all spans before returning any bytes to the writer. Unlike an SRT
    player, a plain-text reader does not need us to replace arrows in the speech.
    """
    cues = []
    previous_end = 0.0
    for segment in segments:
        start, end = segment['start'], segment['end']
        if (not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= start < end or start + 1e-6 < previous_end):
            raise RuntimeError('Invalid or overlapping transcript segment times.')
        previous_end = end
        # Trim only framing whitespace, not words, punctuation, symbols or numbers.
        text = segment['text'].strip()
        if text:
            cues.append(f'{len(cues)+1}\n{timecode(start)} --> {timecode(end)}\n{text}\n')
    return '\n'.join(cues)


def atomic_text(path: Path, text: str) -> None:
    """Never expose a half-written final TXT file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def download_name(input_name: str, used: set[str]) -> str:
    """Readable Windows-safe filenames, unique within one multi-file upload."""
    stem = Path(input_name).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', '_', stem).strip(' .')
    # Limit encoded length, not character count (UTF-8 filenames can be multibyte).
    stem = stem.encode('utf-8')[:170].decode('utf-8', errors='ignore').rstrip(' .') or 'audio'
    candidate = stem + '_transcript.txt'
    number = 2
    while candidate.casefold() in used:
        candidate = f'{stem}_transcript_{number}.txt'
        number += 1
    used.add(candidate.casefold())
    return candidate
