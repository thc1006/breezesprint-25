# v0.1.1: no-speech tokenizer compatibility

## What failed

BreezeSprint25 0.1.0 looked up only `<|nospeech|>`. The official published
Breeze vocabulary uses `<|nocaptions|>` for the no-speech marker. Initialization
therefore raised `ValueError: Missing special token: <|nospeech|>` before audio
inference. The input format, recording size and TPU matrix operations were not
the failing call in the reported traceback.

## Resolution

The ID is now read from the actual vocabulary under either accepted spelling.
No hardcoded numeric fallback, tokenizer-file rewrite, silence-filter removal,
model substitution or CPU fallback is used. Configuration IDs must agree with
vocabulary IDs. Timestamp IDs remain contiguous and validated.

For the published Whisper-large-v2-derived Breeze layout, the relevant values are:

```text
<|nocaptions|>     50362
<|notimestamps|>   50363
<|0.00|>          50364
<|30.00|>         51864
```

Sources checked on 2026-09-21:
- https://huggingface.co/MediaTek-Research/Breeze-ASR-25/blob/main/added_tokens.json
- https://huggingface.co/MediaTek-Research/Breeze-ASR-26/blame/main/added_tokens.json
- https://huggingface.co/MediaTek-Research/Breeze-ASR-25/blob/main/generation_config.json

## Existing Colab session

Do not delete the runtime merely for this error. Replace its first cell with the
first cell from the v0.1.1 notebook and run that cell again. It closes only its own
worker and starts the patched one, retaining available model/package caches and
uploaded files. The second cell is unchanged. Running the OLD first cell would
restore the broken source payload; a change to GitHub alone does not update a
saved notebook copy.

Normal use remains two cells: prepare, then upload/transcribe/download TXT.
The upload cell intentionally does not silently reuse earlier audio after a
cancelled upload. A developer can explicitly reuse a known retained local file
with `sprint.transcribe([path])`, then call `google.colab.files.download()` on each
returned TXT. This is an optional manual recovery, not a new default UI field.

## Verification scope

`evidence/regression-reproduction.json` records the old failure and the corrected
initialization on the same fixture. The fixture uses publisher-documented control
IDs and synthetic ordinary vocabulary; it is NOT the complete downloaded model
vocabulary. All 191 local tests pass. Network downloads of the full vocabulary
could not be performed in this build environment.

The new `tools/check_model_assets.py` downloads only the five real metadata files
at the pinned model commit and validates them on CPU. It is also a separate CI
job. A network failure fails that check; it is not silently skipped. Weight files
are never requested by that check. At runtime the same validation occurs before
weight downloads. Neither result is a substitute for a real TPU transcription
and an independent Mandarin/Taigi reference.
