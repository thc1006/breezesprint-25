# BreezeSprint25 — Breeze-ASR-25 on Colab TPU

> **v0.1.1 startup fix:** supports the official `<|nocaptions|>` token.
> Replace both the repository sources and generated notebook; do not reuse the v0.1.0 notebook payload. [Details](docs/TOKENIZER_FIX.md).

**Transcribe Taiwanese Mandarin and Mandarin–English recordings into timestamped text. Upload a recording; get one timestamped TXT.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thc1006/breezesprint-25/blob/main/BreezeSprint25.ipynb)
[繁體中文](README.zh-TW.md) · [Model contract](docs/MODEL.md) · [Validation](docs/VALIDATION.md)

> **Release status: experimental TPU integration.** The JAX equations descend from
> a working WhisperSprint implementation, but this Breeze checkpoint has NOT been
> executed on a physical TPU by the packager. Local tests do not substitute for
> Mandarin/Taigi audio validation. See `evidence/validation.json` for measured scope.

This targets the ASR-25 Mandarin/code-switching checkpoint, not Taigi-specific ASR-26. Traditional Chinese is the model tendency, not a per-character guarantee.

## Two cells, no output settings

Open the badge, select **Runtime → Change runtime type → TPU v5e-1**, and run cell 1.
In cell 2, upload one or more recordings. Each finished file automatically requests
a **UTF-8 `.txt` download with numbered SRT-style timestamps**. Nothing to unzip.
Rerun only cell 2 for the next recording; the worker keeps weights and compiled code.
The browser may ask for download permission. Cancellation does not reuse old inputs.
There is no Drive mount, file-path field, model selector, or external ASR API.

The notebook is self-contained. Upload `BreezeSprint25.ipynb` to Colab before publication;
the badge becomes usable after the repo is published to the path above.

```text
1
00:00:01,200 --> 00:00:04,800
這是含時間戳的輸出格式示例，不是模型測試結果。
```

## Model and implementation

Model: [`MediaTek-Research/Breeze-ASR-25`](https://huggingface.co/MediaTek-Research/Breeze-ASR-25)  
Pinned commit: `cffe7ccb404d025296a00758d0a33468bec3a9d0`  
Source weights: **BF16**, about **3.09 GB**; not included in this repo.  
Architecture: Whisper-large-v2, **80 mel bins, 32 encoder and 32 decoder layers**.  
Runtime: JAX/JAXLIB **0.9.0.1**, libtpu **0.0.34**, BF16 computation, single v5/v6 TPU.  
Decode profile: explicit **`zh` + `transcribe`**, greedy, full 448-token context.

The fixed `zh` token is this application's Chinese-output choice; it is not a
claim that the source language is always Mandarin or that this setting reproduces
the publisher's benchmark. The upstream generation config allows automatic language
selection; our simpler UI deliberately does not expose that setting. No `nan`
Whisper token is invented. ASR-26's source speech can be Taigi even though the
output/control profile uses Chinese. No OpenCC, LLM cleanup, forced alignment,
translation API, or automatic duplicate deletion is added.

The host loader accepts native BF16 arrays and validates all expected tensor
names/shapes. Multi-shard checkpoints are checked against the index, including
duplicate/misplaced tensors. Vocabulary decoding reads the model's own `vocab.json`
and `added_tokens.json`, so ASR-26 does not need a nonexistent `tokenizer.json`.
All downloads use an explicit file allow-list and immutable commit, with content
ETag verification. No training state, Pickle, model conversion, or remote Python
code is loaded. [Integrity details](docs/MODEL.md).

## What is checked at runtime

The worker must pass a real TPU computation and checkpoint-schema validation.
The first nonzero window of your recording is decoded twice to check token
repeatability before committing it. This is **not** an independent known-speech
accuracy test. Silence-only jobs may have no inference probe at all; their report
stays explicit about that. Local JSON, progress and logs retain the check scope.
A failed run does not auto-download a partial TXT as a completed result.

## Performance and resource limits

One persistent worker overlaps small metadata downloads with TPU initialization.
It validates the full vocabulary before downloading weights; setup can still
overlap the user upload.
Weights are packed on CPU, then transferred in bounded groups; sequence shapes and
K/V cache capacity stay fixed. The next independent file is decoded in a bounded
prefetch queue. Long recordings use disk-backed PCM and timestamp-driven seeking.
No quota workaround or machine-wide tuning is performed. Run the two Breeze
projects in separate runtimes, not simultaneous workers on one TPU.

A 32-layer decoder has different compute/cache costs from Whisper Turbo's four
layers. **WhisperSprint's measured timings do not apply here.** First-job time also
includes JIT compilation and one repeatability pass. Colab allocation, downloads,
RAM limits and browser transfers can dominate. [Timing scope](docs/PERFORMANCE.md).

## Privacy and limits

Recordings go to your **Google-hosted Colab runtime**. This is cloud processing,
not on-device/offline operation. No third-party ASR upload is added. Temp files,
transcripts and logs remain in the runtime; treat diagnostics as potentially
sensitive. Resetting a runtime can remove local progress.

Segment timestamps are not word-level alignment or speaker labels. Names, numbers,
quiet speech, repetitions and domain terms still need listening checks. This targets the ASR-25 Mandarin/code-switching checkpoint, not Taigi-specific ASR-26. Traditional Chinese is the model tendency, not a per-character guarantee.

## Develop or publish

```bash
python -m pip install -r requirements-cpu.txt -r requirements-dev.txt
# Independent PyTorch numeric reference used only by tests:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m pytest -q
python tools/build_notebook.py
python tools/check_release.py
```

[Publishing](PUBLISH.md) · [Agent instructions](AGENTS.md) · [Troubleshooting](docs/TROUBLESHOOTING.md)

Application code: MIT. Breeze model weights: Apache-2.0, supplied by the original
publisher. Preserve attribution. This is an **independent integration**, not an
official MediaTek, NYCU, Google or OpenAI release. [Notices](THIRD_PARTY_NOTICES.md).
