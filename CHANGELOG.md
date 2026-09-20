# Changelog

## 0.1.1 — 2026-09-21

Fix initialization against the published Breeze tokenizer. Its no-speech marker
is named `<|nocaptions|>` (ID 50362), not `<|nospeech|>`. Resolve either spelling
from the vocabulary without guessing a numeric ID; reject missing/conflicting
aliases and mismatched generation IDs.

Validate small configuration/vocabulary assets before requesting weight files.
Report the precise `tokenizer/configuration validation` stage and save resolved
token IDs in `tokenizer_validation.json`. Preserve the model pin, numerical core,
weight-loading math, two upload-only cells and automatic timestamped TXT output.

Add 45 regressions (191 local tests total), plus an independent online CI check
that validates the full pinned publisher vocabulary without downloading weights.
The online check was NOT run successfully in the build environment; network
access to the Hub failed. Local fixtures use documented special tokens with
synthetic ordinary tokens, not a downloaded complete tokenizer. No full Breeze
TPU inference or target-language accuracy claim is made.

## 0.1.0 — 2026-09-21

Independent Breeze-ASR-25 port of the two-cell WhisperSprint workflow.
Added native BF16 and strict multi-shard loading, source-vocabulary output decoder,
80-bin/32-decoder profile, Chinese-output prefix and per-worker first-input
repeatability report. Removed English-only model/JFK acceptance assumptions.
Preserved timestamped TXT-only download, failure logging, resume and core equations.
No Breeze hardware throughput or accuracy claim is introduced.
