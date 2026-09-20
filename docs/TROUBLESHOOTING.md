# Troubleshooting

Use a fresh TPU v5e-1 runtime. CPU/GPU-only sessions fail rather than changing the
backend. Free allocation is not guaranteed. Wait for the explicit stage/error
message; startup can be dominated by a multi-gigabyte model download and JIT.

On missing file, download or ETag failure, check the displayed path/model commit
and network quota; do not remove verification. On OOM, restart the runtime and
run just one model worker; this notebook does not claim a complete memory budget
for every Colab account. On repeated setup, the existing project worker is closed
before a replacement; never start two Breeze variants in one runtime.

First-window token mismatch stops with a local JSON report. This is not a WER
measurement. A translated-looking ASR-26 output can be the model's documented
Taigi-to-Chinese behavior. `nan` is not a token supported by Whisper-large-v2;
the fixed Chinese-output profile is `zh`.

Quality/timing flags preserve text rather than auto-deleting it. Raw model text,
filenames and logs may contain private content: remove identifying material
before sharing diagnostics. Cancelling upload does nothing. Browser download
failure leaves the complete file in the runtime; no output ZIP is created.

## `Missing special token: <|nospeech|>`

This was a v0.1.0 tokenizer compatibility bug, fixed in v0.1.1. Use the updated
notebook payload; restarting or reinstalling the old notebook alone cannot fix
it. See [the repair instructions](TOKENIZER_FIX.md). Later tokenizer errors now
report `tokenizer/configuration validation`, not the misleading model-download
stage. Do not guess token IDs or delete no-speech handling to suppress the error.
