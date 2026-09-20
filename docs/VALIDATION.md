# Validation scope — v0.1.1

191 local tests passed with no failures or skips. Results and exact environment
versions are in `evidence/validation.json`, `cpu-tests.log`, and `cpu-tests.xml`.
They cover CPU/synthetic numerical kernels, BF16 and sharded loading, decoding,
TXT export, worker lifecycle, failure handling and mocked Colab APIs.

The no-speech regression uses the publisher-documented `<|nocaptions|>` spelling
and control IDs. Ordinary vocabulary is explicitly synthetic; the full publisher
vocabulary was NOT downloaded in this build environment because the network/DNS
request failed. The original source reproduces the reported exception with that
fixture; the fixed source resolves ID 50362, prefix and all 1501 timestamp IDs.

A separate online CI job runs `tools/check_model_assets.py` on the real pinned
metadata. It never downloads weights and fails instead of skipping on network
errors. It has not been executed remotely by the packager. Runtime performs the
same metadata check before downloading the weight files.

Full Breeze-ASR-25 weights were NOT loaded/executed here, and there was no physical
TPU or live Colab browser test. Fixing the reported initialization failure is not
proof of full-model performance, target-language correctness, WER/CER, long-audio
completeness or exact timestamp alignment.

`tokenizer_validation.json` records tokenizer checks separately from
`hardware_probe_passed`, `first_window_repeatability_passed` and
`model_accuracy_validated=false`. A metadata pass is not an inference pass.
