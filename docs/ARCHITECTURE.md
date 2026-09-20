# Architecture

`colab.py`: stdlib-only notebook controller, isolated environment, persistent
worker, upload-only UI, automatic timestamped TXT download. `worker.py`: TPU
selection, pinned assets, schema checks and request loop. `assets.py`: commit and
content verification. `checkpoint.py`/`loading.py`: single/sharded BF16/F32 input
and host packing. `core.py`: unchanged numerical equations from WhisperSprint.
`engine.py`: fixed Chinese-output control tensors. `vocabulary.py`: byte-BPE
output decoding. `frontend.py`: 16 kHz audio, 80-bin mel, timestamp parsing and
first-window repeatability. `pipeline.py`: sequential seek, resume, journaling and
prefetch. `text_output.py`: validates and writes the final timestamped TXT.

No new model is trained. The repo implements an independent runtime for
`MediaTek-Research/Breeze-ASR-25`. Model ownership stays with the original publisher.

`preflight.py`: validates model dimensions, preprocessing settings, complete
vocabulary IDs, no-speech aliases and generation/prefix IDs before requesting
weight shards. `tools/check_model_assets.py` runs the same check on real pinned
metadata without a TPU or weight download. Startup now has distinct metadata,
tokenizer-validation and weight-download stages.
