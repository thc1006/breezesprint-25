# Publish BreezeSprint25

This archive does not create a remote repo or push files. Copy the contents of
`breezesprint-25/` to the root of a new public `thc1006/breezesprint-25` repository, branch `main`.
The notebook must remain named `BreezeSprint25.ipynb` for the badge. Before publishing:

```bash
python tools/build_notebook.py
python tools/check_release.py
python -m pytest -q
```

Use a new Colab runtime to test real 臺灣華語與中英混用 audio. Save a sanitized manifest and
check the words/times against the audio. Until then retain the experimental label.
Do not publish WhisperSprint performance figures as Breeze figures.

The notebook can be uploaded directly to Colab before a remote exists. To change
owner/repo URLs locally, run `python tools/rebrand.py --owner NAME --repo NAME`.
The helper changes files only and rebuilds the notebook. Metadata helper is a dry
run unless explicitly given `--apply`. No automatic commit or push is performed.

## v0.1.1 patch release

Replace source files AND the notebook, then run the offline tests and
`JAX_PLATFORMS=cpu python tools/check_model_assets.py` before publishing a claim
about the real model metadata. The online check uses no TPU or model weights.
Old saved copies of the notebook retain their old embedded source until replaced.
