# Research sources and boundaries

Reviewed 2026-09-21.

- Official model card: https://huggingface.co/MediaTek-Research/Breeze-ASR-25
- Selected immutable source tree: https://huggingface.co/MediaTek-Research/Breeze-ASR-25/tree/cffe7ccb404d025296a00758d0a33468bec3a9d0
- Original application: https://github.com/thc1006/whispersprint
- JAX TPU examples and precision guidance: https://github.com/jax-ml/jax/tree/main/cloud_tpu_colabs
- JAX compilation cache trust: https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- Colab hardware and quotas: https://research.google.com/colaboratory/faq.html
- GitHub Python CI reference: https://github.com/actions/setup-python

The model card describes the publisher's model, not this JAX port's quality.
No published WER/CER or performance table is adopted as an application result.
The fixed commit is an intentionally reviewed source, not a floating-latest claim.
Weight and vocabulary bytes are fetched and checked by the user's runtime, not
pretended to have been fully downloaded during packaging.

Breeze-ASR-25 targets Taiwanese Mandarin and Mandarin–English code-switching.
Breeze-ASR-26 targets Taigi input with Mandarin Chinese-character output, not
native Taigi orthography. Both are separate use cases, not a ranked replacement.
