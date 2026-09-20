# Performance scope

No Breeze throughput or speedup has been measured on a TPU for this release.
Do not copy Whisper Turbo figures into these projects. Both Breeze checkpoints
have 32 decoder layers. The 26 snapshot uses about 6.18 GB of F32 source weights;
loading as BF16 reduces device storage, not the original network download size.

Model preparation overlaps upload. One worker stays resident; static-shape JIT,
fixed K/V caches, host-packed groups and bounded file prefetch avoid redundant
work. No runtime is kept alive to bypass Colab limits. No global clock settings,
quota tricks or system-wide THP settings are changed.

First nonzero window executes twice for repeatability; both passes are included
in wall time, while journal component sums describe the committed first pass.
First-job wall time can include compilation. Later jobs can reuse JIT and weights.
Model setup, upload, file work, local export and browser transfer are not the same
interval. Fully or partly resumed work has no full-file RTF claim. The first-window
report is a repeatability check, not benchmark accuracy.

Official sources:
- https://docs.jax.dev/en/latest/benchmarking.html
- https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- https://cloud.google.com/tpu/docs/v5e
- https://research.google.com/colaboratory/faq.html

Version 0.1.1 intentionally checks the small tokenizer/configuration files before
starting weight downloads. This fails early for incompatible metadata rather
than discovering a token-name error after transferring gigabytes. A valid local
HF cache is still reused. No new inference speedup is claimed by this bug fix.
