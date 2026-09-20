# Model contract

Source: https://huggingface.co/MediaTek-Research/Breeze-ASR-25  
Pinned revision: `cffe7ccb404d025296a00758d0a33468bec3a9d0`  
Research: https://arxiv.org/abs/2506.11130  
Review date: 2026-09-21.

This targets the ASR-25 Mandarin/code-switching checkpoint, not Taigi-specific ASR-26. Traditional Chinese is the model tendency, not a per-character guarantee.

The files and architecture are listed in `src/breezesprint25/model.json`. No weights are
bundled. The snapshot is deliberate, not a claim to follow the latest mutable
`main` branch. Files are resolved only at the full commit and checked against
publisher ETags: SHA256 for LFS, Git-blob SHA1 for ordinary files. All downloaded
SHA256 values are written to local manifests. This trusts the publisher and HTTPS;
a digest does not certify model accuracy or publisher identity by itself.

ASR-26 is a two-shard F32 checkpoint. An index must agree with the actual contents
of every shard. Native BF16 (ASR-25) has NumPy dtype kind `V`; it is explicitly
accepted as `ml_dtypes.bfloat16`, not rejected as a non-floating tensor. Nonfinite
values, missing/extra keys and untied output embeddings fail closed.

The byte-BPE output decoder uses each model's own vocabulary/added-token mapping,
including a contiguous 1501-token timestamp range. It is not a general-purpose
text encoder. Only the literal space and fixed special-token prefix are encoded.
UTF-8 bytes are assembled across tokens before conversion, preserving Chinese,
English and numbers as generated. Special tokens are not written as ordinary text.

Generation config and model card are source-derived. The fixed `zh` prefix,
greedy decoding, BF16 inference and segment-export choices are our implementation
choices. They have not been shown to reproduce the source paper's scores.
