# Third-party notices

The inference engine is an original compact JAX implementation of the standard
Whisper architecture. Its equations, audio decoding conventions, timestamp
constraints were checked against the
OpenAI Whisper implementation. It is not endorsed by Google, OpenAI, or Hugging Face.
The new notebook glue and inference implementation may be used under the MIT
terms below. Preserve the OpenAI notice for the referenced/adapted decoding logic.

Model weights, tokenizer assets, JAX, SciPy, NumPy, SafeTensors, Hugging Face Hub,
FFmpeg, and their dependencies retain their own licenses. Weights and
third-party binary packages are downloaded at runtime, not redistributed here.
The notebook uses one pinned MediaTek-Research/Breeze-ASR-25 checkpoint.
No model weights or recorded user audio are bundled. The notebook UI was inspired by a
user-supplied compact Colab example; its faster-whisper backend is not used.

OpenAI Whisper source: https://github.com/openai/whisper
License: https://github.com/openai/whisper/blob/main/LICENSE

## OpenAI Whisper license

MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Breeze model used by this application

Source: https://huggingface.co/MediaTek-Research/Breeze-ASR-25
Revision: cffe7ccb404d025296a00758d0a33468bec3a9d0
Model license: Apache License 2.0. Model weights are not redistributed here.
Publisher retains model ownership. This application is an independent integration.
Research reference: https://arxiv.org/abs/2506.11130

Modelcard descriptions are paraphrased with attribution. Test vocabularies and
synthetic tensor fixtures are application tests, not distributed model weights.

The small `tests/fixtures/publisher-token-contract.json` contains selected special
IDs transcribed from MediaTek-Research's published Breeze-ASR-25 tokenizer and
generation files, with source URLs. They remain under their publisher's Apache-2.0
model-asset terms. Ordinary fixture vocabulary and test tensors are synthetic.
