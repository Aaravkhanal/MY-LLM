# Open-source acknowledgements

`aarav-lm` is my own project. The code in this repository is written for it; the following
public works informed design decisions, and the original authors deserve the credit for the ideas.

| Work | Author / licence | What it informed here |
|---|---|---|
| [nanoGPT](https://github.com/karpathy/nanoGPT) | A. Karpathy, MIT | Splitting AdamW parameters into decayed / non-decayed groups; scaled init of residual projections (GPT-2 paper); flat token files read through `np.memmap` |
| [llama2.c](https://github.com/karpathy/llama2.c) | A. Karpathy, MIT | The Llama-style block: RMSNorm, RoPE, SwiGLU with a 2/3·4d hidden width, GQA head repetition |
| [llm.c](https://github.com/karpathy/llm.c) | A. Karpathy, MIT | The idea of token shards with a fixed binary header and magic number; parity-testing habits |
| [nanogpt-mps-lab](https://github.com/poting-lin/nanogpt-mps-lab) | P. Lin | Pre-shifting SFT targets with `-1` for masked positions; the KV-cache-on-Apple-Silicon approach |
| Hugging Face [`tokenizers`](https://github.com/huggingface/tokenizers) | Apache-2.0 | The byte-level BPE trainer/encoder used for our own tokenizer (a runtime dependency) |
| PyTorch | BSD-3 | Everything numerical |

## Datasets

See [DATASETS.md](DATASETS.md).
