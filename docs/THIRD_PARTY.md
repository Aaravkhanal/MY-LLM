# Open-source acknowledgements

`aarav-lm` is my own project, but several well-known open-source works informed the design choices I made.
Where I adopted a technique or algorithm from existing code I note it here.

---

## nanoGPT

**Author**: Andrej Karpathy  
**License**: MIT  
**Repo**: https://github.com/karpathy/nanoGPT

The idea of separating 2-D weight matrices from 1-D biases / norms when configuring AdamW
came from nanoGPT's `configure_optimizers`.  The MFU estimation formula also follows the same
approach.

---

## llama2.c

**Author**: Andrej Karpathy  
**License**: MIT  
**Repo**: https://github.com/karpathy/llama2.c

The RoPE, RMSNorm, and SwiGLU formulations I used match those in `llama2.c/model.py`.
GQA head-expansion (repeat_kv) also comes from there.

---

## llm.c

**Author**: Andrej Karpathy  
**License**: MIT  
**Repo**: https://github.com/karpathy/llm.c

Used `train_llama3.py` as a reference for how the modern LLaMA-3 architecture
differs from GPT-2, particularly around how SwiGLU hidden dimensions are computed.

---

## nanogpt-mps-lab

**License**: MIT  
**Repo**: local clone

The KVCache layout (pre-allocated tensor, index-slice writes) and the idea of
using `torch.amp.autocast(device_type="mps", dtype=torch.bfloat16)` for training
on Apple Silicon came from this repository.  The bits-per-byte evaluation idea
also originates here.

---

All four repositories are MIT-licensed.  Credit belongs to their respective authors.
