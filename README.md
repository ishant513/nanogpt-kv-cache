# nanogpt-kv-cache

A KV cache for nanoGPT, with a correctness-gated, benchmarked decode speedup —
and a roadmap toward speculative decoding on top of it.

## The idea
Vanilla nanoGPT recomputes the full forward pass over the whole context every
step. This adds an incremental key/value cache so each decode step only does work
for the new token. The cached path is verified to produce bit-identical output to
the original, then benchmarked.

## Results
_(fill in after running the benchmark)_
- Correctness: cached output identical to uncached (greedy) — ✅ / ❌
- Speedup: __x on __ (cpu/mps/cuda) for 200 new tokens

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python benchmark_stage1.py
```
First run downloads GPT-2 124M (~500MB) via Hugging Face.

## Roadmap
- [ ] Stage 1 — KV cache + verified speedup
- [ ] Stage 2 — speculative decoding reusing this cache
- [ ] Stage 3 — KV-cache quantization / prefix sharing

## Attribution
`model.py` and `LICENSE-nanoGPT` are from nanoGPT by Andrej Karpathy (MIT).
All KV-cache and benchmark code is original.
