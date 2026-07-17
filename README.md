# nanogpt-kv-cache

This project exists to turn nanoGPT into a small but real demonstration that you can do inference-systems engineering. This emphasizes kv cache management, prefill/decode split, speculative decoding, the masking and position bookkeeping that make incremental generation correct.

Nanogpt-kv-cache adds the optimizations that real LLM systems utilize. Vanilla nanoGPT recomputes the full forward pass over the entire context on every decode step, while this project layers on the techniques production serving systems use to avoid that. Every optimization is verified to produce bit-identical output to the unoptimized baseline, then benchmarked for the speed it adds.

## Results
_(fill in after running the benchmark)_
- Correctness: cached output identical to uncached (greedy)
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
