# T22 — STRIX_LEAN vs Unsloth UD-IQ4_XS on Flash-Next (and an official-build dead end)

2026-08-29, same machine (Radeon 8060S, gfx1151), same fork build (`2c6309e3b`,
Vulkan/RADV), same protocol as the backend matrix (4 prompts x 3 reps, greedy, external
MTP drafter Q8_0, n_max 3, p_min 0.75, ctx 8192, `--no-mmap`, `-fit off`, warm-up
discarded). Arms differ only in the target GGUF.

## Quant verdict: STRIX_LEAN ahead everywhere

| tok/s (median of 3) | STRIX_LEAN (ours) | UD-IQ4_XS (unsloth) | delta |
|---|---|---|---|
| prose (Roma) | **31.5** | 27.8 | -11.7% |
| prose (Rinascimento) | **31.0** | 26.5 | -14.5% |
| counting 1-200 | **45.5** | 44.0 | -3.3% |
| alphabet | **36.0** | 34.7 | -3.6% |

Speculative acceptance is identical across the two quants (mean accepted length
3.0-3.25, per-position ~0.94): the gap is pure decode speed of the quantized kernels.
Prose — the workload that matters for agent sessions — shows the widest margin.
Production unchanged: LEAN + Vulkan + KV q8_0 + MTP n6.

## Engine arm (official llama.cpp): did not load, and here is the trail

Goal: run the same UD model on official master to isolate engine vs quant. Result: FAIL
at load for every GGUF we tried:

```
llama_model_load: error: check_tensor_dims: tensor 'blk.0.hc_attn_norm.weight' not found
```

Verification chain (each step kills one hypothesis):

1. Not a missing arch: the build's reported commit (`3173a5647`, 2026-08-29) contains
   `src/models/qwen4exp.cpp` and `hc_attn_norm` in `llama-arch.cpp` (PR #27742 merged on
   08-27); the error message itself proves the qwen4exp dispatch is active (otherwise
   `unknown model architecture`).
2. Not unsloth's sharding: a `llama-gguf-split --merge` of the 3 shards (93.7 GB,
   1,224 tensors verified with gguf-py) fails identically.
3. Not a shape/metadata mismatch: the tensor is present with shape `(10240,)` which is
   exactly `hc(4) x n_embd(2560)`; `hyper_connection.count=4`, `low_rank=320` are in the
   file; upstream computes `hc_dim = hc_mult * n_embd` — identical. And `check_tensor_dims`
   only says "not found" when the tensor is absent from the map (a dims mismatch prints
   "has wrong shape; expected ...").
4. The drafter failure is expected and legitimate: our MTP head declares `block_count=49`
   but only carries `blk.48`; official creates tensors for all 49 layers. Our fork has a
   dedicated draft-only path for this (see the qwen4exp-mtp branch).
5. What remains for the target: our `llama-official-vk` build does not match the source
   of its reported commit — CMake stamps the git HEAD, not the working tree. Most likely
   a stale/partial qwen4exp port was in the tree at build time. (A 2-month-old official
   CI binary said `unknown model architecture`, consistent with pre-#27742 master, and
   is not evidence about current master.)

Follow-up (cheap): clean rebuild of the official image from current master, then a
load-only probe with `-ngl 0` — the tensor check happens before any GPU allocation.
If it loads, the engine comparison reopens.

## Artifacts

- Full Italian report with rep-by-rep numbers:
  `docs/benchmarks/results-2026-08-29-t22-lean-vs-ud.md` in the source workspace
- Bench logs: `logs/bench-qwen4exp-mtp/bench-FRKUD.log`
- HF card updated with the quant table: commit `28aee5d5`
