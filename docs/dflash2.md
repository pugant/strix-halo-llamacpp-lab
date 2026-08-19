# DFlash2 speculative drafting (block-diffusion drafter)

This branch adds **DFlash2** draft-model support on top of this fork, via
upstream llama.cpp PR [#27342](https://github.com/ggml-org/llama.cpp/pull/27342)
("spec: add DFlash2 support (local convolution + candidate selector)") by
Jian Chen, cherry-picked and adapted to this fork's internals (commit history
on branch `dflash2`: cherry-pick `ba2485545` + adaptation `ebf1cc855` and
follow-ups).

## What it is

DFlash2 (Inco AI, [blog](https://inco.ai/blog/dflash2/)) is a **block-diffusion
drafter**: it predicts a whole block of tokens in a single forward pass, keeps
top-k candidates per position and walks the coherent path with a low-rank
bilinear **path selector**; a **two-tap local convolution** repairs decay toward
the end of the block. Verification is standard lossless rejection sampling.

Works on **Vulkan** (tested on gfx1151 / Strix Halo, RADV) and should work on
CUDA/Metal (untested here).

## Usage

```bash
# drafter GGUF (1.9B): https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF
llama-server -m Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  -ngl 999 -fa on --jinja -c 16384 \
  --spec-type draft-dflash \
  --spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-draft-ngl all --spec-draft-n-max 7 \
  --spec-draft-p-min 0.75
```

The drafter metadata (`dflash.block_size`, `dflash.selector_top_k`,
`dflash.selector_rank`, `dflash.conv_kernel_size`, `dflash.conv_group_size`,
`dflash.target_layers`) is read from the GGUF automatically. With
`block_size = 8` the effective n-max is 7 (the server clamps).

## Measured results (Qwen3.8-27B STRIX_LEAN, Vulkan RADV, gfx1151, temp 0,
p_min 0.75, ctx 16384, single stream; full data in the workspace report)

| Workload | MTP n6 | DFlash2 n7 | DFlash2 n5 |
|---|---|---|---|
| deterministic (counting) | 45.2 / 26.1 | **57.4 / 36.3** | 52.2 / 39.5 |
| code (repetitive Python) | 27.5 | **33.8** | — |
| fixed-format logs | 28.0 | **35.8** | — |
| structured JSON | 35.0 | 36.2 | — |
| free prose (Italian) | **19.6 / 20.2** | 14.2 / 15.0 | 17.5 / 15.9 |

Interpretation: block-diffusion acceptance stays ≥0.5 per position up to
position 7 on structured/predictable content (MTP collapses after position 1:
0.93 → 0.33), giving +23-39% tok/s there. On free prose the acceptance is on
par with MTP (~2.4 tokens/round) but the round has a fixed cost (the whole
n+1-position noise block is always decoded, plus the per-token encoder/KV
injection), so MTP wins. **Route by workload**: MTP for chat/prose, DFlash2 for
code/agents/deterministic output; n-max 5 is the best single-drafter compromise.

## Porting notes (fork vs upstream)

Adaptations applied on top of the upstream commit (see workspace handoff for
the full list):

1. API renames: `llama_*_embeddings_nextn` → `llama_*_embeddings_pre_norm`,
   `res->t_h_nextn` → `res->t_h_pre_norm` (this fork's pre-existing names);
   `ml->get_tensor_meta` → `ml.get_tensor_meta` (reference form).
2. The upstream framework-level `build_post_sampling()` hook does not exist
   here: it is called explicitly at the end of the `graph<false>` constructor.
3. The KV-injection (embd-batch) decoder paths clear `res->t_h_pre_norm` to
   avoid stale pointers from the encoder graph (backend == nullptr assert).
4. The noise-block decode keeps `embeddings_pre_norm` enabled and unmasked for
   DFlash2 (the legacy fork toggle disabled it for non-DSpark drafters, which
   deallocated the lattice buffer mid-flight).
5. Logits are requested on every noise-block position: the selector lattice
   reads the full-block logits.

## Related

- Target model (this fork's quant): [pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN](https://huggingface.co/pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN)
- Drafter GGUF: [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
- Upstream PR: [ggml-org/llama.cpp#27342](https://github.com/ggml-org/llama.cpp/pull/27342)
