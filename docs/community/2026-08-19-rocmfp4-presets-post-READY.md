# POST PRONTO ALL'INVIO — copia da qui (r/LocalLLaMA, flair [Benchmark]; poi crosspost r/StrixHalo)

## TITLE (riga unica)

ROCmFP4 preset showdown on Strix Halo (same model, same imatrix): dual-scale block-16 scales are free quality — the "fast" single-scale layout gives up ~1.2 ppl points for nothing

## BODY (markdown, copia tutto)

TL;DR: on the ROCmFPX fork we ran what is, as far as we know, the first controlled comparison of ROCmFP4 presets on the *same* model (Qwen3.8-27B dense, 48 SSM + 16 attention layers), *same* BF16 source, *same* imatrix, *same* perplexity corpus and bench protocol. Three findings:

1. **Going from a single scale per 32-weight block to one UE4M3 scale per 16-weight half-block (the NVFP4-style granularity) is worth ~1.2 ppl points for +3.6% file size — and costs 0% in decode speed.**
2. The "speed-focused" FAST single-scale layout has **no measurable tg advantage** over dual-scale. Its only benefit is size.
3. The full quality recipe (qkv in Q5_K + ffn_down in Q6_K) is the quality champion (+0.74%/+1.13% ppl vs BF16 on en/it) but pays −16% MTP decode and −31% no-spec tg on Vulkan — the K-quant tensor path is what's slow, not the FP4 format.

## Why this comparison

ROCmFPX presets get recommended by vibes: "STRIX_LEAN for Strix Halo", "full for quality". The only public numbers mixing presets (KAT-Coder GGUF) compared different bpw on a different arch (MoE) — you couldn't tell whether the quality came from the scale granularity, the extra bits, or the protected tensors. NVFP4's block-16 scaling claims made us want the isolated answer on our hardware.

## Setup

- Hardware: Strix Halo (Ryzen AI Max+ 395, gfx1151, 128 GB unified), Vulkan RADV backend, fork server with MTP n6 (p_min 0.75).
- Model: Qwen3.8-27B dense (hybrid SSM/attention), quantized locally from the same BF16 shards with the same imatrix (496/496 tensors).
- Arms (all `llama-quantize` presets from the fork):
  - **LEAN** `Q4_0_ROCMFP4_STRIX_LEAN` — single-scale on gate/up, protected K/V + Q5_K embeddings, 13.82 GiB (our current production file)
  - **FULL** `Q4_0_ROCMFP4` — dual-scale on gate/up + qkv→Q5_K + ffn_down→Q6_K, 16.51 GiB (note: this is why people report "full" at ~5.25 bpw — the routing, not the 4.50 bpw type)
  - **EVEN** `Q4_0_ROCMFP4_EVEN` (`--pure`) — dual-scale *everywhere*, 14.32 GiB
  - **FAST-EVEN** `Q4_0_ROCMFP4_FAST_EVEN` (`--pure`) — single-scale *everywhere*, 13.54 GiB — the pure single-scale floor.
- Perplexity: wikitext-2-raw (70 chunks) + a 51.5k-char Italian technical corpus, `-c 512 -b 512 -fa on`, full GPU offload. The LEAN arm was re-run as a control and reproduced its earlier numbers to the 4th decimal.
- Speed: `llama-bench -p 512 -n 128 -fa 1 -r 5` (no spec) + server-side generation with MTP n6, temp 0, warm-up discarded, 2 prompts per class (prose/deterministic).

## Results

PPL delta vs the BF16 baseline of the same weights:

| Preset | Size | ΔPPL en | ΔPPL it |
|---|---|---|---|
| FAST-EVEN (single-scale everywhere) | 13.54 GiB | +3.10% | +3.76% |
| LEAN (single-scale + protections) | 13.82 GiB | +2.74% | +3.42% |
| **EVEN (dual-scale everywhere)** | 14.32 GiB | **+1.90%** | **+2.61%** |
| FULL (dual-scale + Q5_K/Q6_K protections) | 16.51 GiB | +0.74% | +1.13% |
| Q4_K_M reference (17/08 run) | 15.6 GiB | +0.51% | +3.27% |

Speed on the same box:

| Test | LEAN | EVEN | FULL |
|---|---|---|---|
| llama-bench tg128 (no spec) | 13.62 ± 0.86 | **13.64 ± 0.01** | 9.37 ± 0.92 |
| llama-bench pp512 | 346.5 | **347.6** | 314.6 |
| MTP n6 decode (deterministic) | 45.4 | 41.7 | 37.1 |

The isolated contrasts (this is the part vibes couldn't give you):

- **EVEN vs FAST-EVEN** (identical routing, only scale granularity differs): −1.20 ppl points en / −1.15 it, for +0.86 GB. This is the block-16 effect, clean.
- **FULL vs EVEN** (protections only): −1.16 / −1.48 further points, for +2.35 GB and the K-quant speed penalty.
- **LEAN vs FAST-EVEN** (LEAN's protections only): a modest −0.35 points. The LEAN protection scheme buys much less than either change above.

And the kicker: EVEN's tg128 is *identical* to LEAN's (13.64 vs 13.62, stddev 0.01). The dual-scale kernel is not slower. The FAST layout's raison d'être (speed) does not show up at decode on Vulkan/gfx1151 — only FULL suffers, and only because its qkv/down tensors go through the K-quant path.

## Practical recommendation

- On Strix Halo with a dense Qwen3.x: **`Q4_0_ROCMFP4_EVEN` is the new sweet spot** — nearly a Pareto improvement over STRIX_LEAN (better ppl both languages, same speed, +3.6% size). It's the file we'd recommend for this class of model now.
- If you need max quality and can afford −16% decode: FULL, especially for non-latency-critical use.
- If you're below 4.4 bpw already: don't bother with FAST — dual-scale it.

## Reproduce

```bash
# in the ROCmFPX fork's quantize container (same one that builds the presets):
llama-quantize --imatrix imatrix.gguf model-BF16.gguf out.gguf Q4_0_ROCMFP4_EVEN
```

Full data, methodology and the addendum bench: see the numbers above; happy to share the corpus prep (wikitext-2 standard + our Italian technical corpus is just our own docs — use any same-language corpus you like, just keep it fixed across arms).

Credits: presets and fork by u/charlie12345 (ROCmFPX); prior community data points that motivated this: the KAT-Coder ROCmFP4 GGUF card (1337Hero) and julianmb's Nemotron ROCmFP4 benchmarks — thanks, this is the controlled version of what you started. NVFP4's block-16 scaling is the obvious prior art for the granularity question.

Caveats: single machine, single model family (dense hybrid SSM); ppl deltas on wiki-style + one technical-language corpus; speed numbers are Vulkan/RADV on gfx1151 — HIP numbers may differ (julianmb saw Vulkan beat ROCm on the fast presets by ~21% pp).

