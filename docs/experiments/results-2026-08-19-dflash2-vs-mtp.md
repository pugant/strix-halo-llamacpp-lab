# Results — T7 A/B: DFlash2 (draft-dflash) vs MTP n6 · Qwen3.8-27B STRIX_LEAN

**Date:** 2026-08-19 · **Plan:** internal experiment plan `2026-08-19-dflash2-vs-mtp-ab.md` (not included in this repo)
**Setup:** Vulkan RADV gfx1151, dedicated GPU (llm-service stopped, restart+health at the end of the run),
LEAN 13.8 GB, `-c 16384 -fa on --jinja`, temp 0, p_min 0.75, p_split 0.10, warm-up discarded,
TREATMENT marker in the logs (`logs/bench-dflash-vs-mtp/`). Script: `scripts/bench-dflash-vs-mtp.sh`.
**Porting:** branch `dflash2` (cherry-pick PR #27342 `ba2485545` + 4 fixes `ebf1cc855`),
image `docker-llm-service:vulkan-fork-dflash2`; drafter `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
(1.9B, 1.1 GB, block_size 8 → n-max 7).

## Tok/s (2 Italian prose / 2 deterministic)

| Arm | prose 1/2 | det 1/2 | Δ det vs MTP6 |
|---|---|---|---|
| MTP6 (control, ckpt7) | 19.6 / 20.2 | 45.2 / 26.1 | — |
| **DF7** | 14.2 / 15.0 | **57.4 / 36.3** | **+27% / +39%** |
| DF5 | 17.5 / 15.9 | 52.2 / 39.5 | +16% / +51% |

**Det 57.4 tok/s = new record for the LEAN on this hw** (previous 45.4).

## Acceptance (from the log, per prompt)

| Arm | prose (mean len) | det (mean len) |
|---|---|---|
| MTP6 | 2.48 (per-pos 0.93, 0.33, 0.13, …) | 6.26 / 3.94 |
| DF7 | 2.43 (per-pos 0.62, 0.33, 0.20, …) | **7.59** (0.99, 0.97, 0.96, 0.94, 0.92, 0.91, **0.90**) / 6.03 |
| DF5 | 2.29 | 5.80 (1.00, 0.98, 0.97, 0.93, 0.92) / 5.23 |

**Clear DFlash2 signature**: on predictable content the per-position acceptance stays
≥0.90 up to position 7 (MTP collapses after the 1st: 0.93→0.33). On Italian prose
acceptance is ON PAR with MTP (2.43 vs 2.48) but the round costs more → tg −26/28%.

## Physical interpretation

- det: acceptance ~7.5 tokens/round × verify batch → tg almost linear in the number of
  accepted tokens (SSM-dominated, KV 69.6 KB/token) → +27-39%.
- prose: acceptance ~2.4 → the fixed cost of the DFlash round (decode of the WHOLE
  noise block at 8 positions always + KV encoder/injection per token) is not amortized
  → −26/28% penalty vs MTP, which stops the AR drafting when confidence drops.

## "Lossless" spot-check (greedy, same prompt)

| Comparison | Outcome |
|---|---|
| NONE vs MTP6 | reasoning divergence at 422 char (common) |
| NONE vs DF7 | reasoning divergence at 422 char |
| MTP6 vs DF7 | identical up to 1497 char |

The NONE-vs-spec divergence at the SAME point for both drafters = **batched verify
numerics** (the target decodes a multi-token batch vs single → argmax flip on nearly
tied tokens), a PRE-EXISTING characteristic of the stack (MTP production included),
NOT a defect of the DFlash2 porting. DFlash2's "lossless" claim is valid only within
this caveat.

## Verdict (plan criteria)

**NO-GO to production replacement** (prose −26% >> the −3% threshold), **BUT**:
- absolute det record (57.4), dominant across the whole deterministic workload;
- open thread: adaptive drafter (DFlash for coding/counting/agentic-det, MTP for
  prose) or workload-dependent n-max; possible PR of the porting to charlie.

---

## Addendum 1 — DF3 n-max curve + agentic workload (19/08 afternoon)

Internal experiment plan `2026-08-19-dflash2-df3-agentic.md` (not included in this repo), script
`scripts/bench-dflash-df3-agentic.sh` (not included), log `logs/bench-dflash-df3-agentic/`.
Same protocol (dedicated GPU, temp 0, p_min 0.75, TREATMENT marker).

### Agentic workload (MTP6 vs DF7, coding/JSON/log prompts)

| Prompt | MTP6 | DF7 | Δ |
|---|---|---|---|
| coding (10 functions) | 27.5 | 33.8 | **+23%** |
| JSON (30 objects) | 35.0 | 36.2 | +3% |
| log (20 fixed-format lines) | 28.0 | 35.8 | **+28%** |

Acceptance mean: MTP 3.97-4.98 vs **DF7 5.63-5.90**; DF7 per-pos on agentic
(0.92, 0.77, 0.71, 0.65, 0.56, 0.52, 0.51) — stays >0.50 up to pos 7 where
MTP drops to 0.25-0.50. JSON is the case already nearly saturated for MTP (floor).
**Criterion ≥+10% on 2/3 prompts: PASS → per-workload routing confirmed.**

### DFlash n-max curve (standard prompts)

| Arm | prose 1/2 | det 1/2 |
|---|---|---|
| MTP6 | 19.6 / 20.2 | 45.2 / 26.1 |
| DF3 | 17.6 / 18.2 | 39.5 / 34.5 |
| DF5 | 17.5 / 15.9 | 52.2 / 39.5 |
| DF7 | 14.2 / 15.0 | 57.4 / 36.3 |

DF3 brings prose to quasi-parity (−10%) BUT det1 drops below MTP (39.5 vs
45.2): the optimal **single**-drafter compromise is **DF5** (det +16/+51%,
prose −11/−21%). DF3 is dominated by DF5 (prose on par, det much lower).

### T7 conclusion (final)

The answer to "prose" is not n-max but **per-workload routing**:
- prose/chat → MTP6 (19.6-20.2)
- agentic/coding/det → DFlash2 (33.8-57.4, +23-39%)
- possible single profile without routing → DF5.

Possible future implementation: per-request flag in the fork or dual instance;
lightweight alternative: different server parameters per slot/service.
