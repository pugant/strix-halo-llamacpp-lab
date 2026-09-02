# Patches

Every feature of this lab is carried as a `git am`-clean series under [`patches/`](patches/) — but you rarely need to apply anything: the [`rocmfpx/`](rocmfpx/) snapshot **already contains every merged series** (it is the fork at `6144779cc`, buildable as-is — see the [README build section](README.md#build)). The patches exist to read the history, cherry-pick a single feature, or re-create a branch.

Apply notes — the non-obvious ones:

- Every series applies with `git am` onto the **fork base named in the Base column** (the fork is [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX), a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); a series whose base is "fork `main`" will not apply on ggml-org master).
- [`patches/drafter-routing/0001-drafter-routing-mtp-dflash-per-request.patch`](patches/drafter-routing/0001-drafter-routing-mtp-dflash-per-request.patch) is **one file containing the full 14-commit series** — `git am` splits it back into its 14 commits.
- [`patches/upstream-llamacpp/0001-server-reasoning-budget-forced-newline.patch`](patches/upstream-llamacpp/0001-server-reasoning-budget-forced-newline.patch) is a **plain diff for `git apply`** against **ggml-org master `3dc7285`**, not the fork — see its [README](patches/upstream-llamacpp/README.md).
- [`patches/t25-ple-disk/`](patches/t25-ple-disk/) applied on fork `main` yields `bc85fcb1d`; [`patches/t23-kv-disk-persist/`](patches/t23-kv-disk-persist/) applied on top yields `f629365da` — the snapshot state before the `t27` and `optim-camp` series.
- [`patches/optim-camp/`](patches/optim-camp/) applied on `dadc23e44` reproduces the **tree** of `6144779cc` — the commit the [`rocmfpx/`](rocmfpx/) snapshot in this repo matches. The patch messages carry a `Co-Authored-By` trailer, so the shas after `git am` differ from the fork's; the tree does not.

## Patch index

| Series | Base | Purpose | Upstream status |
|---|---|---|---|
| [`spec-cache-trailing-rollback/`](patches/spec-cache-trailing-rollback/) | fork `main` (0001+0003 merged there via PR #69) | Spec-boundary cache salvage + reasoning-budget request surface (9 patches, breakdown below) | partially merged — PR #69 |
| [`drafter-routing/`](patches/drafter-routing/) | fork `main` (branch `drafter-routing`) | Dual draft contexts, per-request routing, drafter-tagged cache, boot fallback, metrics | included in [`rocmfpx/`](rocmfpx/) |
| [`reasoning-pressure/`](patches/reasoning-pressure/) | fork branch `reasoning-pressure` | Reasoning-pressure steering (notice + squeeze) — NO-GO experiment, documented, not for production | archived |
| [`spec-verify-log/`](patches/spec-verify-log/) | fork branch `spec-verify-log` | Per-position acceptance instrumentation of the verify batch | instrumentation only, not in the runtime build |
| [`upstream-llamacpp/`](patches/upstream-llamacpp/) | **ggml-org master `3dc7285`** (`git apply`) | The reasoning-budget forced-newline fix as prepared against ggml-org master | **PR archived, never sent** — see the patch dir README for the story |
| [`ckpt-device-split-restore/`](patches/ckpt-device-split-restore/) | fork branch `ckpt-device-split-restore` | Accept multi-range device state saves on checkpoint restore — fixes the 1-in-16 restore refusal that forced a full re-prefill (+ `llama-state-split-test` repro tool) | included in the `rocmfpx/` snapshot |
| [`f4-rollback-fix/`](patches/f4-rollback-fix/) | fork `main` (pi-stack `rs3` lineage) | Reset the MTP drafter state on partial-reject rollback — removes ghost tokens (p0-reject 0.272 → 0.161) | included in the `rocmfpx/` snapshot |
| [`hipcub-enablement/`](patches/hipcub-enablement/) | fork branch `pi-f1-hipcub` | Enable the hipCUB paths for top_k/argsort/cumsum on HIP (port of ggml-org PR #27874) | NO-GO by measurement — pp at 131k context −42%; not in the runtime build |
| [`ngram-drafter-instrumentation/`](patches/ngram-drafter-instrumentation/) | fork `main` | Per-drafter draft-round counters for spec-decoding engagement (T20/F3 precondition probe) | included in the `rocmfpx/` snapshot |
| [`qwen4exp-mtp/`](patches/qwen4exp-mtp/) | fork branches `qwen4exp-mtp` → `qwen4exp-conv-ring-slots` | Full qwen4exp architecture + external MTP drafter (`-md`), incl. the conv/PLE ring-slot rollback fix and the reasoning-budget warn window (20 commits) | included in the `rocmfpx/` snapshot (branch `qwen4exp-conv-ring-slots`) |
| [`t10/`](patches/t10/) | fork branch `t10` | `Q2_3_ROCMFPX_MIX` 2/3-bit mixed preset family (+ V2 variant) — the round-budget compression lever | NO-GO by measurement (perplexity gate ~5×); presets kept for the record |
| [`t11/`](patches/t11/) | fork branch `t11` | Fused draft chain + verify dispatch switch — both levers on the ~38 ms/round software residue, behind default-off flags | NO-GO by measurement — stays on branch `t11`, not in the runtime build |
| [`t25-ple-disk/`](patches/t25-ple-disk/) | fork `main` → result `bc85fcb1d` | PLE n-gram table disk offload — `--ple-disk` reads table blocks on demand from the GGUF itself (15 patches: 12 base + 3 v2) | not submitted |
| [`t27-ple-store-v2-0001-*.patch`](patches/) | `f629365da` → result `dadc23e44` | ple-store v2: `--ple-disk` extended to the 2-bit `Q2_0_ROCMFPX` PLE (generic row-bytes via ggml traits, dual-type allowlist, bit-exact test battery) — required to serve the ROCmFP2MIX-64GB build | not submitted |
| [`optim-camp/`](patches/optim-camp/) | `dadc23e44` → tree of `6144779cc` | Per-round graph reuse (QSA/PLE custom inputs learn `can_reuse`: rebuilds 1025 → 6) + dense decode when the QSA budget covers the cache — plain tg512 +18.6% HIP / +32.5% Vulkan, ppl unchanged to the 4th decimal, bit-identical greedy fingerprints | not submitted |
| [`t23-kv-disk-persist/`](patches/t23-kv-disk-persist/) | fork `main` @ `bc85fcb1d` → result `f629365da` (the `rocmfpx/` snapshot before `t27`/`optim-camp`) | Persistent cross-restart prompt-cache library — `--cache-disk-persist`: ds4-inspired entries with hit-decay eviction (6 h half-life), crash-safe commit-by-rename (sidecar written last), boot adoption/GC, CRC verify-then-load (12 patches: 9 base + 3 multimodal save-path fixes) | not submitted |

## The 9-patch `spec-cache-trailing-rollback` series

Breakdown of [`patches/spec-cache-trailing-rollback/`](patches/spec-cache-trailing-rollback/):

- `0001` + `0003` — bounded trailing rollback at the spec boundary (**merged upstream** via [charlie12345/ROCmFPX#69](https://github.com/charlie12345/ROCmFPX/pull/69))
- `0002` — checkpoint-based rollback for the spec-boundary cache
- `0004` — push all batch/verify rows into the MTP boundary state
- `0005` — forced-end sequence prefixed with a newline, for cache round-trip
- `0006` — accept `thinking_token_budget` (vLLM name) as an alias
- `0007` — checkpoint-based salvage (the ~91% prefill figure)
- `0008` — accept `reasoning_budget_message` per request (OpenAI-compat)
- `0009` — newline between wrap-up message and end tag

## Relationship to upstream

```text
ggml-org/llama.cpp (main)
  └── charlie12345/ROCmFPX          (ROCmFP4 preset, HIP kernels, GGUF types)
        └── pugant fork (GitHub, since removed): branch drafter-routing
              = charlie main + upstream merges PR #67–#82 + our work
              (routing, DFlash2 port, reasoning budget, cache salvage)
                    │  snapshots of its states (34a127168 → 62416acd3 → bc85fcb1d → f629365da)
                    ▼
  pugant/strix-halo-llamacpp-lab  ← THIS REPO — the full fork source
                                    included in rocmfpx/ (buildable),
                                    plus the same work as git am-able
                                    patches, docs and replication scripts
```

What is merged where:

| Work | Where it lives |
|---|---|
| Spec-boundary cache trailing rollback (+ reasoning-budget resend alignment) | Merged in [charlie12345/ROCmFPX#69](https://github.com/charlie12345/ROCmFPX/pull/69) |
| Reasoning-budget forced-newline, prepared for ggml-org | Upstream PR **archived, not sent** — story in [`patches/upstream-llamacpp/README.md`](patches/upstream-llamacpp/README.md) |
| Everything else (drafter routing, DFlash2 port, verify-log, remaining cache patches) | **Included in this repo**: full source in [`rocmfpx/`](rocmfpx/), plus the `git am`-able series in `patches/` |

> History note: our pull requests to the fork (incl. #69, merged; #80, open at
> the time) went through the now-removed GitHub fork; the patches and this
> snapshot preserve everything.
