# HF publish report — grug-35b-v2 + Ornith-1.0-35B ROCmFP4-STRIX_LEAN

**Date**: 2026-08-12
**Spec**: `docs/design/2026-08-11-hf-publish-grug-ornith-design.md` (APPROVED iter 2)
**Plan**: internal implementation plan `2026-08-11-hf-publish-grug-ornith-impl.md` (APPROVED iter 3, not included in this repo)

## Publication

Strategy used: **private → verify → public** (anti-surprises).

| Repo | URL | License |
|---|---|---|
| grug-35b-v2-ROCmFP4-STRIX_LEAN | <https://huggingface.co/pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN> | apache-2.0 |
| Ornith-1.0-35B-ROCmFP4-STRIX_LEAN | <https://huggingface.co/pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN> | mit |

Both repos created as `private=True`, verified post-upload (file presence/size, card_data YAML parsing, model tree), then flipped to `private=False` with `update_repo_settings(private=False)` after user OK.

## GGUF metadata sanitization

Script: `scripts/sanitize-gguf-v2.py` (not included in this repo; based on `gguf-new_metadata.py` from the ROCmFPX fork, uses the low-level API `add_tensor_info` + `write_tensor_data`, bypassing the broken `add_tensor` for the custom dtype Q4_0_ROCMFP4*). Root cause documented in internal notes (RESOLVED section).

### Changes applied to both GGUFs

| Field | grug (before → after) | Ornith (before → after) |
|---|---|---|
| `general.name` | `35B BF16` → `grug-35b-v2` | `Ornith-1.0-9B` ⚠️ → `Ornith-1.0-35B` |
| `general.basename` | (none) → `grug` | `Ornith-1.0-9B` ⚠️ → `Ornith-1.0-35B` |
| `general.finetune` | `35B` → `grug-v2` | (none) → `1.0` |
| `general.size_label` | `256x2.6B` → `35B` | `35B` (unchanged) |
| `general.organization` | (none) → `ProCreations` | (none) → `DeepReinforce` |
| `general.repo_url` | (none) → `hf.co/ProCreations/grug-35b-v2` | `hf.co/unsloth` → `hf.co/ornith-ai/Ornith-1.0-35B` |
| `general.license` | (none) → `apache-2.0` | `mit` (unchanged) |
| `general.quantized_by` | (none) → `pugant` | `Unsloth` → `pugant` |
| `general.description` | (none) → tested on ROCmFP4-STRIX_LEAN | (none) → tested |
| `general.tags` | (none) → `[gfx1151, rocmfpx, strix-halo, rocm, amdgpu]` | `[unsloth, text-generation]` → `[gfx1151, rocmfpx, strix-halo, rocm, amdgpu]` |
| `general.base_model.0.name` | (none) → `Ornith 1.0 35B` | `Ornith 1.0 35B` (unchanged) |
| `general.base_model.0.repo_url` | (none) → `hf.co/ornith-ai/Ornith-1.0-35B` | `hf.co/deepreinforce-ai/...` (unchanged) |
| `quantize.imatrix.file` | `/llmodels/GRUG/imatrix-grug-35b-v2.gguf` ⚠️ → `imatrix-grug-35b-v2.gguf` (neutral basename) | `/llmodels/ORNITH/imatrix_unsloth.gguf_file` ⚠️ → `imatrix.dat` |
| `quantize.imatrix.dataset` | `/calibration/grug-calibration.txt` ⚠️ → `ProCreations/grug-think-v3-10k` | `unsloth_calibration_Ornith-1.0-35B.txt` → `unsloth-calibration` |

⚠️ = sanitized issue (embedded local path, or wrong name inherited from unsloth).

## Published files

### grug repo
| File | Size | SHA256 |
|---|---:|---|
| `grug-35b-v2-ROCmFP4-STRIX_LEAN.gguf` | 18,597,337,568 B (17.32 GiB) | `5356e713137fdc15511fd4832a012f0695fed143400275313a3c6f7080d3a9dc` |
| `mmproj-grug-35b-v2-f16.gguf` | 899,284,192 B | (existing, unchanged) |
| `imatrix-grug-35b-v2.gguf` | 192,223,904 B | (existing, unchanged) |
| `LICENSE` | 11,358 B | Apache-2.0 full text |
| `NOTICE` | 760 B | attribution chain |
| `README.md` | 7,804 B | model card EN |
| `.gitattributes` | 127 B | LFS rules |

### Ornith repo
| File | Size | SHA256 |
|---|---:|---|
| `Ornith-1.0-35B-ROCmFP4-STRIX_LEAN.gguf` | 18,597,337,792 B (17.32 GiB) | `98fbae1e284223b0fe62ed1913fbcb0e20dca0541b518c642db62e3880a56ccd` |
| `mmproj-F16.gguf` | 899,283,680 B | (existing, unchanged) |
| `imatrix.dat` | 192,223,904 B | (ex `imatrix_unsloth.gguf_file`, redistributed with attribution) |
| `LICENSE` | 1,152 B | MIT full text |
| `NOTICE` | 755 B | attribution chain + unsloth credit |
| `README.md` | 7,759 B | model card EN |
| `.gitattributes` | 175 B | LFS rules (entry `imatrix.dat` added automatically by HF) |

## Benchmarks (reference)

See `results-2026-08-11-grug.md` for full methodology. Strix Halo AMD Ryzen AI Max+ 395, 128 GB LPDDR5X, profile `balanced` (not forced to performance).

| Model | Quant | Size | tg128 (tok/s) | pp512 (tok/s) |
|---|---|---:|---:|---:|
| grug-35b-v2 | ROCmFP4-STRIX_LEAN | 17.31 GiB | 70.92 | 1418 |
| grug-35b-v2 | Q4_K_M (baseline) | 19.70 GiB | 61.18 | — |
| Ornith-1.0-35B | ROCmFP4-STRIX_LEAN | 17.32 GiB | 66.68 | 1486 |
| Qwen3.6-35B-A3B (production ref) | ROCmFP4-STRIX_LEAN | 17.31 GiB | 63 | — |

Post-sanitization smoke test (2026-08-12, on the sanitized GGUF):
- grug: pp128 452 t/s, tg16 30 t/s (ROCm gfx1151, build `00d5452`)
- Ornith: pp128 407 t/s, tg16 26 t/s

## Post-upload verifications

- `private=False` confirmed for both ✓
- HTTP 200 on both public URLs ✓
- HTTP 200 CDN redirect (`us.aws.cdn.hf.co`) for public GGUF downloads ✓
- `card_data.license` populated (`apache-2.0` / `mit`) ✓
- `card_data.base_model` populated (`ProCreations/grug-35b-v2` / `ornith-ai/Ornith-1.0-35B`) ✓
- `card_data.tags` populated (8 tags: rocmfpx, gfx1151, strix-halo, qwen35moe, moe, rocm, amdgpu, ROCmFP4) ✓
- `card_data.library_name=llama.cpp`, `card_data.pipeline_tag=image-text-to-text` ✓
- 7 files per repo, all sizes match ✓
- Key numbers present in the READMEs (grep `70.92`, `66.68`, `1418`, `1486`) ✓

## HF token used

- User: `pugant` (the author), free account (isPro=False), 500 GB total storage / 50 GB per-file LFS limits
- Token `hf_work` (role=write), replaces the previous read-only `bonsai_ternary_img`
- `create_repo` requires role:write (with read-only: 403 Forbidden)

## Cost / time

- Total upload: ~35 GB in ~37 min (average 16 MB/s to the HF CDN)
- Sanitization: 15-25 sec per GGUF (NVMe cached reads)
- Total publication time (spec→plan→upload→verify): ~1 session
