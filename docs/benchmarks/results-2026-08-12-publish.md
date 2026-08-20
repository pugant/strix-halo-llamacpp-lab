# HF publish report — grug-35b-v2 + Ornith-1.0-35B ROCmFP4-STRIX_LEAN

**Date**: 2026-08-12
**Spec**: `docs/design/2026-08-11-hf-publish-grug-ornith-design.md` (APPROVED iter 2)
**Plan**: piano di implementazione interno `2026-08-11-hf-publish-grug-ornith-impl.md` (APPROVED iter 3, non incluso nel repo)

## Pubblicazione

Strategia usata: **private → verify → public** (anti-surprises).

| Repo | URL | License |
|---|---|---|
| grug-35b-v2-ROCmFP4-STRIX_LEAN | <https://huggingface.co/pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN> | apache-2.0 |
| Ornith-1.0-35B-ROCmFP4-STRIX_LEAN | <https://huggingface.co/pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN> | mit |

Entrambi i repo creati come `private=True`, verificati post-upload (file presence/size, card_data YAML parsing, model tree), poi flippati a `private=False` con `update_repo_settings(private=False)` dopo OK utente.

## Sanificazione GGUF metadata

Script: `scripts/sanitize-gguf-v2.py` (non incluso nel repo; basato su `gguf-new_metadata.py` del fork ROCmFPX, usa API low-level `add_tensor_info` + `write_tensor_data`, bypassando `add_tensor` rotta per dtype custom Q4_0_ROCMFP4*). Root cause documentata nelle note interne (sezione RISOLTO).

### Modifiche applicate a entrambi i GGUF

| Campo | grug (prima → dopo) | Ornith (prima → dopo) |
|---|---|---|
| `general.name` | `35B BF16` → `grug-35b-v2` | `Ornith-1.0-9B` ⚠️ → `Ornith-1.0-35B` |
| `general.basename` | (nessuno) → `grug` | `Ornith-1.0-9B` ⚠️ → `Ornith-1.0-35B` |
| `general.finetune` | `35B` → `grug-v2` | (nessuno) → `1.0` |
| `general.size_label` | `256x2.6B` → `35B` | `35B` (invariato) |
| `general.organization` | (nessuno) → `ProCreations` | (nessuno) → `DeepReinforce` |
| `general.repo_url` | (nessuno) → `hf.co/ProCreations/grug-35b-v2` | `hf.co/unsloth` → `hf.co/ornith-ai/Ornith-1.0-35B` |
| `general.license` | (nessuno) → `apache-2.0` | `mit` (invariato) |
| `general.quantized_by` | (nessuno) → `pugant` | `Unsloth` → `pugant` |
| `general.description` | (nessuno) → testata ROCmFP4-STRIX_LEAN | (nessuno) → testata |
| `general.tags` | (nessuno) → `[gfx1151, rocmfpx, strix-halo, rocm, amdgpu]` | `[unsloth, text-generation]` → `[gfx1151, rocmfpx, strix-halo, rocm, amdgpu]` |
| `general.base_model.0.name` | (nessuno) → `Ornith 1.0 35B` | `Ornith 1.0 35B` (invariato) |
| `general.base_model.0.repo_url` | (nessuno) → `hf.co/ornith-ai/Ornith-1.0-35B` | `hf.co/deepreinforce-ai/...` (invariato) |
| `quantize.imatrix.file` | `/llmodels/GRUG/imatrix-grug-35b-v2.gguf` ⚠️ → `imatrix-grug-35b-v2.gguf` (basename neutro) | `/llmodels/ORNITH/imatrix_unsloth.gguf_file` ⚠️ → `imatrix.dat` |
| `quantize.imatrix.dataset` | `/calibration/grug-calibration.txt` ⚠️ → `ProCreations/grug-think-v3-10k` | `unsloth_calibration_Ornith-1.0-35B.txt` → `unsloth-calibration` |

⚠️ = problema sanificato (path locale embedded, oppure nome errato ereditato da unsloth).

## File pubblicati

### Repo grug
| File | Size | SHA256 |
|---|---:|---|
| `grug-35b-v2-ROCmFP4-STRIX_LEAN.gguf` | 18,597,337,568 B (17.32 GiB) | `5356e713137fdc15511fd4832a012f0695fed143400275313a3c6f7080d3a9dc` |
| `mmproj-grug-35b-v2-f16.gguf` | 899,284,192 B | (esistente, immutato) |
| `imatrix-grug-35b-v2.gguf` | 192,223,904 B | (esistente, immutato) |
| `LICENSE` | 11,358 B | Apache-2.0 full text |
| `NOTICE` | 760 B | attribution chain |
| `README.md` | 7,804 B | model card EN |
| `.gitattributes` | 127 B | LFS rules |

### Repo Ornith
| File | Size | SHA256 |
|---|---:|---|
| `Ornith-1.0-35B-ROCmFP4-STRIX_LEAN.gguf` | 18,597,337,792 B (17.32 GiB) | `98fbae1e284223b0fe62ed1913fbcb0e20dca0541b518c642db62e3880a56ccd` |
| `mmproj-F16.gguf` | 899,283,680 B | (esistente, immutato) |
| `imatrix.dat` | 192,223,904 B | (ex `imatrix_unsloth.gguf_file`, ridistribuito con attribution) |
| `LICENSE` | 1,152 B | MIT full text |
| `NOTICE` | 755 B | attribution chain + unsloth credit |
| `README.md` | 7,759 B | model card EN |
| `.gitattributes` | 175 B | LFS rules (entry `imatrix.dat` aggiunta da HF automaticamente) |

## Benchmarks (riferimento)

Vedi `results-2026-08-11-grug.md` per metodologia completa. Strix Halo AMD Ryzen AI Max+ 395, 128 GB LPDDR5X, profile `balanced` (non forzato a performance).

| Modello | Quant | Size | tg128 (tok/s) | pp512 (tok/s) |
|---|---|---:|---:|---:|
| grug-35b-v2 | ROCmFP4-STRIX_LEAN | 17.31 GiB | 70.92 | 1418 |
| grug-35b-v2 | Q4_K_M (baseline) | 19.70 GiB | 61.18 | — |
| Ornith-1.0-35B | ROCmFP4-STRIX_LEAN | 17.32 GiB | 66.68 | 1486 |
| Qwen3.6-35B-A3B (production ref) | ROCmFP4-STRIX_LEAN | 17.31 GiB | 63 | — |

Smoke test post-sanificazione (2026-08-12, su GGUF sanitizzato):
- grug: pp128 452 t/s, tg16 30 t/s (ROCm gfx1151, build `00d5452`)
- Ornith: pp128 407 t/s, tg16 26 t/s

## Verifiche post-upload

- `private=False` confermato per entrambi ✓
- HTTP 200 su entrambi gli URL pubblici ✓
- HTTP 200 redirect CDN (`us.aws.cdn.hf.co`) per download GGUF pubblici ✓
- `card_data.license` popolato (`apache-2.0` / `mit`) ✓
- `card_data.base_model` popolato (`ProCreations/grug-35b-v2` / `ornith-ai/Ornith-1.0-35B`) ✓
- `card_data.tags` popolato (8 tag: rocmfpx, gfx1151, strix-halo, qwen35moe, moe, rocm, amdgpu, ROCmFP4) ✓
- `card_data.library_name=llama.cpp`, `card_data.pipeline_tag=image-text-to-text` ✓
- 7 file per repo, size tutte match ✓
- Numeri chiave presenti nei README (grep `70.92`, `66.68`, `1418`, `1486`) ✓

## Token HF usato

- User: `pugant` (l'autore), account free (isPro=False), 500 GB total storage / 50 GB per-file LFS limits
- Token `hf_work` (role=write), sostituisce il read-only `bonsai_ternary_img` precedente
- `create_repo` richiede role:write (con read-only: 403 Forbidden)

## Cost / tempo

- Upload totale: ~35 GB in ~37 min (media 16 MB/s verso HF CDN)
- Sanificazione: 15-25 sec per GGUF (NVMe cached reads)
- Tempo totale pubblicazione (spec→plan→upload→verify): ~1 sessione
