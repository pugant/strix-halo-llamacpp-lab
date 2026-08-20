# Spec: HF publication of grug + Ornith ROCmFP4-STRIX_LEAN

**Date**: 2026-08-11
**Author**: user pugant
**Status**: DRAFT → to be validated by adversarial review
**Prerequisites**: grug PHASE 1 + Ornith PHASE 2 completed (T4-T9 ✓). See `docs/design/2026-08-11-grug-35b-v2-strix-lean-design.md`.

## 1. Goal and scope

Publish two public Hugging Face repositories with the `Q4_0_ROCMFP4_STRIX_LEAN` quantized GGUFs of **grug-35b-v2** and **Ornith-1.0-35B**, produced by the ROCmFPX pipeline on Strix Halo, for sharing with the community (in particular other Strix Halo / gfx1151 users).

### In scope
- 2 public HF repos under `pugant/`
- Quantized GGUF + mmproj + imatrix (grug) for each model
- EN model card with benchmarks, methodology, usage, attribution
- LICENSE + NOTICE files for each repo
- Upload via `huggingface_hub` (token in `~/.cache/huggingface/token`, user `pugant`)

### Out of scope
- Publishing the pipeline scripts (user choice: "Core, ready model")
- Publishing the full calibration text (contains third-party OS code)
- Quality eval (perplexity/MMLU) — declared as a limitation, invitation for community feedback
- Modifications to the GGUF weights or metadata (we publish as generated)
- Restart of the production llm-service (separate)

## 2. Legal verification (licenses)

### Derivation chain (model tree)
```
Qwen3.5-VL-MoE (base arch)
    └── ornith-ai/Ornith-1.0-35B (alias `deepreinforce-ai/Ornith-1.0-35B`, MIT)  [base]
            └── ProCreations/grug-35b-v2 (Apache-2.0) [fine-tune]
```
Model tree verified via HF API: `ProCreations/grug-35b-v2` cardData.base_model declares `deepreinforce-ai/Ornith-1.0-35B` (legacy namespace). That repo 307-redirects → `ornith-ai/Ornith-1.0-35B` (canonical). In our repos' YAML frontmatter we will use `ornith-ai/Ornith-1.0-35B` (canonical) for alignment with the HF model tree.

### Component licenses
| Component | License | Verification source | Permits redistributing the derivative GGUF? |
|---|---|---|---|
| `ProCreations/grug-35b-v2` (grug base model) | **Apache-2.0** | HF metadata `license: apache-2.0` | Yes, with attribution + license inclusion + NOTICE |
| `ornith-ai/Ornith-1.0-35B` (Ornith base model) | **MIT** | HF metadata `license: mit` + unsloth card | Yes, with attribution + license inclusion |
| `charlie12345/ROCmFPX` (quantization fork) | **MIT** | GitHub repo README | Yes (MIT on the code; imposes no terms on the output GGUFs) |
| `kyuz0/amd-strix-halo-toolboxes` (container) | **No LICENSE** (GitHub API 404 verified 2026-08-11) | GitHub repo | N/A (we do not redistribute the container; we only use attribution as a reference link) |
| `ProCreations/grug-think-v3-10k` (calibration dataset) | Apache-2.0, **public, not gated** | HF API verified | We do not redistribute the dataset, only reference + attribution. The community can download it freely. |
| Qwen3.5-VL-MoE (base arch) | not directly in grug's HF model tree (declared base = Ornith) | — | Effective license = the one in the HF model tree |

### Attribution obligations
- **Apache-2.0** (grug): include the license text + NOTICE file + "derivative work" statement. Apache-2.0 section 4 requires keeping all attribution/NOTICE.
- **MIT** (Ornith): include the license text + copyright notice.
- Both repos will have `LICENSE` (full text) + `NOTICE` (attribution chain).

### Legal caveats (to state in the model card)
- "No affiliation with AMD, Qwen, ProCreations, DeepReinforce, unsloth, kyuz0, or charlie12345."
- "Provided as-is, without warranty."
- "Users must comply with the base model license (Apache-2.0 / MIT)."

## 3. Security verification (sensitive data)

### Checks performed empirically (2026-08-11, adversarial review)

| Check | Result |
|---|---|
| HF tokens in published scripts | N/A (scope = "Core, ready model" → no scripts published) |
| **Local paths embedded as GGUF metadata** | **CONFIRMED via `grep -aoE` on the whole file** (NOT via `strings` which has a minimum length threshold and missed them):<br>• grug: `quantize.imatrix.file = '/llmodels/GRUG/imatrix-grug-35b-v2.gguf'`, `quantize.imatrix.dataset = '/calibration/grug-calibration.txt'`<br>• Ornith: `quantize.imatrix.file = '/llmodels/ORNITH/imatrix_unsloth.gguf_file'`, `quantize.imatrix.dataset = 'unsloth_calibration_Ornith-1.0-35B.txt'`<br><br>No paths in the user home or HF tokens. The `/llmodels/` paths are not credentials but reveal internal conventions → **sanitization required** (see §3.1). |
| **Wrong or generic GGUF metadata** | **CONFIRMED via Python GGUF parser**:<br>• grug: `general.name = '35B BF16'` (generic, no "grug" mention)<br>• Ornith: `general.name = 'Ornith-1.0-9B'` (**WRONG: says 9B instead of 35B**, error inherited from unsloth)<br><br>→ **sanitization required** (see §3.1). |
| Calibration text (not published) | 140 "sensitive" keyword matches = **all false positives**: OS Python code (paramiko, marshmallow) with `password`/`api_key` attribute names + public OS authors' emails in copyright notices. No real sensitive data. |
| HF tokens in internal scripts | Present in the scripts (`scripts/*.sh`) but NOT published (scope = Core, no scripts). |

### 3.1 GGUF metadata sanitization (pre-upload, **MANDATORY**)

Before the HF upload, both GGUFs are sanitized via a Python script with the `gguf` library (path reuses the `/usr/local/bin/convert_hf_to_gguf.py` deps installed in the `docker-llm-service-convert` container).

**Operations** (for both GGUFs):
1. Read all the existing KV metadata
2. Modify / add:
   - `general.name`: `'grug-35b-v2'` (grug) / `'Ornith-1.0-35B'` (Ornith) — fix C1/C2
   - `general.basename`: `'grug'` / `'Ornith-1.0-35B'`
   - `general.size_label`: `'35B'`
   - `general.finetune`: `'grug-v2'` (grug) / `'1.0'` (Ornith)
   - `general.organization`: `'ProCreations'` / `'DeepReinforce'`
   - `general.repo_url`: `'https://huggingface.co/ProCreations/grug-35b-v2'` / `'https://huggingface.co/ornith-ai/Ornith-1.0-35B'`
   - `general.base_model.0.name`: `'Ornith 1.0 35B'` (both, Ornith as base)
   - `general.base_model.0.repo_url`: `'https://huggingface.co/ornith-ai/Ornith-1.0-35B'`
   - `general.license`: `'apache-2.0'` (grug) / `'mit'` (Ornith)
   - `general.quantized_by`: `'pugant'`
   - `general.tags`: add `[gfx1151, rocmfpx, strix-halo, rocm, amdgpu]`
3. Remove or replace with neutral placeholders:
   - `quantize.imatrix.file`: `'grug-calibration.gguf'` (grug) / `'unsloth-imatrix.gguf'` (Ornith) — basename only, no path — fix C3
   - `quantize.imatrix.dataset`: `'grug-think-v3-10k'` (grug) / `'unsloth-calibration'` (Ornith)
4. Copy the tensors 1:1 unchanged
5. Write the new GGUF file to a temporary path (e.g. `*.sanitized.gguf`)
6. Post-sanitization verification: recompute SHA256, `llama-bench` smoke (load + 1 token), `llama-gguf ... r` to dump the KVs and verify the new values
7. Replace the original file with the sanitized one (rename)

**Cost**: ~10 min per GGUF (17 GB copy with new metadata, on internal NVMe ~1 GB/s).
**SHA256**: changes post-sanitization → recompute for the report.

### Security conclusion (post-sanitization)
Post-sanitization, the published GGUFs will have:
- ✓ Correct and identifiable model name
- ✓ Complete attribution (organization, repo URL, base model)
- ✓ License embedded in the metadata
- ✓ No local paths embedded (only neutral basenames)
- ✓ Tags for HF discoverability

## 4. Repo structure

### 4.1 grug repo
**Path**: `pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN`
**Visibility**: public
**Files**:
```
README.md                                  # EN model card
LICENSE                                    # Apache-2.0 full text
NOTICE                                     # attribution chain + ROCmFPX commit
.gitattributes                             # LFS config (*.gguf, *.png)
grug-35b-v2-ROCmFP4-STRIX_LEAN.gguf        # 17.32 GiB (main, LFS)
imatrix-grug-35b-v2.gguf                   # 183 MB (ours, LFS)
mmproj-grug-35b-v2-f16.gguf                # 857 MB (vision projector, LFS)
```

### 4.2 Ornith repo
**Path**: `pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN`
**Visibility**: public
**Files**:
```
README.md                                  # EN model card
LICENSE                                    # MIT full text
NOTICE                                     # attribution chain (ornith-ai + unsloth + ROCmFPX)
.gitattributes                             # LFS config
Ornith-1.0-35B-ROCmFP4-STRIX_LEAN.gguf     # 17.32 GiB (main, LFS)
imatrix.dat                                # 183 MB (copy of the unsloth imatrix, LFS) — attribution in the NOTICE
mmproj-F16.gguf                            # 857 MB (LFS)
```

**Ornith imatrix note**: the imatrix derives from `unsloth/Ornith-1.0-35B-GGUF/imatrix_unsloth.gguf_file`. We redistribute it as `imatrix.dat` with explicit attribution to unsloth in the NOTICE (unsloth's MIT license permits this).

## 5. Model card (README.md, EN)

Section structure (inspired by unsloth, adapted Strix-specific):

1. **HF metadata YAML** (frontmatter)
   - `library_name: llama.cpp`
   - `license: apache-2.0` (grug) / `mit` (Ornith)
   - `base_model: ProCreations/grug-35b-v2` / `ornith-ai/Ornith-1.0-35B`
   - `tags: [rocmfpx, gfx1151, strix-halo, qwen35moe, moe, rocm, amdgpu, ROCmFP4]`
   - `pipeline_tag: image-text-to-text` (vision + text model)
   - `language: [en, multilingual]`
   - `version: 1.0` + `date: 2026-08-11`

2. **Version header**: `> Version 1.0 — 2026-08-11`

3. **TL;DR** (2 lines): model + quant + target hardware

4. **⚠️ Critical warnings** (top, visible callout):
   - Requires the **kyuz0 Strix Halo toolbox** (which builds `charlie12345/ROCmFPX`). Type 106 (Q4_0_ROCMFP4_STRIX_LEAN) = **INVALID in stock llama.cpp**.
   - Profiled for **gfx1151** (Strix Halo / RDNA 3.5). Not tested elsewhere.
   - FP4 is **software** on RDNA 3.5 (no FP4 silicon units) — the benefit is memory bandwidth, not compute.

5. **Benchmarks** — tg128/pp512 table (grug ROCmFP4: 70.92 vs grug Q4_K_M 61.18, +16%; Ornith ROCmFP4: 66.68; grug size 17.31 GiB vs Q4_K_M 19.70 GiB, -12%). On Strix Halo Ryzen AI Max+ 395, 128 GB LPDDR5X. Methodology: `llama-bench -ngl 999 -fa on -p 512 -n 128 -mmap 0`. Production reference: Qwen3.6-35B-A3B ROCmFP4-STRIX_LEAN 63 tok/s. Sources: `docs/benchmarks/bench-grug-{rocmfp4,q4_k_m}.txt`, `logs/bench-ornith-vs-grug.log`.

   **System configuration at bench time** (declared for reproducibility):
   - **Bare metal host**: Bosgame BeyondMax Series (`bosgame-m5`), Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic
   - **CPU power profile**: `balanced` (`powerprofilesctl get`) — default configuration, **NOT forced to `performance`**. Representative of an out-of-the-box setup.
   - **CPU scaling driver**: `amd-pstate-epp`, scaling_governor `performance` (amd-pstate-epp default), EPP `performance`
   - **IOMMU/iGPU power**: auto (no manual tuning)

   Note: the tok/s are measured on a NON-optimized system (balanced power profile). Users who set `powerprofilesctl set performance` may get slightly higher values.

6. **Quantization details** — preset `Q4_0_ROCMFP4_STRIX_LEAN` (type 106, 4.29 BPW). What it protects: attention K/V (`attn_qkv`/`attn_v` → q4_0_rocmfp4) + Q5_K token embeddings. Experts (`ffn_*_exps`) in `q4_0_rocmfp4_fast` (max speed). Fork reference: `charlie12345/ROCmFPX` commit `00d5452`.

7. **imatrix methodology**
   - **grug**: generated with `llama-imatrix` 256 chunks, 16 threads, CPU-only. Calibration from `ProCreations/grug-think-v3-10k` (**public Apache-2.0 dataset, not gated** — anyone can download it to replicate). Explicit thanks to the grug team. 510 entries out of 733 tensors. Warning `partial data 99.61%` = 1/256 experts not activated during calibration (normal for MoE, see the llama.cpp code `tools/imatrix/imatrix.cpp`).
   - **Ornith**: imatrix precomputed by unsloth (46 chunks). Included as `imatrix.dat`.

8. **Usage** — Strix Halo commands:
   ```bash
   # Requires the kyuz0/charlie12345 ROCmFPX container
   llama-server -m <model>.gguf --mmproj <mmproj>.gguf \
     -ngl 999 -fa on --jinja -c 32768 --host 0.0.0.0 --port 1234
   ```
   (Note: MTP not enabled for grug — absent from the model. For Ornith — `mtp_num_hidden_layers=1` present but not enabled at runtime for plain MoE alignment.)

9. **How to replicate** (text, no published scripts) — pipeline in words: build the ROCmFPX container → download BF16 → `convert_hf_to_gguf.py` (grug) / use the unsloth BF16 GGUF (Ornith) → `llama-imatrix` (grug, grug-think-v3-10k dataset) → `llama-quantize ... Q4_0_ROCMFP4_STRIX_LEAN 16`.

10. **Attribution & model tree** — base model link, chain (Qwen3.5-VL-MoE → Ornith → grug), ROCmFPX commit, kyuz0 toolbox tag, unsloth credit.

11. **License** — Apache-2.0 (grug) / MIT (Ornith). "Derivative work: original model and its license are preserved. See LICENSE and NOTICE."

12. **Acknowledgements** — dedicated section (see §5.1).

13. **Limitations & community feedback** — speed benchmark only, no quality eval. Explicit invitation: "We invite the community — especially fellow Strix Halo owners — to test and share quality results. Open a Discussion."

14. **Citation** — bibtex placeholder (grug: ProCreations; Ornith: DeepReinforce Team, URL `deep-reinforce.com/ornith_1_0.html`).

15. **Disclaimer** — no affiliation, as-is.

### 5.1 Acknowledgements section
```markdown
## Acknowledgements
Built on the shoulders of giants:
- [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) — Strix Halo container runtime
- [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) — llama.cpp fork with ROCmFP4 presets (type 106)
- [unsloth](https://huggingface.co/unsloth) — BF16 GGUF + imatrix for Ornith
- [ProCreations](https://huggingface.co/ProCreations) — grug-35b-v2 + grug-think-v3-10k calibration dataset
- [ornith-ai / DeepReinforce Team](https://huggingface.co/deepreinforce-ai) — Ornith-1.0-35B
- [llama.cpp](https://github.com/ggerganov/llama.cpp) community + Kawrakow (imatrix methodology)
- Hardware: Bosgame BeyondMax Series (Strix Halo bare metal host)
```

## 6. Upload mechanism

### Tool
`huggingface_hub` Python lib v1.22.0 (verified on host). Token: `~/.cache/huggingface/token`. User confirmed: `pugant`, no org.

### Preventive HF storage check (2026-08-11)
- `pugant` is a **free** account (`isPro=False, plan=None, canPay=False` via `hf whoami-v2`)
- HF free limits: 500 GB total storage, 50 GB per single LFS file ([docs](https://huggingface.co/docs/hub/en/storage-limits))
- Total publication: ~36 GB (grug ~18 GB + Ornith ~18 GB) → **OK within the threshold**
- Per file: 17 GB GGUF < 50 GB limit ✓

### Procedure (per repo)
1. `HfApi().create_repo(repo_id="pugant/<name>", repo_type="model", private=False, exist_ok=False)`
2. Upload `.gitattributes` + `LICENSE` + `NOTICE` (small files)
3. `upload_large_folder()` or `upload_file()` for the GGUF (17 GB), mmproj (857 MB), imatrix (183 MB) — automatic LFS, resume on failure
4. `upload_file()` for `README.md` (last, so the repo is "complete" when the card goes live)
5. sha256 check post-upload (optional, `huggingface_hub` does not do it natively; compare `hf_hub_url` size vs local)

### Upload order
1. LICENSE, NOTICE, .gitattributes (setup)
2. Large files (GGUF, mmproj, imatrix)
3. README.md (live card)

### Resume
`huggingface_hub` handles automatic resume for LFS. If an upload interrupts, re-running the same command resumes it.

## 7. Success criteria

1. Both repos created `public` under `pugant/`
2. All files present with the correct size (verification via HF API `repo_info`)
3. README.md renders correctly (HF preview)
4. License visible in the HF sidebar (`license` metadata in the frontmatter)
5. HF model tree populated (`base_model` link working)
6. No sensitive data published (post-upload re-check: grep for local paths in the README/NOTICE)
7. Repos findable via the tags `rocmfpx`, `gfx1151`, `strix-halo`

## 8. Declared limitations (in the model card)

- Speed benchmark only (no quality eval)
- Profiled for gfx1151 only (not tested on other GPUs)
- MTP not enabled (plain inference). Possible MTP activation = future work
- grug imatrix: 1/256 experts not covered (normal for MoE, negligible impact)

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Upload fails due to size | LFS resume; re-run |
| Wrong license metadata in the frontmatter | Pre-upload check: `license: apache-2.0` (grug), `license: mit` (Ornith) |
| Community confuses it with a stock-llama.cpp quant | Top critical warning + explicit tags |
| Apache-2.0 NOTICE attribution violation | Complete NOTICE file + derivative work statement |
| grug-think-v3-10k gated: a user cannot replicate the imatrix | **Not applicable**: public Apache-2.0 dataset, not gated (verified via HF API). The community can download it freely to replicate the imatrix. A positive point for replicability, stated in the card. |

## 10. Source artifacts (post-completion, reference)

- grug: `~/llmodels/models/GRUG/grug-35b-v2-ROCmFP4-STRIX_LEAN.gguf` (17.32 GiB), `imatrix-grug-35b-v2.gguf` (183 MB), `mmproj-grug-35b-v2-f16.gguf` (857 MB)
- Ornith: `~/llmodels/models/ORNITH/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN.gguf` (17.32 GiB), `imatrix_unsloth.gguf_file` (183 MB, → `imatrix.dat`), `mmproj-F16.gguf` (857 MB)
- Bench logs: `<lab-repo>/logs/bench-grug-t9.log`, `bench-ornith-vs-grug.log`

## 11. Open items / deferred decisions

- **kyuz0 toolbox license**: **verified 2026-08-11** — the `kyuz0/amd-strix-halo-toolboxes` repo has NO LICENSE file (HTTP 404 on the GitHub API for both `/license` and the raw `LICENSE`). The default "All rights reserved" copyright applies. **Decision**: in the model card we use only attribution as a GitHub reference link (fair use), without clauses implying redistribution of the container. If kyuz0 adds a LICENSE in the future, align the NOTICE.
- **quality eval**: future work, on community input
- **Production llm-service restart**: separate, user decision post-publication
