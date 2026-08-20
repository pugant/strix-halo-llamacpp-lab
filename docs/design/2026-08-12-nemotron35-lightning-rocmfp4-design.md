# Spec: Nemotron 3.5 Lightning 30B-A3B → ROCmFP4-STRIX_LEAN + HF publishing

**Date**: 2026-08-12
**Author**: user pugant
**Status**: DRAFT → to be validated by adversarial review (100% gate)
**Prerequisites**: feasibility verified 2026-08-12. grug/Ornith pipeline completed as a reference (see `2026-08-11-grug-35b-v2-strix-lean-design.md`).

## 1. Goal and scope

Quantize **NVIDIA-Nemotron-3.5-Lightning-30B-A3B** (Mamba-2 + MoE + Attention hybrid MoE, arch `nemotron_h_moe`, 30B total / 3B active) into the **`Q4_0_ROCMFP4_STRIX_LEAN`** preset (type 106) via the ROCmFPX fork on Strix Halo, validate runtime operation (first Mamba-hybrid model on this toolchain), and publish the GGUF on Hugging Face as a public repo under `pugant/`.

### In scope
- Rebuild of the `docker-llm-service` binary from current main (prerequisite: arch `nemotron_h_moe` not in the `00d5452` binary)
- Load + generate smoke test on gfx1151 (real Mamba2-on-HIP validation)
- imatrix generation (or reuse if compatible)
- ROCmFP4-STRIX_LEAN quantization
- GGUF metadata sanitization (as with grug/Ornith)
- tg128/pp512 bench (plain + MTP if supported)
- HF publishing: 1 public repo `pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN`

### Out of scope
- Publishing the pipeline scripts (as with grug/Ornith: scope = "Core, ready model")
- Quality eval (perplexity/MMLU) — declared as a limitation
- Modifying weights beyond quantization
- Switching the production llm-service (separate, user decision post-publication)
- mmproj: the model is NOT vision (text-only) → no mmproj needed (difference vs grug/Ornith)

## 2. Feasibility basis (verified 2026-08-12, not speculative)

All technical dimensions verified with evidence (3 subagents + binary grep + config check):

| Dimension | Outcome | Evidence |
|---|:---:|---|
| Strix Halo fit (memory/bandwidth) | ✅ | 30B/3B-active = same class as grug/Ornith. Ceiling ~180 tok/s. Disk 257 GB free |
| Arch enum in the fork | ✅ | `LLM_ARCH_NEMOTRON_H_MOE` in `llama-arch.h/.cpp` + gguf-py (main HEAD) |
| Converter | ✅ | `NemotronHModel` auto-detects MoE. N/A: we use the direct BF16 GGUF |
| Compute graph | ✅ | `nemotron-h.cpp`: `build_mamba2_layer` + attention + `build_moe_ffn` + shared expert |
| Mamba2/SSM on HIP | ✅ | `ssm-conv.cu`/`ssm-scan.cu` kernels via `hip.h`. NOT CUDA-only |
| ROCmFP4 K/V protection | ✅ | Attention uses standard tensor names `blk.N.attn_k/attn_v` → automatic protection |
| Fork regression tests | ✅ | `Nemotron-Nano-3-30B-A3B` in test suite, `case 52: LLM_TYPE_31B_A3_5B` |
| Dim bug #20570 | ✅(arith) | `n_group=1`, `hidden_size=2688`, `expand=2→d_inner=5376` (verified from `config.json`) → `5376 % (1×2688)=0` passes. Assert expression to be confirmed at smoke test |
| Production binary `00d5452` | ❌ | rodata grep: it has `qwen35moe` but ZERO `nemotron_h_moe`/`mamba2` → **rebuild required** |

## 3. Artifact sources

### 3.1 BF16 GGUF (quantization source)
- **Repo**: `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` (official ggml mirror)
- **File**: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf` (65.85 GB, single file)
- **Pattern**: like Ornith (direct BF16 GGUF, skip HF→GGUF conversion). Different from grug (which converted from safetensors).
- **Download**: `wget -c --header="Authorization: Bearer $TOKEN"` into `~/llmodels/models/NEMOTRON/` (already started 2026-08-12, ~13-14 MB/s, ETA ~80 min)
- **unsloth NOTE**: the `unsloth/...-GGUF` repo does NOT contain BF16 (verified via API: main branch only, only compressed quants). Authoritative source = ggml-org.

### 3.2 imatrix
- **Option A (preferred, fast)**: reuse `bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-imatrix.gguf` (55.3 MB, size verified via API). To verify: tensor-count compatibility vs the ggml-org BF16 (if mismatch → Option B).
- **Option B (fallback, like grug)**: generate our own imatrix via `llama-imatrix` (CPU, 16 threads, ~256 chunks). Calibration: general dataset (wikitext/fineweb sample) or the Nemotron domain (reasoning/coding).
- **Decision**: try A, fall back to B.

### 3.3 Tokenizer / chat template
- Included in the BF16 GGUF (standard ggml-org conversion). chat_template.jinja present in the nvidia base repo. Post-download check: `llama-server` loads the template without errors.

## 4. Legal verification (license)

### Model license
- **OpenMDW-1.1** (Linux Foundation, permissive for ML distributions). NVIDIA adopts it for the Nemotron family. Source: [openmdw.ai/license/1-1/](https://openmdw.ai/license/1-1/), [LF press release](https://www.linuxfoundation.org/press/linux-foundation-releases-openmdw-1-1-nvidia-adopts-openmdw-for-cosmos-isaac-gr00t-ising-and-nemotron-ai-model-families).
- **Derivative redistribution (quantized GGUF)**: **allowed** with obligations: (1) include a copy of the OpenMDW-1.1 agreement, (2) keep all copyright/origin notices.
- → HF publication OK. `LICENSE` file (OpenMDW-1.1 full text) + `NOTICE` (attribution NVIDIA + ggml-org + ROCmFPX + bartowski for the imatrix).

### Derivation chain
```
nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16  (OpenMDW-1.1)  [base]
    └── ggml-org/...-GGUF (BF16 GGUF conversion)  [our source]
            └── pugant/...-ROCmFP4-STRIX_LEAN  (OpenMDW-1.1 derivative)  [output]
```

### Third-party components
| Component | License | GGUF redistribution? |
|---|---|---|
| nvidia base model | OpenMDW-1.1 | Yes, with license + notices |
| ggml-org BF16 GGUF | OpenMDW-1.1 (derivative of nvidia) | Yes, attribution |
| charlie12345/ROCmFPX (quant fork) | MIT | Yes (MIT code; imposes no terms on the output GGUF) |
| bartowski imatrix (if used) | OpenMDW-1.1/mit (to verify) | Yes, NOTICE attribution |
| kyuz0 toolbox (container) | No LICENSE (404 verified for grug) | N/A (attribution as a link only) |

## 5. `docker-llm-service` binary rebuild (PREREQUISITE)

### Why
The production binary (commit `00d5452`, built Aug 9) does NOT have `LLM_ARCH_NEMOTRON_H_MOE`. rodata grep: zero matches for `nemotron_h`/`mamba2`. Current main is needed.

### How — direct `docker build`, NOT compose
- **Dockerfile**: `<your-workspace>/workspace/docker/amd-strix-halo-toolboxes/toolboxes/Dockerfile.rocm-7.2.4-rocmfpx` (verified to exist, 3676 B).
- The Dockerfile clones `charlie12345/ROCmFPX@main` (`ARG BRANCH=main`, line 38). **Rebuilding the image = automatically pulls main HEAD** with nemotron_h_moe. No Dockerfile edit.
- Build context = the `toolboxes/` dir (contains the source COPYs `llama-grammar.patch` 377 B + `gguf-vram-estimator.py` 7885 B, both verified present).
- Target: gfx1151, ROCm 7.2.4, `GGML_HIP_FORCE_MMQ=ON`.

**Build command (separate tag, bypasses compose)**:
```bash
docker build --network host -t docker-llm-service:nemotron \
  -f <your-workspace>/workspace/docker/amd-strix-halo-toolboxes/toolboxes/Dockerfile.rocm-7.2.4-rocmfpx \
  <your-workspace>/workspace/docker/amd-strix-halo-toolboxes/toolboxes
```
Estimated time: ~20-40 min (HIP gfx1151 compilation with nproc=32, additional SSM kernels).

### Production coordination — SAFETY (verified)
- The `docker-llm-service:latest` container **serves in production** (managed by the compose service `llm-service` with `restart: unless-stopped`).
- **⚠️ Do NOT use `docker compose build llm-service` / `docker compose up -d` for this work**: the compose service has `build:` **without `image:`** (verified in the local deployment) → compose build **would overwrite `:latest`** and `up -d` **would recreate the production container**, interrupting the service — adjust your local llama.cpp service deployment accordingly.
- **Strategy**: build only the `docker-llm-service:nemotron` tag (command above). Smoke tests and quantization run in `docker run docker-llm-service:nemotron`. `:latest` and the production container **are not touched**. Promotion to `:latest` = separate user decision, out of scope.
- (Note: any local "kill+start" restart rules refer to a secondary dflash server on `<llm-service-port>`, not to the compose-managed llm-service — adjust your local llama.cpp service deployment.)

### Post-rebuild verification
- grep the new binary's rodata for `nemotron_h_moe` AND `mamba2` (both must now match).
- `llama-server --version`: commit more recent than `00d5452`.

## 6. Smoke test (Mamba2-on-HIP runtime validation)

After the rebuild, **before** quantizing, validate that the model runs (all in the `docker-llm-service:nemotron` container — **never `:latest`/production**, which does not recognize the arch). Cheap test for de-risking.

### ⚠️ Memory prerequisite (verified 2026-08-12)
Strix Halo is an APU: `rocm-smi` reports only **VRAM partition = 512 MB** (no significant dedicated VRAM). The real memory is the unified LPDDR5X (124 GB), accessible to ggml only with **`-e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`** (this is what production sets, compose line 364). **Without this env var, no model loads** on this GPU. → Mandatory in EVERY `docker run` doing GPU inference (smoke §6, bench §10). Quantization (§8) is CPU-only → not needed.

### Procedure
1. **Arch + metadata verification** of the BF16 GGUF: `docker run --rm -v ~/llmodels/models:/llmodels docker-llm-service:nemotron llama-gguf /llmodels/NEMOTRON/<BF16>.gguf r` → `general.architecture = nemotron_h_moe`, 52 layers, mixed layer types (mamba/moe/attention).
2. **Load + generate smoke** — two options:

   **Option 1 (preferred, validates the real quant source)**: briefly stop production to isolate the GPU, load the BF16 (65 GB in unified UMA, fits comfortably):
   ```bash
   docker stop llm-service                                          # ~10s, brief interruption
   docker run --rm -d --name nemotron-smoke \
     --device /dev/kfd --device /dev/dri --group-add video --group-add render \
     -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
     -p 1235:1234 -v ~/llmodels/models:/llmodels \
     docker-llm-service:nemotron llama-server \
       -m /llmodels/NEMOTRON/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf \
       -ngl 999 -c 4096 --host 0.0.0.0 --port 1234
   # poll health until "ready", then curl /v1/chat/completions, then cleanup:
   docker stop nemotron-smoke && docker start llm-service           # restores production
   ```
   **Option 2 (no production interruption)**: download a light quant (e.g. unsloth `UD-IQ2_M` 19 GB, same arch + Mamba path), smoke-test alongside production (19 + ~20 GB prod = 39 GB ≪ 124 GB UMA). Cheaper in RAM, costly in download time (~25 min). Validates arch+Mamba but NOT the BF16 source.

3. **Checks** (both options):
   - Poll `curl localhost:1235/health` until `{"status":"ok"}` (model loaded)
   - curl `/v1/chat/completions` with a simple prompt → coherent text
   - Absence of the `d_inner % (n_group*n_embd)` assert #20570 in the logs (Mamba path OK on HIP)
4. **MANDATORY cleanup**: `docker stop nemotron-smoke` (the container has `--rm` → auto-removed on stop) + restart production if Option 1.
5. **If it fails**: diagnose (bug #20570, SSM kernel on gfx1151, arch not recognized). Stop and re-evaluate BEFORE quantizing.

> Proceed to quantization only if the smoke test passes. Saves hours if there are unexpected runtime blocks on the Mamba kernels.

## 7. MTP / nextn verification

- Model config: `num_nextn_predict_layers: 1`, `mtp_layers_block_type: ["attention", "moe"]`.
- **Post-download check**: does the ggml-org BF16 contain the nextn/MTP block? For Nemotron-H the GGUF naming of MTP layers must be verified (different from Qwen `blk.64`). Look for `nextn.*` tensors or an extra layer block beyond the 52 declared.
- **⚠️ Observed split-file convention**: bartowski publishes a **separate** file `mtp-NVIDIA-Nemotron-…-Q4_0.gguf` (~1.15 GB) as the drafter, alongside the main GGUF. This suggests that Nemotron-H MTP follows the **split-file / separate draft-model convention** (loaded via `--spec-model` or `-md`), NOT the embedded MTP `blk.N` + `--spec-type draft-mtp` used by Qwen3.6. So: even if the nextn block is present, `--spec-type draft-mtp` may not apply.
- **If present + embedded convention**: test `--spec-type draft-mtp`. Measure acceptance + tok/s.
- **If split-file convention**: download/extract the MTP drafter and test speculative decoding with the draft model.
- **If absent or not working**: plain inference (still valid; the model is already fast as a 3B-active MoE). MTP = bonus declared as work-in-progress.

## 8. ROCmFP4-STRIX_LEAN quantization

```bash
docker run --rm -v ~/llmodels/models:/llmodels \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service:nemotron llama-quantize \
    --imatrix /llmodels/NEMOTRON/<imatrix> \
    /llmodels/NEMOTRON/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf \
    /llmodels/NEMOTRON/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
    Q4_0_ROCMFP4_STRIX_LEAN 16
```
- `nthreads=16` (optimal on Ryzen AI Max+ 395).
- `--dry-run` first to estimate the output size.
- **Output filename**: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf` (consistent with the HF repo name and the §11 upload).
- **Expected size**: ~30B × 4.38 bpw / 8 ≈ **16.4 GB** (range 16-17 GB, comparable to grug/Ornith 17.32 GiB).
- **Expected time**: a few minutes (27B dense = 2:43; 30B MoE similar, quantization is CPU-bound on the **30B total parameters** — all the experts + mamba + attention, not the 3B active).

### Post-quant validation
- `llama-server` loads the quant without errors.
- `llama-bench -ngl 999 -fa on -p 512 -n 128 -mmap 0` → size confirmed, tg128/pp512.

## 9. GGUF metadata sanitization (pre-upload, MANDATORY)

As with grug/Ornith (see `reference-gguf-sanitization-bugs` for the SOLUTION low-level API `add_tensor_info` + `write_tensor_data` — NOT `add_tensor` which is broken for custom dtypes).

**Operations** (Python script with gguf-py in the `docker-llm-service-convert` container):
1. Read the existing KV metadata
2. Modify/add:
   - `general.name`: `'NVIDIA-Nemotron-3.5-Lightning-30B-A3B'`
   - `general.basename`: `'NVIDIA-Nemotron-3.5-Lightning-30B-A3B'`
   - `general.size_label`: `'30B-A3B'`
   - `general.organization`: `'NVIDIA'`
   - `general.repo_url`: `'https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16'`
   - `general.base_model.0.name`: `'NVIDIA Nemotron 3.5 Lightning 30B A3B'`
   - `general.base_model.0.repo_url`: (as above)
   - `general.license`: `'openmdw-1.1'`
   - `general.quantized_by`: `'pugant'`
   - `general.tags`: `[gfx1151, rocmfpx, strix-halo, rocm, amdgpu, nemotron, mamba2, moe]`
3. Replace local paths (`quantize.imatrix.file`/`.dataset`) with neutral basenames
4. Copy the tensors 1:1 unchanged, write `*.sanitized.gguf`
5. Verification: SHA256, `llama-bench` smoke (load + 1 token), KV dump confirms the new values

**Cost**: ~10 min (~16 GB copy at ~1 GB/s NVMe). Reuses `scripts/sanitize-gguf-v2.py` (not included in this repo; adapting the config).

## 10. Bench

- **Methodology**: `llama-bench -ngl 999 -fa on -p 512 -n 128 -mmap 0` (in the `docker-llm-service:nemotron` container, with `-e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — see §6 memory prerequisite) + (for MTP) `llama-server` + curl `/v1/chat/completions` reading `timings.predicted_per_second` (llama-bench does NOT exercise MTP — internal observation).
- **Declared system configuration** (as with grug/Ornith): Bosgame BeyondMax, Ubuntu 24.04.4, kernel 7.0.0-28, power profile `balanced` (not forced to performance), amd-pstate-epp.
- **Expected plain tg128 range**: ~50-70 tok/s (3B-active MoE class; grug 70.92, Ornith 66.68). Uncertainty: SSM kernel maturity on gfx1151. Mamba (fixed KV) should help at long context.
- **Comparative baseline**: Q4_K_M (downloaded from unsloth or ggml-org) same bench.
- **Output**: `docs/benchmarks/bench-nemotron-{rocmfp4,q4_k_m}.txt`, report `docs/benchmarks/results-2026-08-12-nemotron.md`.

## 11. HF publication

### Repo
- **Path**: `pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN`
- **Visibility**: public
- **License**: openmdw-1.1
- **Files**:
```
README.md                                              # EN model card
LICENSE                                                # OpenMDW-1.1 full text
NOTICE                                                 # attribution chain (NVIDIA + ggml-org + ROCmFPX + bartowski)
.gitattributes                                         # LFS config (*.gguf)
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf   # ~16.4 GB (main, LFS)
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-imatrix.gguf     # imatrix (LFS) — ours or bartowski (consistent name §3.2)
```
(NO mmproj — text-only model.)

### Model card (README.md, EN)
Structure modeled on grug/Ornith, sections:
1. HF YAML frontmatter: `library_name: llama.cpp`, **`license: other` + `license_name: openmdw-1.1` + `license_link: https://openmdw.ai/license/1-1/`** (the HF sidebar does not recognize `openmdw-1.1` as a bare `license:` value — the `other`+`license_name` pattern as in the NVIDIA card), `base_model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, `tags: [rocmfpx, gfx1151, strix-halo, nemotron, mamba2, moe, rocm, amdgpu]`, `pipeline_tag: text-generation`
2. TL;DR + ⚠️ Critical warnings (type 106 = invalid in stock llama.cpp; requires the ROCmFPX fork; FP4 software on RDNA 3.5; **to our knowledge, the first Mamba-hybrid model published in this ROCmFP4 format**)
3. Benchmarks (tg128/pp512 ROCmFP4 vs Q4_K_M, declared system)
4. Quantization details (STRIX_LEAN, 4.38 bpw, K/V protect + Q5_K emb)
5. **Architecture** (extra section vs grug/Ornith): explain nemotron_h_moe = Mamba-2 + MoE + sparse Attention hybrid, why it is interesting on Strix Halo (fixed KV → long-context)
6. imatrix methodology
7. Usage (Strix Halo commands, MTP status note)
8. Attribution & model tree
9. License (OpenMDW-1.1), Acknowledgements, Limitations, Disclaimer

### Upload
- `huggingface_hub` 1.22, write-mode token `~/.cache/huggingface/token` (user `pugant`).
- Reuses `scripts/upload-to-hf.py` (adapting repo_id + file list).
- Free account: 500 GB storage, 50 GB/file. Total ~17 GB → OK.
- Order: LICENSE/NOTICE/.gitattributes → GGUF + imatrix → README.md (live card last).

## 12. Success criteria

1. Binary rebuilt (`:nemotron` tag) with `nemotron_h_moe` + `mamba2` in rodata
2. BF16 smoke test: load + generate on gfx1151 without failed asserts
3. ROCmFP4-STRIX_LEAN GGUF produced (~16-17 GB, loads in llama-server)
4. Metadata sanitized (post-sanitization verification)
5. Bench documented (tg128/pp512, plain; + MTP if available)
6. Public HF repo created with all files, card rendered, license visible, model tree populated
7. No sensitive data published (re-check grep for local paths)

## 13. Risks and mitigations

| Risk | Probability | Mitigation |
|---|:---:|---|
| Immature gfx1151 SSM kernels → BF16 smoke test fails/slow | Medium | Smoke test BEFORE quantizing. If it fails, stop and diagnose. The model remains valid as a documented experiment |
| Dim bug #20570 triggers on our model at runtime | Low | `n_group=1` config → arithmetic passes. Confirmed at smoke test |
| MTP/nextn not supported at runtime on nemotron_h_moe | Medium | Plain inference retains value. MTP = bonus, declared as work-in-progress |
| Rebuild breaks production (:1234) | Low | New tag `:nemotron`, do not touch `:latest` |
| bartowski imatrix incompatible (tensor mismatch) | Low-Medium | Fallback: generate our own (Option B) |
| ggml-org BF16 does not include the MTP/nextn block | Low | Plain inference; or evaluate the repo `h1st0ry3D/...-MTP-GGUF` |
| Wrong OpenMDW license metadata in the frontmatter | Low | Pre-upload check of the HF pattern `license: other` + `license_name: openmdw-1.1` + `license_link` (NOT bare `license: openmdw-1.1`, the HF sidebar does not render it) |

## 14. Deferred decisions / out-of-scope

- **Promotion of the rebuild to production `:latest`**: separate, user decision post-validation
- **Quality eval (perplexity/MMLU)**: future work on community input
- **COHERENT vs STRIX_LEAN preset comparison** for tool-call: future
- **Publishing a "safe" stock-llama.cpp Q4_K_M quant** as a fallback for non-ROCmFPX users: future
