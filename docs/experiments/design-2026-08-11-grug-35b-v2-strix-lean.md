# Spec — grug-35b-v2 → ROCmFP4-STRIX_LEAN for Strix Halo

**Date:** 2026-08-11
**Status:** Draft (awaiting adversarial review)
**Target hardware:** AMD Ryzen AI MAX+ 395 (Strix Halo, gfx1151, 124 GB unified LPDDR5X-8000 ~270 GB/s)
**Author:** GLM by z.ai session with pugant

---

## 1. Context and motivation

### 1.1 Source model
`ProCreations/grug-35b-v2` is a LoRA fine-tune of `Ornith-1.0-35B`, in turn a fine-tune of **Qwen3.5-VL-MoE** (`model_type: qwen3_5_moe`).

Architecture (from `config.json`):
- MoE: **256 experts, 8 active + 1 shared → ~3B active params/token** (comparable to Qwen3.6-35B-A3B)
- 40 total layers, hybrid **30 linear_attention (Gated DeltaNet) + 10 full_attention** (pattern `[L,L,L,F] × 10`)
- hidden_size 2048, num_attention_heads 16, num_key_value_heads 2, head_dim 256
- vocab 248320, max_position_embeddings 262144, partial_rotary_factor 0.25, full_attention_interval 4
- Qwen3.5-VL vision tower (`qwen3_5_moe_vision`, depth 27, hidden 1152, patch_size 16)
- **MTP disabled**: `mtp_num_hidden_layers: 0`
- License Apache-2.0
- Source: BF16 safetensors, 18 shards, ~70 GB total

Distinctive model characteristics: token-efficient reasoning (`<think>` in "grug-speak"), "agentic" / "tool-use" / "conversational" tags. 97% token reduction on MATH-500 vs base, at equal accuracy.

### 1.2 Why STRIX_LEAN and not Q4_K_M
The `Q4_0_ROCMFP4_STRIX_LEAN` preset (type 106, ~4.38 bpw) is the reference target for Strix Halo:
- ~4.38 bits per weight → weights ~4× smaller than BF16 → ~4× less memory read per token → proportional tok/s on LPDDR5X UMA (~270 GB/s, memory-bound)
- Protects **attention K/V + Q5_K token embeddings** (keeps quality where it matters)
- FP4 is **pure software** on gfx1151 (RDNA 3.5 has no FP4 silicon units); the advantage is memory bandwidth only

Current production baseline: `Qwen3.6-35B-A3B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` (19 GB, **63 tok/s plain without MTP**, on `docker-llm-service` port :1234).

### 1.3 Fair comparison
grug-35b-v2 is a 3B-active MoE (like 35B-A3B), **not** a dense model (like Qwen3.6-27B). Therefore:
- The target pipeline is **plain, without MTP** (aligned with the 35B-A3B production baseline)
- LPDDR5X ceiling: 3B active × 0.5 bytes (FP4) = 1.5 GB/token → ~180 tok/s theoretical
- Expected plain speed: ~45-65 tok/s
- The lack of MTP in grug **is not a problem** — it aligns us with the MoE baseline

### 1.4 ROCmFPX container constraints (verified in adversarial review)
The `docker-llm-service` container (minimal Fedora 43) does NOT include the Python deps for HF→GGUF conversion:
- `transformers`, `torch`, `safetensors`, `sentencepiece`: **absent**
- `pip` / `python3 -m pip`: **absent**
- The converter is installed as a binary: `/usr/local/bin/convert_hf_to_gguf.py` (NOT in `/opt/llama.cpp/examples/`)

Phase 3 (conversion) therefore needs a derived image with Python deps. The other phases (imatrix, quantize, bench) use the base container because they do not require Python deps.

---

## 2. Goals and non-goals

### Goals
1. Produce `Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf` (~20 GB) optimized for Strix Halo
2. Produce `mmproj-grug-35b-v2-f16.gguf` to enable vision mode (optional at runtime)
3. Generate an imatrix calibrated on the model's domain (grug-think traces)
4. A/B compare vs `ProCreations/grug-35b-v2-Q4_K_M.gguf` (official baseline)
5. Document the pipeline in reusable shell scripts
6. Empirically verify that the qwen3_5_moe arch + Gated DeltaNet works in the current ROCmFPX container

### Non-goals
- Updating/rebasing the ROCmFPX fork (out of scope)
- Enabling MTP at runtime (the model has none, and alignment with the plain MoE pipeline)
- Fine-tuning, retraining, applying LoRA ex post
- Multi-GPU / distribution (single-node Strix Halo)
- Systematic quality validation (MMLU/SWE-bench) — functional smoke tests only

---

## 3. Pipeline architecture

9-phase pipeline (0-7, with 4a and 4b as sub-phases of phase 4) that replicates `quantize-qwen36-27b.sh` as a template, with three adaptations:
- **A1**: starts from safetensors (not an already-ready BF16 GGUF) → HF→GGUF conversion phase (requires the derived image `docker-llm-service-convert`)
- **A2**: imatrix not precomputed → on-prem generation with a custom calibration dataset
- **A3**: no MTP → no draft layer handling phase

```
[Build img] → [Setup+precheck] → [Download 70GB] → [Convert HF→GGUF] → [Prep calibration] → [Gen imatrix] → [Quantize ROCmFP4] → [Bench A/B] → [Cleanup]
    0              1                   2                  3                   4a                  4b                 5               6           7
```

Dedicated window: `llm-service` is stopped during phases 4b-6 (frees RAM and GPU; phases 0-4a are disk/CPU and do not disturb the production service). Restarted at the end of phase 6.

---

## 4. Detailed phases

> **Global convention**: all shell blocks assume `HF_TOKEN` exported at the start of the script (see Phase 1 precheck). If you copy-paste a block into a separate shell, re-run `export HF_TOKEN="$(cat ~/.cache/huggingface/token)"`.

### Phase 0 — Build the `docker-llm-service-convert` image (~10 min, one-time)
The base container lacks the Python deps for HF→GGUF conversion (see §1.4). Creates a reusable derived image (also valid for Ornith PHASE 2):

```bash
mkdir -p <lab-repo>/docker
cat > <lab-repo>/docker/Dockerfile.convert <<'EOF'
# Derived from docker-llm-service: adds Python deps for convert_hf_to_gguf.py
FROM docker-llm-service:latest

# Fedora 43 base — dnf for python3-pip, then pip for the ML packages.
# NOTE: --index-url REPLACES the default PyPI (pytorch.org hosts only torch/torchvision/torchaudio),
# so TWO separate pip installs are needed: torch from pytorch.org, the rest from PyPI.
RUN dnf -y --nodocs --setopt=install_weak_deps=False install python3-pip \
  && dnf clean all && rm -rf /var/cache/dnf/* \
  && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir transformers==5.14.1 safetensors sentencepiece accelerate
EOF

docker build -t docker-llm-service-convert:latest \
  -f <lab-repo>/docker/Dockerfile.convert \
  <lab-repo>/docker/
```

Expected derived image size: ~12 GB (base 10.1 + CPU torch ~800 MB + transformers/safetensors/sentencepiece/accelerate + deps ~1 GB). One-time, reusable for Ornith PHASE 2.

Verification:
```bash
docker run --rm docker-llm-service-convert python3 -c \
  "import transformers, torch, safetensors, sentencepiece; print('OK')"
# expected: OK
docker run --rm docker-llm-service-convert which convert_hf_to_gguf.py
# expected: /usr/local/bin/convert_hf_to_gguf.py
```

### Phase 1 — Setup + precheck
- Verify free disk ≥ 180 GB (pipeline peak)
- Verify `docker-llm-service` UP
- Verify the `docker-llm-service-convert` image is present (if not, see Phase 0)
- Verify HF reachable
- Create workspace: `~/llmodels/models/GRUG/35B-BF16/`
- Verify that `~/llmodels/models/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf` does not already exist
- Export `HF_TOKEN`: `export HF_TOKEN="$(cat ~/.cache/huggingface/token)"`

### Phase 2 — Download BF16 safetensors (~70 GB)
Use `wget -c` with Bearer (NOT `hf download` with hf_xet, which hangs on files > 1 GB on this network — known issue). The repo is public and not gated, but the Bearer header is kept for consistency with the proven pattern and for authenticated bandwidth:

```bash
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
REPO="https://huggingface.co/ProCreations/grug-35b-v2/resolve/main"
DLDIR=~/llmodels/models/GRUG/35B-BF16
mkdir -p "$DLDIR"

# 18 safetensors shards (~70 GB total)
for i in $(seq -w 1 18); do
  wget -c --header "Authorization: Bearer $HF_TOKEN" --tries=20 --timeout=60 --retry-connrefused \
    --progress=dot:giga \
    "${REPO}/model-000${i}-of-00018.safetensors" \
    -O "${DLDIR}/model-000${i}-of-00018.safetensors"
done

# Accessory files (~25 MB)
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja \
         model.safetensors.index.json preprocessor_config.json generation_config.json; do
  wget -c --header "Authorization: Bearer $HF_TOKEN" "${REPO}/${f}" -O "${DLDIR}/${f}"
done
```

Verification:
- `stat -c%s "${DLDIR}"/model-*.safetensors | awk '{s+=$1} END {printf "%.1f GB\n", s/1024/1024/1024}'`
- Expected: ~65.4 GiB (~70 GB)
- Integrity check: 18 shards + 7 accessory files, no `.safetensors.tmp` files left behind by interrupted wgets

### Phase 3 — HF→GGUF BF16 conversion
Uses the derived image `docker-llm-service-convert` (see Phase 0). The converter path inside the container is `/usr/local/bin/convert_hf_to_gguf.py`:

```bash
docker run --rm \
  -v ~/llmodels/models:/llmodels:rw \
  docker-llm-service-convert python3 /usr/local/bin/convert_hf_to_gguf.py \
    /llmodels/GRUG/35B-BF16 \
    --outfile /llmodels/GRUG/grug-35b-v2-BF16.gguf \
    --outtype bf16
```

Expected output (verified in adversarial review — the converter recognizes `Qwen3_5MoeForConditionalGeneration` as `MmprojModel`, so it produces both files):
- `grug-35b-v2-BF16.gguf` (~70 GB)
- `mmproj-grug-35b-v2-f16.gguf` (~857 MB) automatically extracted from the vision tower

Fallback if the converter does NOT produce the mmproj: download from the official GGUF repo:
```bash
wget -c --header "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/ProCreations/grug-35b-v2-gguf/resolve/main/mmproj-grug-35b-v2-f16.gguf" \
  -O ~/llmodels/models/GRUG/mmproj-grug-35b-v2-f16.gguf
```

Verify `general.architecture` of the produced GGUF = `qwen35moe` (expected; the fork's arch):
```bash
docker run --rm -v ~/llmodels/models:/llmodels:ro docker-llm-service \
  llama-gguf --verbose /llmodels/GRUG/grug-35b-v2-BF16.gguf 2>&1 | grep "general.architecture"
# expected: general.architecture = qwen35moe
```

### Phase 4a — Calibration text preparation (on HOST)
The script runs on the HOST Python 3.12 (which already has `datasets 5.0.0` + `jinja2 3.1.2`). It does NOT use `transformers` (not needed): it applies the chat template with `jinja2.Template` directly.

```python
# <lab-repo>/scripts/prep-grug-calibration.py
# Requires (on HOST): datasets, jinja2  (already present)
# Usage: python3 prep-grug-calibration.py
from datasets import load_dataset
from jinja2 import Environment, BaseLoader
from jinja2.exceptions import TemplateError
import os, random, sys

OUT_DIR = os.path.expanduser("~/llmodels/calibration")
TEMPLATE_PATH = os.path.expanduser("~/llmodels/models/GRUG/35B-BF16/chat_template.jinja")
OUT_PATH = os.path.join(OUT_DIR, "grug-calibration.txt")
N_SAMPLE = 800  # target ~4 MB plain text (imatrix best practice: 1-5 MB)

os.makedirs(OUT_DIR, exist_ok=True)
ds = load_dataset("ProCreations/grug-think-v3-10k", split="train")
print(f"[i] Dataset rows: {len(ds)}", file=sys.stderr)

with open(TEMPLATE_PATH) as f:
    chat_template_src = f.read()
# NOTE: trim_blocks/lstrip_blocks left at jinja2 defaults (False) to align
# with transformers.apply_chat_template behavior. Irrelevant for imatrix.
template = Environment(loader=BaseLoader()).from_string(chat_template_src)

random.seed(42)
sampled_idx = random.sample(range(len(ds)), min(N_SAMPLE, len(ds)))


def fallback_concat(row):
    """Simple R9 fallback: concatenates role:content for each message without the template.
    Good enough for imatrix (the goal is to activate the weights, not chat fidelity)."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])


rendered_lines, skipped = [], 0
for i in sampled_idx:
    row = dict(ds[i])  # robust: returns a standard dict even if the lib changes row type
    messages = row["messages"]
    tools = row.get("tools", [])
    try:
        rendered = template.render(
            messages=messages,
            add_generation_prompt=False,
            tools=tools,
        )
        rendered_lines.append(rendered)
    except TemplateError:
        # R9 fallback: if pure jinja2 cannot render, use the simple version
        rendered_lines.append(fallback_concat(row))
        skipped += 1
        continue

# R9 safety: if ALL renders fail, explicit alert (do not write an empty file)
if not rendered_lines:
    sys.exit("[✗] All renders failed — empty calibration text. See R9.")

print(f"[i] Rendered OK: {len(rendered_lines)}, of which fallback: {skipped}", file=sys.stderr)

with open(OUT_PATH, "w") as f:
    f.write("\n\n".join(rendered_lines))

size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"[✓] Calibration text: {OUT_PATH} ({size_mb:.2f} MB)", file=sys.stderr)
```

Run on host:
```bash
mkdir -p <lab-repo>/scripts
# (create the script above with the editor)
python3 <lab-repo>/scripts/prep-grug-calibration.py
```

Target size: ~3-5 MB plain text.

### Phase 4b — imatrix generation (in the base container, CPU-only)
Stop `llm-service` before this phase:

```bash
docker compose -f <your-workspace>/workspace/docker/docker-compose.yml stop llm-service

docker run --rm \
  -v ~/llmodels/models:/llmodels:rw \
  -v ~/llmodels/calibration:/calibration:ro \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-imatrix \
    -m /llmodels/GRUG/grug-35b-v2-BF16.gguf \
    -f /calibration/grug-calibration.txt \
    -o /llmodels/GRUG/imatrix-grug-35b-v2.gguf \
    --output-frequency 10 --save-frequency 0 \
    --threads 16 --chunks 256 \
    --no-ppl --parse-special \
    -ngl 0
```

- `-ngl 0`: CPU-only, weights in system RAM (LPDDR5X). imatrix generation is memory-bound on RAM, the GPU does not help
- `--threads 16`: compromise choice (at 32 the RAM/LPDDR5X bandwidth saturates; lesson from `quantize-qwen36-27b.sh`)
- `--chunks 256`: good dataset coverage
- `--no-ppl`: skips the perplexity computation (useless overhead for pure calibration)
- `--parse-special`: correctly tokenizes the special tags of the chat template (`<think>`, `<|im_start|>`, etc.) — critical for a reasoning dataset
- Output: `imatrix-grug-35b-v2.gguf` (~150-300 MB)

Expected time: 20-40 min.

### Phase 5 — ROCmFP4-STRIX_LEAN quantization (in the base container, CPU-only)
`llm-service` stays stopped (already stopped in Phase 4b).

Dry-run first (checks expected size without executing, ~80ms):
```bash
docker run --rm \
  -v ~/llmodels/models:/llmodels:ro \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-quantize --dry-run \
  /llmodels/GRUG/grug-35b-v2-BF16.gguf Q4_0_ROCMFP4_STRIX_LEAN 2>&1 | tail -10
```

Real quantization (`:rw` mount because it writes the output):
```bash
docker run --rm \
  -v ~/llmodels/models:/llmodels:rw \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-quantize \
    --imatrix /llmodels/GRUG/imatrix-grug-35b-v2.gguf \
    /llmodels/GRUG/grug-35b-v2-BF16.gguf \
    /llmodels/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
    Q4_0_ROCMFP4_STRIX_LEAN 16
```

- `16` is the last positional argument = `nthreads` (NOT a flag) — 32 saturate the LPDDR5X UMA bandwidth (lesson from Qwen3.6-27B and 35B-A3B)
- Expected output: ~18-22 GB
- Expected time: 3-5 min (40 MoE layers → faster than the 27B dense which took 2:43)

### Phase 6 — A/B bench + smoke test
`llm-service` already stopped in Phase 4b (stays stopped for the whole window).

Download the baseline `ProCreations/grug-35b-v2-Q4_K_M.gguf` (21.2 GB, via `wget -c`):
```bash
wget -c --header "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/ProCreations/grug-35b-v2-gguf/resolve/main/grug-35b-v2-Q4_K_M.gguf" \
  -O ~/llmodels/models/GRUG/grug-35b-v2-Q4_K_M.gguf
```

```bash
# ROCmFP4-STRIX_LEAN bench
docker run --rm \
  -v ~/llmodels/models:/llmodels:ro \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-bench \
    -m /llmodels/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
    -p 512 -n 128 -ngl 999 -fa on --mmap 0 \
    2>&1 | tee docs/benchmarks/bench-grug-rocmfp4.txt

# Q4_K_M baseline bench
docker run --rm \
  -v ~/llmodels/models:/llmodels:ro \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-bench \
    -m /llmodels/GRUG/grug-35b-v2-Q4_K_M.gguf \
    -p 512 -n 128 -ngl 999 -fa on --mmap 0 \
    2>&1 | tee docs/benchmarks/bench-grug-q4_k_m.txt
```

Sidecar smoke test (port 8083), `-c 32768` fixed to avoid context-shift (known DeltaNet issue on Qwen3.5-MoE):
```bash
docker rm -f grug-test 2>/dev/null || true
docker run -d --name grug-test -p 8083:1234 \
  -v ~/llmodels/models:/llmodels:ro \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-server \
    -m /llmodels/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
    --mmproj /llmodels/GRUG/mmproj-grug-35b-v2-f16.gguf \
    -ngl 999 -fa on --jinja -c 32768 \
    --host 0.0.0.0 --port 1234

# Wait for healthy (max 90s)
for i in {1..18}; do
  curl -s -o /dev/null -m 2 http://localhost:8083/health && break
  sleep 5
done

# Coding test
curl -s http://localhost:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"grug","messages":[{"role":"user","content":"Scrivi una funzione Python che calcola Fibonacci iterativamente. Solo codice."}]}' \
  | tee docs/benchmarks/smoke-grug-coding.json

# Tool-call JSON test
curl -s http://localhost:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"grug","messages":[{"role":"user","content":"Dammi le temperature di Roma in JSON con campi city, temp, unit."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}' \
  | tee docs/benchmarks/smoke-grug-toolcall.json

# Sidecar CLEANUP (important — frees 20+ GB RAM and port 8083)
docker rm -f grug-test >/dev/null
```

Report results in `docs/benchmarks/results-2026-08-11-grug.md`.

Restart `llm-service`:
```bash
docker compose -f <your-workspace>/workspace/docker/docker-compose.yml start llm-service
# wait for health on :1234 (max 5 min)
for i in {1..60}; do
  curl -s -o /dev/null -m 2 http://localhost:1234/health && break
  sleep 5
done
```

### Phase 7 — Cleanup
- Verify the quantized output is sound (size in range, smoke test passed)
- Delete safetensors (~70 GB)
- Delete the BF16 GGUF (~70 GB)
- Keep: ROCmFP4-STRIX_LEAN output, imatrix, mmproj, calibration text
- Verify final disk ≥ 100 GB free

---

## 5. Technical parameters

| Parameter | Value | Rationale |
|---|---|---|
| Quant preset | `Q4_0_ROCMFP4_STRIX_LEAN` | Strix Halo default (K/V + Q5_K emb protect, ~4.38 bpw) |
| `llama-quantize` threads (positional) | 16 | 32 saturate the LPDDR5X UMA bandwidth (lesson from Qwen3.6-27B and 35B-A3B) |
| `llama-imatrix --threads` | 16 | Same logic (CPU-only, weights in LPDDR5X system RAM) |
| `llama-imatrix --chunks` | 256 | Good coverage of the calibration text |
| `llama-imatrix --no-ppl` | (flag present) | Skips the perplexity computation (useless overhead for pure calibration) |
| `llama-imatrix --parse-special` | (flag present) | Correctly tokenizes the chat template's `<think>`, `<|im_start|>` etc. |
| Smoke test context size | 32768 fixed | Avoids context-shift → single-token spam (known DeltaNet issue on Qwen3.5-MoE) |
| Calibration dataset | `grug-think-v3-10k` only | Perfect target domain (grug-speak reasoning + tool-call + coding) |
| Calibration size | ~4 MB plain (~800 convs out of 10000) | imatrix best practice: 1-5 MB (see [llama.cpp discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263)) |
| BF16 download | `wget -c` with Bearer (optional header, public repo) | hf_xet hangs on files > 1 GB on this network (known issue) |
| Final mmproj name | `mmproj-grug-35b-v2-f16.gguf` | Consistent with ProCreations' official GGUF repo |

---

## 6. Success criteria

| Criterion | Target | Measure |
|---|---|---|
| Output size | 18-22 GB | `stat -c%s` |
| Plain tg128 (ROCmFP4) | ≥ Q4_K_M baseline | `llama-bench -n 128` (A/B) |
| Plain pp512 (ROCmFP4) | ≥ Q4_K_M baseline | `llama-bench -p 512` (A/B) |
| Absolute plain tg128 | ≥ 40 tok/s | Derivation: 35B-A3B ROCmFP4-STRIX_LEAN in production = 63 tok/s (3B active × 0.5 B = 1.5 GB/token, ceiling ~180). grug has the same 3B active and the same preset → I expect ~50-65 tok/s. Target ≥ 40 is conservative (−30% of the reference as margin: attention weights, MoE router, DeltaNet overhead). |
| Coding smoke test | Valid iterative Fibonacci function (Python syntax, correct logic) | inspect the JSON response |
| Tool-call smoke | Tool-call returned with well-formed JSON args | inspect the JSON response |
| No single-token spam | Normal conversation, no infinite `/` or loops | inspection (DeltaNet context-shift bug) |
| Final disk | ≥ 100 GB free after cleanup | `df -h` |

---

## 7. Risks and mitigations

| ID | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R1 | `convert_hf_to_gguf.py` does not recognize `qwen3_5_moe` with DeltaNet | very low | high (blocks Phase 3) | Already verified in adversarial review: the ROCmFPX fork's converter registers `Qwen3_5MoeForConditionalGeneration` as `MmprojModel` → `MODEL_ARCH.QWEN35MOE`. Phase 3 has the explicit check `general.architecture = qwen35moe`. If it fails: full log, then escalation (there is no equivalent fallback via LoRA: grug's LoRA is merged into the published model, NOT available as a separate adapter). Realistic fallback: do Ornith PHASE 2 first (same arch, BF16 GGUF already ready) to de-risk the pipeline, then retry grug. |
| R2 | DeltaNet path (`linear_attention` layers) does not work in the container at runtime | very low | high (inference NaN/spam) | Already demonstrated in production: `Qwen3.6-35B-A3B` (same Qwen3.5-MoE family, identical `[L,L,L,F]×10` pattern) serves in `llm-service` at 63 tok/s plain. The container has `--swa-full` and the `llama-memory-hybrid` infrastructure for recurrent+attention. The Phase 6 smoke test validates this empirically. |
| R3 | Context-shift corrupts DeltaNet state → single-token spam (infinite `/`) | medium | medium | Force `-c 32768` in all smoke tests (Phase 6); avoid `--context-shift`. The issue is documented in the README of ProCreations' official GGUF. |
| R4 | hf_xet stuck on the 3.9 GB shards during download | high | low | `wget -c` with `--tries=20 --timeout=60 --retry-connrefused` (known issue). `-c` resumes if the connection drops. |
| R5 | Disk saturated during conversion | low | high | Peak ~180 GB (safetensors 70 + BF16 GGUF 70 + output 20 + Q4_K_M bench 21); 257 GB currently free. Explicit cleanup at the end of Phase 7. |
| R6 | **Missing Python dependencies in the container** (transformers/torch/safetensors/sentencepiece) | high (certain) | high (blocks Phase 3) | Phase 0 builds the derived image `docker-llm-service-convert`. Verified in adversarial review: the base container is minimal Fedora 43 without pip. The converter starts with `from transformers import AutoConfig` so it is NOT independent of transformers. |
| R7 | Unacceptable quality regression vs Q4_K_M | low | high | Coding + tool-call smoke tests in Phase 6; if an obvious regression appears (invalid code, malformed JSON), try the `Q4_0_ROCMFP4_COHERENT` preset (Q6_K token embeddings, maximum JSON/tool quality, bpw ~4.70 vs 4.38 of STRIX_LEAN — size not yet measured in this workspace, expected slightly higher than STRIX_LEAN due to the Q6_K emb, ~+200-400 MB on a 35B MoE). |
| R8 | Calibration text too homogeneous → imatrix overfitting | low | medium | `grug-think-v3-10k` has 6 different sources with counts verified from the README: glaive 1000 / hermes 1000 / nebius 2200 / smith_ticks 2100 / smith_tool 2200 / toolace 1500. Intrinsic diversity assured. |
| R9 | Phase 4a script fails: grug chat template does not render with pure jinja2 | medium | medium | grug's chat_template.jinja is standard Qwen3.5 (verified: 7.54 kB); jinja2 should render it. Simple fallback on TemplateError: concatenate `role: content` for each message without the template (good enough for imatrix — the goal is to activate the weights, not chat fidelity). |
| R10 | `docker-llm-service-convert` image build fails (heavy CPU torch wheel, network bandwidth) | low | medium | Pre-build in Phase 0 with margin; real CPU `torch` wheel ~800 MB via `--index-url https://download.pytorch.org/whl/cpu`. If it fails, fallback: install deps in a HOST venv and copy the converter from the container. |

---

## 8. Deliverables

1. `~/llmodels/models/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf` (~20 GB)
2. `~/llmodels/models/GRUG/mmproj-grug-35b-v2-f16.gguf` (~857 MB)
3. `~/llmodels/models/GRUG/imatrix-grug-35b-v2.gguf` (~150-300 MB)
4. `~/llmodels/calibration/grug-calibration.txt` (~4 MB)
5. `docker/Dockerfile.convert` (derived image for conversion)
6. `scripts/prep-grug-calibration.py` (calibration text preparation)
7. `scripts/quantize-grug-35b-v2.sh` (replica of `quantize-qwen36-27b.sh` with adaptations)
8. `docs/benchmarks/results-2026-08-11-grug.md`
9. `docs/benchmarks/bench-grug-{rocmfp4,q4_k_m}.txt`
10. `docs/benchmarks/smoke-grug-{coding,toolcall}.json`

---

## 9. Final serving command (for future deploy)

```bash
llama-server -m /llmodels/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  --mmproj /llmodels/GRUG/mmproj-grug-35b-v2-f16.gguf \
  -ngl 999 -fa on --jinja -c 32768 \
  --host 0.0.0.0 --port 1234 \
  --reasoning on --reasoning-budget 8192 \
  --temp 0.6 --top-p 0.95 --top-k 20
```

(Without `--spec-type draft-mtp`: grug has no MTP, and alignment with the plain MoE pipeline.)

---

## 10. Follow-up

### PHASE 2 — Ornith-1.0-35B (separate spec)
Same pipeline applied to `unsloth/Ornith-1.0-35B-GGUF`:
- BF16 GGUF already ready (69.4 GB, 18 shards) — no HF→GGUF conversion
- imatrix `imatrix_unsloth.gguf_file` already included (192 MB)
- mmproj-F16 included (899 MB)
- `mtp_num_hidden_layers=1` (MTP present) → **but not enabled at runtime** for alignment with the plain MoE pipeline
- Technical decision: keep the MTP weights (`blk.40`) in the final GGUF (~1-2 GB extra) or exclude them at conversion — to be evaluated in PHASE 2

### Optional tests (out-of-scope for this spec)
- Systematic quality benchmark (MMLU-Pro, SWE-bench) vs official Q4_K_M
- `STRIX_LEAN` vs `COHERENT` preset comparison for JSON tool-call quality
- Long context test (>32K) with stress on the DeltaNet recurrent state

---

## 11. References

- Reference pipeline: `scripts/quantize-qwen36-27b.sh` (Qwen3.6 pipeline, not included in this repo)
- Previous report: `docs/benchmarks/results-2026-08-10.md`
- Previous plan: internal implementation plan `2026-08-10-qwen36-27b-rocmfp4-strix-lean-impl.md` (not included in this repo)
- Memory: `rocmfpx-toolchain-strix-halo`, `rocmfpx-strix-lean-pipeline-status`, `hf-download-stuck-xet`
- Source model: https://huggingface.co/ProCreations/grug-35b-v2
- Official baseline GGUF: https://huggingface.co/ProCreations/grug-35b-v2-gguf
- Calibration dataset: https://huggingface.co/datasets/ProCreations/grug-think-v3-10k
- Base model: https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B (alias: `ornith-ai/Ornith-1.0-35B`)
- ROCmFPX fork: https://github.com/charlie12345/ROCmFPX (commit `00d5452`, Aug 9 2026)
- imatrix doc: https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md
