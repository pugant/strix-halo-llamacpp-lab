# Replication guide — Qwen3.8-27B (canonical)

End-to-end: BF16 GGUF → imatrix → ROCmFP4-STRIX_LEAN quant → sanitized GGUF → fork server build → dual-drafter server. Every script referenced below is in this repo under `scripts/`; the experiment notes behind each step live in [`docs/experiments/`](../experiments/).

### Step 0 — Prerequisites

- AMD Strix Halo with 128 GB unified memory (Ryzen AI MAX+ 395), a ROCm-capable container stack, and an HF account.
- Free disk: plan for **~75 GB** end-to-end (the BF16 shards alone are ~54.7 GB, plus imatrix and the ~14.8 GB quant output; the quantize script itself pre-checks for 40 GB free at its step).
- Build the **base container first** from [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) (the `docker-llm-service` image). Nothing here works without it.

### Step 1 — Build the convert container

```bash
docker build -t docker-llm-service:convert -f docker/Dockerfile.convert .
```

Note: the base image (`docker-llm-service:latest`) is **not publicly pullable** — it must be built locally from the toolboxes (Step 0). The convert container adds the Python deps needed by the fork's `convert_hf_to_gguf.py` and its custom `gguf-py`.

### Step 2 — Download the target (BF16) and the DFlash2 drafter

```bash
LLMODELS_DIR=~/llmodels HF_TOKEN=... bash scripts/download-qwen38-bf16.sh

# DFlash2 draft model (1.9B; the Q4_K_M file is ~1.1 GB):
huggingface-cli download incoai/Qwen3.8-27B-DFlash2-GGUF \
  --include "*Q4_K_M*" --local-dir "$LLMODELS_DIR/models/QWEN3.8"
```

Env: `LLMODELS_DIR` (default `$HOME/llmodels`), `HF_TOKEN`. The script downloads `unsloth/Qwen3.8-27B-GGUF` BF16 (2 shards, ~54.7 GB) + mmproj; idempotent (`wget -c`). On our high-latency network the `hf`/xet downloader sometimes stalls on files around 1 GB — `wget -c` with a Bearer header is the fallback we use (see the notes in [`docs/experiments/`](../experiments/)).

### Step 3 — Importance matrix

**Canonical path — use the published matrix:** download it from [`pugant/Qwen3.8-27B-imatrix`](https://huggingface.co/pugant/Qwen3.8-27B-imatrix).

**Regenerating it yourself** (what we did): the calibration corpus is a composite of agentic traces + technical prose + real code (~1.7 MB), because the quant must serve both chat and tool-calling:

```bash
python3 scripts/prep-qwen38-calibration.py   # build the calibration corpus
bash scripts/run-imatrix-qwen38.sh           # GPU run: 256 chunks x 512 (probe first)
python3 scripts/check-imatrix-coverage.py    # coverage gate
```

The coverage gate must pass: **all 496 expected tensors covered, layers 0–63, zero NaN**.

### Step 4 — Quantize

```bash
bash scripts/quantize-qwen38-27b.sh
```

Produces `Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` (≈ 13.8 GiB, 4.34 bpw, MTP layer included; automatic dry-run before the real pass).

### Step 5 — Sanitize GGUF metadata

```bash
# CRITICAL: run this ONLY inside the convert container.
# <config.json> is the model's HF config (tokenizer/architecture metadata).
docker run --rm \
  -v "$LLMODELS_DIR/models:/llmodels" -v "$PWD:/lab:ro" \
  docker-llm-service:convert \
  python3 /lab/scripts/sanitize-gguf-v3.py <input.gguf> <output.gguf> <config.json>
```

The host-side `gguf-py` does **not** know the fork's custom tensor types — running the sanitizer on the host corrupts or rejects the file. This is the single most common way to waste an afternoon here.

### Step 6 — Build the fork's llama-server (Vulkan RADV)

The server binary is **not** distributed: build it from the sources, **already included in this repo under `rocmfpx/`** (no extra clone needed). We build inside Docker with the Vulkan Dockerfile from kyuz0's toolboxes, feeding it our sources instead of its default clone (`$TB` = your checkout of [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)):

```bash
mkdir -p image-build/src
cp -r rocmfpx/. image-build/src/          # the full fork source, included in this repo
sed -e 's|^RUN git clone -b ${BRANCH} --single-branch ${REPO} \.$|COPY src/ .|' \
    -e 's|^RUN git clean -xdf \\$|RUN patch -p1 < /tmp/llama-grammar.patch \\|' \
    -e 's|^  \&\& patch -p1 < /tmp/llama-grammar.patch \\$|  \&\& true \\|' \
    "$TB/toolboxes/Dockerfile.vulkan-rocmfpx" > image-build/Dockerfile
cp "$TB"/toolboxes/{llama-grammar.patch,rocmfpx-vulkan-shader-concurrency.patch,gguf-vram-estimator.py} image-build/
docker build -t docker-llm-service:vulkan-fork-dflash2-route image-build
```

The companion files come from the toolboxes too — both patches (`llama-grammar.patch` and `rocmfpx-vulkan-shader-concurrency.patch`) are required by their base Dockerfile and are their work, not ours. Build time is ~7 min on our toolboxes revision; upstream `main` builds with `--parallel 1` (to keep concurrent shader generation from exhausting memory) and will run longer. **Verified backend:** every `llama-server` invocation and every speculative-decoding benchmark was verified on the fork's **Vulkan (RADV)** build produced here — the ROCm column of the backend-choice table was measured on the fork's ROCm build. The full `docker run` pattern we use (devices, render group, `--entrypoint llama-server`) is in `scripts/test-drafter-routing-t1.sh`.

### Step 7 — Serve with dual drafters

Verified invocation — the flags exactly as we run them in production and in the test suite:

```bash
llama-server -m Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  -ngl 999 -fa on --jinja -c 16384 \
  --spec-type draft-mtp,draft-dflash \
  --spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-draft-ngl all --spec-draft-n-max 7 \
  --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
```

Flag notes, learned the hard way:

- `--spec-type` takes a **comma-separated list** of draft implementations. There is **no** `--spec-draft-type` flag.
- `--spec-draft-p-split` is accepted for CLI compatibility but is a **no-op in `llama-server`** — its only consumer is the upstream `examples/speculative` example.
- The `-k` / `-v` variants floating around in discussions are KV-cache type flags, not drafter selection.
- In dual mode the server clamps MTP to n-max 6 and DFlash2 to 7 (its block size is 8).
- The DFlash2 drafter is [`incoai/Qwen3.8-27B-DFlash2-GGUF`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) (1.9B params; the Q4_K_M file is ≈ 1.1 GB).

### Step 8 — Exercise the routing

```bash
# 1) tools present in the request body -> auto-routed to DFlash2
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"List the files in /tmp"}],
       "tools":[{"type":"function","function":{"name":"list_files",
                  "description":"List files in a directory",
                  "parameters":{"type":"object","properties":{}}}}]}'

# 2) explicit override, no tools needed
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Count from 1 to 50, one per line"}],
       "spec_drafter":"dflash"}'

# 3) per-request thinking budget
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Solve this step by step"}],
       "thinking_budget_tokens":4096}'
```

### Step 9 — Smoke-test the routing

```bash
bash scripts/test-drafter-routing-t1.sh
```

14 checks: dual boot / default policy / override / invalid-enum 400 / boot fallback / cache switch across drafter changes / metrics endpoint. Companion suites: `test-drafter-routing-t2.sh` (cache round-trip), `test-drafter-routing-t3.sh` (sacred paths), `test-ckpt-rollback-t1t2.sh` (checkpoint rollback), `bench-routing-vs-mono.sh` (the T4 A/B).

### Other models

The same pipeline produced the other published quants. Their quantize scripts are **not** in this repo — the canonical, fully scripted flow is the Qwen3.8-27B one above.

| Model | Class | HF card |
|---|---|---|
| grug-35b-v2 | MoE 35B-A3B, reasoning/tool-call finetune | [`pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN) |
| Ornith-1.0-35B | MoE 35B-A3B, **multimodal** | [`pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN) |
| Ornith-1.5-35B | MoE 35B-A3B, **multimodal**; ships an MTP head whose pos-2 acceptance collapsed to ~0.07 (0.99 pos-1) — speculative decoding loses to plain at every n-max | [`pugant/Ornith-1.5-35B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/Ornith-1.5-35B-ROCmFP4-STRIX_LEAN) |
| Nemotron-3.5-Lightning-30B-A3B | **Mamba-hybrid** MoE | [`pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN`](https://huggingface.co/pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN) |
| Qwen3.6-35B-A3B | MoE 35B-A3B, Q6_0 | [`pugant/Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX`](https://huggingface.co/pugant/Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX) |
