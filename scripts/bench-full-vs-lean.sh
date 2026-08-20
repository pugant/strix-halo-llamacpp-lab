#!/bin/bash
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# bench-full-vs-lean.sh — Task 4 of the 2026-08-18-rocmfp4-full-vs-strix-lean plan (internal plan, not included in this repo)
# Vulkan server MTP n6 p_min 0.75 c 16384 (production-equivalent), warm-up + 3 runs,
# TREATMENT marker in the logs (lesson 17/08). Vulkan backend ONLY (production).
set -u
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
OUT="${LAB_DIR}/logs/bench-full-vs-lean"
mkdir -p "$OUT"
PORT=1235
NMAX=6
IMG=docker-llm-service:vulkan-fork-ckpt7

start_server() { # $1 model $2 tag
  local m="$1"
  local tag="$2"
  local name="bench-${tag}-q38"
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --name "$name" --network host \
    --device /dev/dri --group-add render \
    -v "${LLMODELS_DIR}/models:/llmodels:ro" \
    -v "${LAB_DIR}/models-test:/out:ro" \
    --entrypoint llama-server "$IMG" \
    -m "$m" -ngl 999 -fa on --jinja -c 16384 \
    --host 127.0.0.1 --port "$PORT" \
    --spec-type draft-mtp --spec-draft-ngl all --spec-draft-n-max "$NMAX" \
    --spec-draft-p-min 0.75 --spec-draft-p-split 0.10 >/dev/null || return 1
  for i in $(seq 1 120); do
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q 200 && return 0
    sleep 5
  done
  echo "FAIL health ${name}:"; docker logs "$name" 2>&1 | tail -20
  return 1
}

bench_tok() { # $1 prompt
  curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"q\",\"max_tokens\":600,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}" |
    python3 -c "import json,sys; t=json.load(sys.stdin).get('timings',{}); print(round(t.get('predicted_per_second',0),1))" 2>/dev/null || echo "ERR"
}

for arm in "LEAN:/llmodels/QWEN3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf" \
           "FULL:/out/Qwen3.8-27B-Q4_0_ROCMFP4-full.gguf" \
           "EVEN:/out/Qwen3.8-27B-Q4_0_ROCMFP4-even.gguf"; do
  # if an argument is passed (LEAN|FULL|EVEN), run only that arm
  if [ -n "${1:-}" ] && [ "${1^^}" != "${arm%%:*}" ]; then continue; fi
  TAG="${arm%%:*}"; M="${arm##*:}"
  echo "== TREATMENT=${TAG} model=${M} n-max=${NMAX} p_min=0.75 c=16384 vulkan =="
  if start_server "$M" "$TAG"; then
    bench_tok 'Rispondi solo OK.' >/dev/null   # warm-up discarded
    P1a=$(bench_tok 'Scrivi un paragrafo dettagliato sulla storia di Roma.')
    P1b=$(bench_tok 'Scrivi un saggio breve sulla stampa e il Rinascimento italiano.')
    P2a=$(bench_tok 'Conta da 1 a 200, un numero per riga, solo i numeri.')
    P2b=$(bench_tok 'Elenco l alfabeto inglese una lettera per riga, poi ripetilo al contrario.')
    echo "RESULT TREATMENT=${TAG}: prose ${P1a}/${P1b} | det ${P2a}/${P2b} tok/s"
    docker logs "bench-${TAG}-q38" > "$OUT/bench-${TAG}.log" 2>&1
    grep -E 'statistics draft' "$OUT/bench-${TAG}.log" | tail -2
  else
    echo "RESULT TREATMENT=${TAG}: SERVER FAIL"
  fi
  docker rm -f "bench-${TAG}-q38" >/dev/null 2>&1
done
echo "DONE bench-full-vs-lean"
