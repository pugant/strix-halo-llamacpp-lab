#!/bin/bash
# Part of strix-nebulosa — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# bench-dflash-vs-mtp.sh — T7 A/B: DFlash2 vs MTP drafter on Qwen3.8-27B STRIX_LEAN
# Plan: docs/superpowers/plans/2026-08-19-dflash2-vs-mtp-ab.md (internal plan, not included in this repo)
# Standard protocol from bench-full-vs-lean.sh: DEDICATED GPU (llm-service STOPPED,
# managed externally), -c 16384, p_min 0.75, temp 0, warm-up discarded,
# TREATMENT marker in the logs (lesson 17/08).
set -u
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
OUT="${LAB_DIR}/logs/bench-dflash-vs-mtp"
mkdir -p "$OUT"
PORT=1235
MODEL=/llmodels/QWEN3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf
DRAFTER=/llmodels/QWEN3.8/Qwen3.8-27B-DFlash2-Q4_K_M.gguf

start_server() { # $1=img $2=tag $3=type $4=nmax $5=drafter(or empty)
  local img="$1" tag="$2" tipo="$3" nmax="$4" dft="${5:-}"
  local name="bench-${tag}-q38"
  local extra=()
  [ -n "$dft" ] && extra+=(--spec-draft-model "$dft")
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --name "$name" --network host \
    --device /dev/dri --group-add render \
    -v "${LLMODELS_DIR}/models:/llmodels:ro" \
    --entrypoint llama-server "$img" \
    -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 \
    --host 127.0.0.1 --port "$PORT" \
    --spec-type "$tipo" --spec-draft-ngl all --spec-draft-n-max "$nmax" \
    --spec-draft-p-min 0.75 --spec-draft-p-split 0.10 "${extra[@]}" >/dev/null || return 1
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

run_arm() { # $1 img $2 tag $3 type $4 nmax $5 drafter
  echo "== TREATMENT=${2} tipo=${3} n-max=${4} model=LEAN p_min=0.75 c=16384 vulkan =="
  if start_server "$1" "$2" "$3" "$4" "${5:-}"; then
    bench_tok 'Rispondi solo OK.' >/dev/null   # warm-up discarded
    P1a=$(bench_tok 'Scrivi un paragrafo dettagliato sulla storia di Roma.')
    P1b=$(bench_tok 'Scrivi un saggio breve sulla stampa e il Rinascimento italiano.')
    P2a=$(bench_tok 'Conta da 1 a 200, un numero per riga, solo i numeri.')
    P2b=$(bench_tok 'Elenco l alfabeto inglese una lettera per riga, poi ripetilo al contrario.')
    echo "RESULT TREATMENT=${2}: prose ${P1a}/${P1b} | det ${P2a}/${P2b} tok/s"
    docker logs "bench-${2}-q38" > "$OUT/bench-${2}.log" 2>&1
    grep -E 'statistics draft|mean acceptance' "$OUT/bench-${2}.log" | tail -3
  else
    echo "RESULT TREATMENT=${2}: SERVER FAIL"
  fi
  docker rm -f "bench-${2}-q38" >/dev/null 2>&1
}

# arms (order: control first)
run_arm docker-llm-service:vulkan-fork-ckpt7    MTP6 draft-mtp    6 ""
run_arm docker-llm-service:vulkan-fork-dflash2  DF7   draft-dflash 7 "$DRAFTER"
run_arm docker-llm-service:vulkan-fork-dflash2  DF5   draft-dflash 5 "$DRAFTER"
echo "=== END (TREATMENT marker on every RESULT line) ==="
