#!/bin/bash
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# quantize-qwen38-27b.sh — Qwen3.8-27B BF16 → Q4_0_ROCMFP4_STRIX_LEAN
# Adapted from quantize-qwen36-27b.sh:
#   - imatrix AUTOPRODOTTA (unsloth non la pubblica per Qwen3.8) → run-imatrix-qwen38.sh
#   - MAI stop/start llm-service (regola workspace: gestione manuale utente)
#   - non interattivo (autonomia)
#   - dry-run automatico prima della quantizzazione vera
set -euo pipefail

LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
LLMODELS_HOST="${LLMODELS_DIR}/models"
QWEN_DIR_HOST="${LLMODELS_HOST}/QWEN3.8"
BF16_FIRST_SHARD="${QWEN_DIR_HOST}/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf"
IMATRIX_HOST="${QWEN_DIR_HOST}/imatrix-qwen38.gguf"
OUTPUT_NAME="Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf"
OUTPUT_HOST="${QWEN_DIR_HOST}/${OUTPUT_NAME}"
PRESET="Q4_0_ROCMFP4_STRIX_LEAN"
NTHREADS=16
EXPECTED_OUT_MIN=13.5   # 27B × 4.38 bpw / 8 ≈ 14.8 GB; arco prudente (deltanet può aggiungere)
EXPECTED_OUT_MAX=16.5

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} $*"; }
ok()   { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}⚠${NC} $*"; }
err()  { echo -e "${RED}✗${NC} $*" >&2; }

docker_run() {  # $1=ro|rw, resto comando
  local mount_mode="$1"; shift
  docker run --rm \
    -v "${LLMODELS_HOST}:/llmodels:${mount_mode}" \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    docker-llm-service:latest "$@"
}

MODE="${1:---all}"
log "Pipeline Qwen3.8-27B → ROCmFP4-STRIX_LEAN | mode=${MODE} (preset ${PRESET})"

precheck() {
  local avail_gb; avail_gb=$(df --output=avail -B1G "$LLMODELS_DIR" | tail -1 | tr -d ' ')
  [ "$avail_gb" -ge 40 ] && ok "Disco: ${avail_gb} GB liberi (≥40)" || { err "Disco: ${avail_gb} GB < 40"; exit 1; }

  docker ps --format '{{.Names}}' | grep -q '^llm-service$' \
    && warn "llm-service UP — quantize CPU-only può contentionare la banda UMA (procedo, regola: non lo tocco)"

  local s1 s2
  s1=$(stat -c%s "${QWEN_DIR_HOST}/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf" 2>/dev/null || echo 0)
  s2=$(stat -c%s "${QWEN_DIR_HOST}/BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf" 2>/dev/null || echo 0)
  [ $(( (s1+s2)/1024/1024/1024 )) -ge 50 ] && ok "BF16 completo (50.9 GiB attesi ($(( (s1+s2)/1024/1024/1024 )) GB))" || { err "BF16 incompleto"; exit 1; }

  [ -s "$IMATRIX_HOST" ] && ok "Imatrix presente" || { err "Imatrix mancante — lanciare run-imatrix-qwen38.sh"; exit 1; }

  [ ! -f "$OUTPUT_HOST" ] && ok "Output target non esiste" || { err "$OUTPUT_HOST esiste già — rimuoverlo prima"; exit 1; }
}

dryrun() {
  log "Dry-run (size attesa senza quantizzare)"
  docker_run ro llama-quantize --dry-run \
    "/llmodels/QWEN3.8/BF16/$(basename "$BF16_FIRST_SHARD")" \
    "$PRESET" 2>&1 | tail -12
}

quantize() {
  mkdir -p "${LAB_DIR}/logs"
  local start_ts; start_ts=$(date +%s)
  log "Quantizzazione (nthreads=${NTHREADS}, imatrix autoprodotta)"
  docker_run rw llama-quantize \
    --imatrix "/llmodels/QWEN3.8/imatrix-qwen38.gguf" \
    "/llmodels/QWEN3.8/BF16/$(basename "$BF16_FIRST_SHARD")" \
    "/llmodels/QWEN3.8/${OUTPUT_NAME}" \
    "$PRESET" "$NTHREADS" 2>&1 | tee -a "${LAB_DIR}/logs/quantize-qwen38.log"

  local elapsed=$(( $(date +%s) - start_ts ))
  ok "Quantizzazione completata in $((elapsed/60)) min $((elapsed%60)) sec"

  local out_gb; out_gb=$(awk "BEGIN {printf \"%.2f\", $(stat -c%s "$OUTPUT_HOST")/1024/1024/1024}")
  log "Output: ${OUTPUT_NAME} = ${out_gb} GB"
  awk "BEGIN {exit !(${out_gb} >= ${EXPECTED_OUT_MIN} && ${out_gb} <= ${EXPECTED_OUT_MAX})}" \
    && ok "Size nel range atteso [${EXPECTED_OUT_MIN}-${EXPECTED_OUT_MAX}] GB" \
    || warn "Size FUORI range atteso — verificare"
}

case "$MODE" in
  --precheck)    precheck ;;
  --dry-run)     precheck; dryrun ;;
  --quantize)    precheck; dryrun; quantize ;;
  --all)         precheck; dryrun; quantize ;;
  *) err "Mode sconosciuto: $MODE"; echo "Usage: $0 [--precheck|--dry-run|--quantize|--all]"; exit 1 ;;
esac
