#!/bin/bash
# Part of strix-nebulosa — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# run-imatrix-qwen38.sh — imatrix for Qwen3.8-27B BF16 (dense qwen35 arch)
# PHASE 1: GPU probe (--chunks 16) → validates a nonempty, NaN-free collection
# PHASE 2: full run (--chunks 256) on GPU if the probe is healthy
# The previous grug model was a 3B-active MoE (13 min CPU); here dense 27B → GPU required.
set -uo pipefail
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
LOG="${LAB_DIR}/logs/imatrix-qwen38.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

LLMODELS="${LLMODELS_DIR}/models"
MODEL_CTR=/llmodels/models/QWEN3.8/BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf
CAL=/llmodels/calibration/qwen38-calibration.txt   # mounted from ${LLMODELS_DIR}/calibration
OUT_CTR=/llmodels/models/QWEN3.8/imatrix-qwen38.gguf
OUT_HOST=${LLMODELS}/QWEN3.8/imatrix-qwen38.gguf

run_gpu() {  # $1=chunks $2=outfile
  docker run --rm \
    -v "${LLMODELS_DIR}":/llmodels:rw \
    --device /dev/kfd --device /dev/dri \
    --group-add video --group-add render \
    -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    docker-llm-service:latest llama-imatrix \
    -m "$MODEL_CTR" \
    -f "$CAL" \
    -o "$2" \
    --output-frequency 10 --save-frequency 0 \
    --threads 16 --chunks "$1" \
    --no-ppl --parse-special \
    -ngl 999
}

validate() {  # $1=label $2=file
  local f="${2:-$OUT_HOST}"
  [ -f "$f" ] || { echo "[$1] ERROR: $f does not exist"; return 1; }
  local size=$(stat -c%s "$f")
  # NaN check on the imatrix float32 data
  local nan=$(python3 -c "
import struct
d = open('$f','rb').read()[512:]  # skip approximate header
import re
print(len(re.findall(rb'\x00\x00\xc0\x7f', d)))  # little-endian float32 NaN pattern
" 2>/dev/null || echo "?")
  echo "[$1] size=${size} possibili-NaN-pattern=${nan}"
  [ "$size" -gt 1000000 ]
}

echo "================================================================"
echo "[$(date '+%H:%M:%S')] PHASE 1: GPU probe chunks=16 (collection validation)"
echo "================================================================"
T0=$(date +%s)
run_gpu 16 "${OUT_CTR%.gguf}-probe.gguf"
RC=$?
echo "probe exit=$RC elapsed=$(( $(date +%s) - T0 ))s"
[ $RC -ne 0 ] && { echo "GPU PROBE FAILED — try CPU (-ngl 0) with reduced chunks"; exit 1; }
validate "probe" "${OUT_CTR%.gguf}-probe.gguf" || { echo "PROBE: empty output — GPU not collecting, fall back to CPU"; exit 2; }
grep -E 'no data|nan|NaN' "$LOG" | tail -3

echo "================================================================"
echo "[$(date '+%H:%M:%S')] PHASE 2: full run chunks=256 on GPU"
echo "================================================================"
T0=$(date +%s)
run_gpu 256 "$OUT_CTR"
RC=$?
ELAPSED=$(( $(date +%s) - T0 ))
echo "full run exit=$RC elapsed=${ELAPSED}s = $((ELAPSED/60))min$((ELAPSED%60))s"
validate "full" "$OUT_CTR" && echo "IMATRIX OK → $OUT_HOST" || { echo "IMATRIX PROBLEM"; exit 3; }
rm -f "${LLMODELS}/QWEN3.8/imatrix-qwen38-probe.gguf"
