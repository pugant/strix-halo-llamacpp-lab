#!/bin/bash
# Part of strix-nebulosa — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# Download BF16 GGUF unsloth/Qwen3.8-27B-GGUF (2 shard, ~54.7 GB) + mmproj
# Usage: bash download-qwen38-bf16.sh  (idempotent: wget -c)
set -uo pipefail
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
LOG="${LAB_DIR}/logs/download-qwen38-bf16.log"
mkdir -p "$(dirname "$LOG")" "${LLMODELS_DIR}/models/QWEN3.8/BF16"
exec > >(tee -a "$LOG") 2>&1

REPO="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main"
DEST="${LLMODELS_DIR}/models/QWEN3.8/BF16"

echo "================================================================"
echo "[$(date '+%H:%M:%S')] START download Qwen3.8-27B BF16 (2 shard + mmproj)"
echo "================================================================"

# Double-download guard (lesson from 2026-08-15)
if pgrep -f 'wget.*Qwen3.8' | grep -qv $$; then
  echo "ERROR: a Qwen3.8 wget is already active — exiting"; pgrep -af 'wget.*Qwen3.8'; exit 1
fi

for f in "BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf" "BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf"; do
  name=$(basename "$f")
  echo "--- $name ---"
  wget -c -q --show-progress --progress=dot:giga "${REPO}/${f}" -O "${DEST}/${name}"
  rc=$?
  echo "[$(date '+%H:%M:%S')] $name exit=$rc size=$(stat -c%s "${DEST}/${name}" 2>/dev/null || echo 0)"
  [ $rc -ne 0 ] && [ $rc -ne 0 ] && echo "WARNING: exit=$rc on $name"
done

# mmproj F16 (for possible VL use, 0.93 GB — can also be downloaded later)
wget -c -q --show-progress --progress=dot:giga "${REPO}/mmproj-F16.gguf" -O "${LLMODELS_DIR}/models/QWEN3.8/mmproj-F16.gguf"

s1=$(stat -c%s "${DEST}/Qwen3.8-27B-BF16-00001-of-00002.gguf" 2>/dev/null || echo 0)
s2=$(stat -c%s "${DEST}/Qwen3.8-27B-BF16-00002-of-00002.gguf" 2>/dev/null || echo 0)
echo "================================================================"
echo "[$(date '+%H:%M:%S')] END: shard1=$((s1/1024/1024)) MB shard2=$((s2/1024/1024)) MB totale=$(( (s1+s2)/1024/1024/1024 )) GB"
# Expected: shard1=49990 MB, shard2=4670 MB, total ~54 GB
