#!/bin/bash
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# test-drafter-routing-t5.sh — T5 spot-check numerica (Task 10 piano T7-f2 — internal plan, not included in this repo — ciclo RS 20/08)
# Gate G4 del piano 2026-08-20-rs-rollback-dflash-experiment (internal plan, not included in this repo):
#   (a) MONO-MTP6 (immagine route)  ==  (b) DUALE+override mtp   -> char-identical
#   (c) DUALE+override dflash        vs  (d) DF7 immagine dflash2 ORIGINALE (RS off)
#   se (c)!=(d): (e) mono-DF7 sulla NUOVA immagine (isola effetto RS dal routing).
# Prompt greedy fisso (piano Task 10): radice quadrata, max_tokens 400, temp 0.
# Output: logs/test-drafter-routing/t5-rs-*.json
set -u
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
OUT="${LAB_DIR}/logs/test-drafter-routing"
mkdir -p "$OUT"
PORT=8090
MODEL=/llmodels/QWEN3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf
DRAFTER=/llmodels/QWEN3.8/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
IMG_NEW=docker-llm-service:vulkan-fork-dflash2-route   # RS ON (commit 38483cc64+)
IMG_ORIG=docker-llm-service:vulkan-fork-dflash2         # RS OFF (riferimento T7)
PROMPT='Spiega in 3 righe cos e la radice quadrata, poi calcola quella di 144.'

boot() { # $1=name $2=img $3=tipo $4=nmax $5=drafter(o vuoto)
  local name="$1" img="$2" tipo="$3" nmax="$4" dft="${5:-}"
  local extra=()
  [ -n "$dft" ] && extra+=(--spec-draft-model "$dft")
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --name "$name" --network host --device /dev/dri --group-add render \
    -v "${LLMODELS_DIR}/models:/llmodels:ro" --entrypoint llama-server "$img" \
    -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 -lv 3 \
    --host 127.0.0.1 --port "$PORT" \
    --spec-type "$tipo" --spec-draft-ngl all --spec-draft-n-max "$nmax" \
    --spec-draft-p-min 0.75 --spec-draft-p-split 0.10 "${extra[@]}" >/dev/null || return 1
  for i in $(seq 1 120); do curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q 200 && return 0; sleep 1; done
  return 1
}
ask() { # $1 outfile $2 spec_drafter(o vuoto)
  local OV=""
  [ -n "$2" ] && OV="\"spec_drafter\":\"$2\","
  curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{$OV\"model\":\"q\",\"max_tokens\":400,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" > "$1"
}
arm() { # $1=lettera $2=name $3=img $4=tipo $5=nmax $6=drafter $7=override
  echo "== T5 braccio ($1) img=$3 tipo=$4 n-max=$5 override=${7:-none} =="
  if boot "$2" "$3" "$4" "$5" "${6:-}"; then
    ask "$OUT/t5-rs-$1.json" "${7:-}"
    docker logs "$2" > "$OUT/t5-rs-$1.log" 2>&1
    python3 -c "
import json
d=json.load(open('$OUT/t5-rs-$1.json')); m=d['choices'][0]['message']
print('braccio ($1): ct=%d finish=%s content_len=%d reason_len=%d' % (d['usage']['completion_tokens'], d['choices'][0]['finish_reason'], len(m.get('content') or ''), len(m.get('reasoning_content') or '')))"
  else
    echo "braccio ($1): BOOT FAIL"; touch "$OUT/t5-rs-$1.FAIL"
  fi
  docker rm -f "$2" >/dev/null 2>&1
}
cmp_pair() { # $1,$2 lettere
  python3 - "$OUT" "$1" "$2" <<'EOF'
import json,sys
def msg(a):
    m=json.load(open(f'{sys.argv[1]}/t5-rs-{a}.json'))['choices'][0]['message']
    return (m.get('content') or ''), (m.get('reasoning_content') or '')
a,c=msg(sys.argv[2]); b,d=msg(sys.argv[3])
ci = a==b and c==d
print(f'({sys.argv[2]}) vs ({sys.argv[3]}): content_identico={a==b} reasoning_identico={c==d} -> {"CHAR-IDENTICAL" if ci else "DIVERGE"}')
EOF
}

arm a t5-a-mono6 "$IMG_NEW"  draft-mtp             6 ""      ""
arm b t5-b-duale  "$IMG_NEW"  draft-mtp,draft-dflash 7 "$DRAFTER" mtp
arm c t5-c-duale  "$IMG_NEW"  draft-mtp,draft-dflash 7 "$DRAFTER" dflash
arm d t5-d-df7    "$IMG_ORIG" draft-dflash          7 "$DRAFTER" ""
echo "== confronti =="
cmp_pair a b
cmp_pair c d
if ! cmp_pair c d >/dev/null 2>&1; then :; fi
# (e) condizionale: solo se (c)!=(d) — isolare l'effetto RS dal routing
if ! python3 - "$OUT" c d <<'EOF'
import json,sys
def msg(a):
    m=json.load(open(f'{sys.argv[1]}/t5-rs-{a}.json'))['choices'][0]['message']
    return (m.get('content') or ''), (m.get('reasoning_content') or '')
a,c=msg('c'); b,d=msg('d')
sys.exit(0 if (a==b and c==d) else 1)
EOF
then
  arm e t5-e-df7new "$IMG_NEW" draft-dflash 7 "$DRAFTER" ""
  cmp_pair c e
else
  echo "(e) non necessario: (c) == (d)"
fi
echo "=== T5 FINE (artefatti t5-rs-*.json in logs/test-drafter-routing/) ==="
