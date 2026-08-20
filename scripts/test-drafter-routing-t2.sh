#!/bin/bash
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# test-drafter-routing-t2.sh — T2 cache round-trip con drafter alternato (spec §7 T2, certifica R3.3)
# Uso: bash scripts/test-drafter-routing-t2.sh simple|prod
#
# EMENDAMENTO 2026-08-19 (stesse decisioni controller applicate a T1; NON indebolimenti):
#  1. --metrics SEMPRE (contratto osservabilita' spec §6).
#  2. Alternanza di classe via override body "spec_drafter" (template IDENTICO,
#     niente tools): e' ciò che isola l'effetto del ROUTING sulla cache (T1 C7).
#  3. Percorso RAM forzato in modo deterministico: --slot-prompt-similarity 0 +
#     micro-richiesta "flush" tra i turni. Motivazione (verificato in T1 run 3 e
#     nel codice, server-context.cpp): (a) l'idle-save della prompt-cache avviene
#     al lancio del task SUCCESSIVO, DOPO la lookup -> la sola sleep non pubblica
#     mai l'entry della conversazione; (b) con similarita' default 0.10 ogni turno
#     resta slot-local sul proprio slot (f_keep~1.0 -> nessun prompt_load) e la
#     verifica del TAG drafter (gate 2) non viene MAI eseguita - in T1 run 3 il
#     riuso c'e' stato (cached=210) con rebuild counters a 0. Con similarity 0 +
#     flush, ogni turno ricarica la conversazione dalla RAM (churn slot multi-
#     client deterministico): esercita save->load->tag-mismatch, la semantica che
#     il gate 2 certifica. I gate NON cambiano soglie ne' semantica.
#  4. Scenario informativo NON-gate in coda: turni 5-6 con alternanza VIA TOOLS
#     (piano originale) - documentiamo cosa accade (divergenza template ~lcp 40,
#     riuso parziale o nullo PER IL TEMPLATE, non per il routing).
#
# FUORI DAI GATE (dichiarato): percorso disk cache (spec §4.5) - non abilitato in
# produzione; copertura solo compile-time.
set -u
CONFIG=${1:-simple}
case "$CONFIG" in
  simple|prod) ;;
  *) echo "uso: $0 simple|prod"; exit 2 ;;
esac
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
OUT="${LAB_DIR}/logs/test-drafter-routing"
mkdir -p "$OUT"
PORT=8090
MODEL=/llmodels/QWEN3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf
DRAFTER=/llmodels/QWEN3.8/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
IMG=docker-llm-service:vulkan-fork-dflash2-route
NAME=t2-duale-$CONFIG
TOOLS='[{"type":"function","function":{"name":"get_time","description":"Ora corrente","parameters":{"type":"object","properties":{}}}}]'
EXTRA=""
if [ "$CONFIG" = prod ]; then EXTRA="--parallel 4 --kv-unified --cache-ram 65535"; fi
PASS=0; FAIL=0
ck() { # $1 nome-check $2 condizione(0=ok)
  if [ "$2" -eq 0 ]; then echo "PASS $1"; PASS=$((PASS+1)); else echo "FAIL $1"; FAIL=$((FAIL+1)); fi
}
BASE="http://127.0.0.1:$PORT/v1/chat/completions"
HIST="$OUT/t2-hist-$CONFIG.json"
LOG="$OUT/t2-$CONFIG.log"

docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --network host --device /dev/dri --group-add render \
  -v "${LLMODELS_DIR}/models:/llmodels:ro" --entrypoint llama-server "$IMG" \
  -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 --host 127.0.0.1 --port $PORT --metrics \
  --slot-prompt-similarity 0 $EXTRA \
  --spec-type draft-mtp,draft-dflash --spec-draft-model "$DRAFTER" \
  --spec-draft-ngl all --spec-draft-n-max 7 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
for i in $(seq 1 120); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200 && break; sleep 1; done

reqturn() { # $1 outfile $2 override(mtp|dflash|none) [$3=tools] -> echo http_code
  local OVR="" TOOLSF=""
  [ "$2" = dflash ] && OVR='"spec_drafter":"dflash",'
  [ "$2" = mtp ]    && OVR='"spec_drafter":"mtp",'
  [ "${3:-}" = tools ] && TOOLSF="\"tools\":$TOOLS,"
  local HIST_JSON; HIST_JSON=$(cat "$HIST")
  curl -s -o "$1" -w '%{http_code}' "$BASE" -H 'Content-Type: application/json' \
    -d "{$OVR$TOOLSF\"model\":\"q\",\"max_tokens\":100,\"temperature\":0,\"messages\":$HIST_JSON}"
}
hist_append() { # $1 response-file $2 prossimo user text
  python3 -c "
import json
hist=json.load(open('$HIST'))
h=json.load(open('$1'))['choices'][0]['message']
a={'role':'assistant','content':(h.get('content') or '')}
if h.get('reasoning_content'): a['reasoning_content']=h['reasoning_content']
hist.append(a); hist.append({'role':'user','content':'''$2'''})
json.dump(hist,open('$HIST','w'))
"
}
flush() { # $1 indice -> micro-richiesta di servizio (fa scattare l'idle-save RAM)
  curl -s -o "$OUT/t2-flush$1-$CONFIG.json" -w '%{http_code}' "$BASE" -H 'Content-Type: application/json' \
    -d "{\"model\":\"q\",\"max_tokens\":8,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Richiesta di servizio numero $1. Rispondi solo Punto.\"}]}"
}
usage3() { # $1 response-file -> "prompt_tokens cached completion"
  python3 -c "
import json
u=json.load(open('$1'))['usage']
print(u['prompt_tokens'], u.get('prompt_tokens_details',{}).get('cached_tokens',0), u['completion_tokens'])
"
}
sensible() { # $1 response-file -> 0 se content o reasoning_content non vuoti
  python3 -c "
import json,sys
m=json.load(open('$1'))['choices'][0]['message']
sys.exit(0 if ((m.get('content') or '').strip() or (m.get('reasoning_content') or '').strip()) else 1)
"
}

echo "== T2[$CONFIG] scenario primario: 4 turni, override alternato mtp->dflash->mtp->dflash =="
echo '[{"role":"user","content":"Ciao, chi era Giulio Cesare?"}]' > "$HIST"
CODES=""; PT_PREV=0; CT_PREV=0; G1=0
turn() { # $1 n-turno $2 override $3 prossimo-user-msg
  local C; C=$(reqturn "$OUT/t2-turn$1-$CONFIG.json" "$2"); CODES="$CODES $C"
  [ "$C" = 200 ] || return 1
  local PT CACHED CT; read -r PT CACHED CT < <(usage3 "$OUT/t2-turn$1-$CONFIG.json")
  local CONV=$((PT_PREV+CT_PREV)) DELTA=$((PT-CACHED))
  echo "T2[$CONFIG] turno $1 ($2): code=$C pt=$PT cached=$CACHED conv_prec=$CONV delta=$DELTA"
  if [ "$1" -ge 2 ]; then
    [ "$CACHED" -ge $((CONV*9/10)) ] && [ "$DELTA" -lt $((PT/2)) ] || G1=1
  fi
  sensible "$OUT/t2-turn$1-$CONFIG.json" || G1=1
  PT_PREV=$PT; CT_PREV=$CT
  hist_append "$OUT/t2-turn$1-$CONFIG.json" "$3"
}

turn 1 none   "Che ore sono adesso a Roma? Rispondi breve."
C=$(flush 1); CODES="$CODES $C"
turn 2 dflash "Dammi due fatti curiosi sulla antica Roma."
C=$(flush 2); CODES="$CODES $C"
turn 3 mtp    "Concludi con una frase di riepilogo in italiano."
C=$(flush 3); CODES="$CODES $C"
turn 4 dflash "Aggiungi un dettaglio finale."

# GATE 1: riuso target attraverso gli switch (turni 2-4)
ck "G1 riuso-target-switch (cached>=90% conv e delta<pt/2 per turni 2-4)" $G1
# GATE 4: tutte 200 (turni+flush) e contenuti sensati
NE=0; for c in $CODES; do [ "$c" = 200 ] || NE=$((NE+1)); done
ck "G4 response-200 (turni+flush; non-200=$NE)" $([ "$NE" -eq 0 ] && echo 0 || echo 1)

docker logs $NAME > "$LOG" 2>&1
# GATE 2: mismatch tag con target restored (attesi 3: dflash-prefix-miss, mtp-resync, dflash-prefix-miss)
N_MM=$(grep -c 'spec-route: cache tag mismatch.*target restored' "$LOG")
N_D=$(grep -c 'draft rebuild=dflash-prefix-miss' "$LOG")
N_M=$(grep -c 'draft rebuild=mtp-resync' "$LOG")
echo "T2[$CONFIG] mismatch: totali=$N_MM dflash-prefix-miss=$N_D mtp-resync=$N_M"
[ "$N_MM" -ge 3 ] && [ "$N_D" -ge 1 ] && [ "$N_M" -ge 1 ]; ck "G2 tag-mismatch+target-restored (3 switch: $N_MM tot, $N_D dfpm, $N_M mrs)" $?
# GATE 3: zero cold-full del target (pipeline senza -q al primo grep)
if grep -iE 'cold fallback' "$LOG" | grep -v 'spec-route' | grep -q .; then ck "G3 no-cold-target" 1; else ck "G3 no-cold-target" 0; fi

# SCENARIO INFORMATIVO (NON-gate): turni 5-6 alternanza VIA TOOLS (piano originale)
flush 4 >/dev/null
C5=$(reqturn "$OUT/t2-turn5-$CONFIG.json" none)
read -r PT5 C5C CT5 < <(usage3 "$OUT/t2-turn5-$CONFIG.json")
hist_append "$OUT/t2-turn5-$CONFIG.json" "Che ore sono? Usa lo strumento."
flush 5 >/dev/null
C6=$(reqturn "$OUT/t2-turn6-$CONFIG.json" none tools)
read -r PT6 C6C CT6 < <(usage3 "$OUT/t2-turn6-$CONFIG.json")
docker logs $NAME > "$OUT/t2-$CONFIG-post.log" 2>&1
echo "INFO (non-gate) via-tools: turno5 no-tools code=$C5 pt=$PT5 cached=$C5C (atteso riuso pieno, stesso template) | turno6 CON tools code=$C6 pt=$PT6 cached=$C6C (atteso riuso parziale/nullo per divergenza template)"
docker logs $NAME 2>&1 | grep -E 'spec-route: cache tag mismatch|lcp=' | tail -3

docker rm -f $NAME >/dev/null 2>&1
echo "=== T2[$CONFIG]: PASS=$PASS FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ]
