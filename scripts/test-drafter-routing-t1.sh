#!/bin/bash
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# test-drafter-routing-t1.sh — T1 smoke dual-load drafter routing (spec §7 T1)
#
# EMENDAMENTO 2026-08-19 (post review T1 + decisione controller; NON indebolimenti,
# ogni gate cambia solo strumento di misura o input, mai soglia di semantica):
#  1. --metrics su TUTTI i docker run: il gate C4c richiede l'endpoint /metrics
#     (omissione infrastrutturale del piano: senza il flag il server risponde 501).
#  2. C7 ridisegnato: l'alternanza di classe mtp/dflash avviene via override body
#     "spec_drafter" con template IDENTICO (niente tools nel body) — è ciò che
#     isola l'effetto del ROUTING sulla cache. Il resend CON tools diverge nel
#     template (blocco tools nel system, lcp ~40, vedi controlli 19/08) e non
#     può riusare il KV: resta come check informativo NON-gate in coda (INFO).
#     Gate C7 originale (prompt_tokens<200) misurava il TOTALE del prompt, non
#     il delta: sostituito con cached_tokens >= 90% della conversazione turn-1 E
#     delta processato < meta' del prompt turn-2 (semantica "prefill delta ~0"
#     della spec §7 T1, misura corretta).
#  3. C4c ora 3/3 naturale: i kind di spec_route_cache_rebuild_total sono
#     pre-registrati a 0 al boot (commit metrics 19/08).
set -u
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
OUT="${LAB_DIR}/logs/test-drafter-routing"
mkdir -p "$OUT"
PORT=8090
MODEL=/llmodels/QWEN3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf
DRAFTER=/llmodels/QWEN3.8/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
IMG=docker-llm-service:vulkan-fork-dflash2-route
NAME=t1-dual-route
TOOLS='[{"type":"function","function":{"name":"get_time","description":"Ora corrente","parameters":{"type":"object","properties":{}}}}]'
PASS=0; FAIL=0
ck() { # $1 nome-check $2 condizione(0=ok)
  if [ "$2" -eq 0 ]; then echo "PASS $1"; PASS=$((PASS+1)); else echo "FAIL $1"; FAIL=$((FAIL+1)); fi
}

docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --network host --device /dev/dri --group-add render \
  -v "${LLMODELS_DIR}/models:/llmodels:ro" --entrypoint llama-server "$IMG" \
  -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 --host 127.0.0.1 --port $PORT --metrics \
  --spec-type draft-mtp,draft-dflash --spec-draft-model "$DRAFTER" \
  --spec-draft-ngl all --spec-draft-n-max 7 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
for i in $(seq 1 120); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200 && break; sleep 1; done

req() { # $1 outfile $2 json
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$2" > "$1"
}
LOGS() { docker logs $NAME 2>&1; }
last_stat() { LOGS | grep -E "statistics +$1:" | tail -1; } # NB: riga reale con padding %16s (speculative.cpp:3389) — le statistics stampano per TUTTI gli impl a ogni task: l'attività si legge dai contatori cumulativi #gen drafts

# CHECK 1: boot dual mode
LOGS > "$OUT/t1.log"
grep -q 'spec-route: dual mode active: draft-mtp (nextn' "$OUT/t1.log"; ck "C1 dual-boot-line" $?

# warm-up + CHECK 2: policy default (no tools -> mtp)
req "$OUT/c2.json" '{"model":"q","max_tokens":50,"temperature":0,"messages":[{"role":"user","content":"Rispondi solo OK."}]}'
LOGS | grep 'spec-route: task' | tail -1 | grep -q 'signal=none → drafter=mtp'; ck "C2 policy-default-mtp" $?
last_stat draft-mtp   | grep -qE '#gen drafts = +[1-9]'; ck "C2b mtp-attivo (#gen drafts>0)" $?
last_stat draft-dflash | grep -qE '#gen drafts = +0';  ck "C2c dflash-inattivo (#gen drafts=0 — invariante un-impl-attivo)" $?

# CHECK 3: tools -> dflash
req "$OUT/c3.json" "{\"model\":\"q\",\"max_tokens\":80,\"temperature\":0,\"tools\":$TOOLS,\"messages\":[{\"role\":\"user\",\"content\":\"Che ore sono? Usa lo strumento.\"}]}"
LOGS | grep 'spec-route: task' | tail -1 | grep -q 'signal=tools → drafter=dflash'; ck "C3 policy-tools-dflash" $?
last_stat draft-dflash | grep -qE '#gen drafts = +[1-9]'; ck "C3b dflash-attivo (#gen drafts>0)" $?

# CHECK 4: override esplicito
req "$OUT/c4.json" '{"model":"q","max_tokens":50,"temperature":0,"spec_drafter":"dflash","messages":[{"role":"user","content":"Conta da 1 a 20."}]}'
LOGS | grep 'spec-route: task' | tail -1 | grep -q 'signal=override:dflash'; ck "C4 override-dflash" $?

# CHECK 4b: ramo tool_choice (senza tools array) -> dflash
req "$OUT/c4b.json" '{"model":"q","max_tokens":50,"temperature":0,"tool_choice":"auto","messages":[{"role":"user","content":"Rispondi OK."}]}'
LOGS | grep 'spec-route: task' | tail -1 | grep -q 'drafter=dflash'; ck "C4b policy-tool_choice-dflash" $?

# CHECK 4c: contatori Prometheus presenti (spec §6 — tutti e 3)
curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/c4c.metrics"
grep -q 'spec_route_requests_total' "$OUT/c4c.metrics" && grep -q 'spec_route_cache_rebuild_total' "$OUT/c4c.metrics" && grep -q 'spec_route_override_total' "$OUT/c4c.metrics"; ck "C4c metrics-presenti (3/3)" $?

# CHECK 5: enum invalido -> 400
code=$(curl -s -o "$OUT/c5.json" -w '%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"q","max_tokens":10,"spec_drafter":"bogus","messages":[{"role":"user","content":"x"}]}')
[ "$code" = "400" ] && grep -q 'must be one of' "$OUT/c5.json"; ck "C5 bogus-400" $?

# CHECK 6: boot fallback mono + drafter-not-loaded 400
docker rm -f $NAME >/dev/null 2>&1; NAME2=t1-mono-fallback
docker run -d --name $NAME2 --network host --device /dev/dri --group-add render \
  -v "${LLMODELS_DIR}/models:/llmodels:ro" --entrypoint llama-server "$IMG" \
  -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 --host 127.0.0.1 --port $PORT --metrics \
  --spec-type draft-mtp,draft-dflash --spec-draft-model /llmodels/NONEXISTENT.gguf \
  --spec-draft-ngl all --spec-draft-n-max 7 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
for i in $(seq 1 120); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200 && break; sleep 1; done
docker logs $NAME2 > "$OUT/t1-mono.log" 2>&1
grep -qi 'dropping draft-dflash' "$OUT/t1-mono.log"; ck "C6a boot-fallback-warning" $?
code=$(curl -s -o "$OUT/c6.json" -w '%{http_code}' "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"model":"q","max_tokens":10,"spec_drafter":"dflash","messages":[{"role":"user","content":"x"}]}')
[ "$code" = "400" ] && grep -q 'not loaded' "$OUT/c6.json"; ck "C6b not-loaded-400" $?
docker rm -f $NAME2 >/dev/null 2>&1

# CHECK 7 (EMENDATO 19/08): conversazione con switch drafter via override
# (template IDENTICO: niente tools — il routing è l'unica variabile).
# NB: $NAME è stato rimosso nel CHECK 6 -> ricrearlo con docker run COMPLETO
# (docker restart su container rimosso fallisce e invalida tutti i check seguenti)
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --network host --device /dev/dri --group-add render \
  -v "${LLMODELS_DIR}/models:/llmodels:ro" --entrypoint llama-server "$IMG" \
  -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 --host 127.0.0.1 --port $PORT --metrics \
  --spec-type draft-mtp,draft-dflash --spec-draft-model "$DRAFTER" \
  --spec-draft-ngl all --spec-draft-n-max 7 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
for i in $(seq 1 120); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200 && break; sleep 1; done
HIST='[{"role":"user","content":"Ciao, chi era Giulio Cesare?"}]'
req "$OUT/h1.json" "{\"model\":\"q\",\"max_tokens\":150,\"temperature\":0,\"messages\":$HIST}"
# resend VERBATIM content+reasoning_content (pattern fix 0005: il reasoning va rimandato per l'hit)
HIST2=$(python3 -c "
import json
h=json.load(open('$OUT/h1.json'))['choices'][0]['message']
a={'role':'assistant','content':(h.get('content') or '')}
if h.get('reasoning_content'): a['reasoning_content']=h['reasoning_content']
msgs=[{'role':'user','content':'Ciao, chi era Giulio Cesare?'}, a,
      {'role':'user','content':'Che ore sono adesso a Roma? Rispondi breve.'}]
print(json.dumps(msgs))")
# turn 2: STESSA history, switch drafter via override body (niente tools)
req "$OUT/h2.json" "{\"model\":\"q\",\"max_tokens\":150,\"temperature\":0,\"spec_drafter\":\"dflash\",\"messages\":$HIST2}"
read -r PT2 CACHED CONV < <(python3 -c "
import json
h1=json.load(open('$OUT/h1.json')); h2=json.load(open('$OUT/h2.json'))
u1=h1['usage']; u2=h2['usage']
print(u2.get('prompt_tokens',0), u2.get('prompt_tokens_details',{}).get('cached_tokens',0), u1.get('prompt_tokens',0)+u1.get('completion_tokens',0))")
DELTA=$((PT2-CACHED))
echo "C7 evidenza: prompt_tokens=$PT2 cached=$CACHED conv_turn1=$CONV delta_processato=$DELTA"
LOGS | grep 'spec-route: task' | tail -1 | grep -q 'signal=override:dflash → drafter=dflash'; E1=$?
[ "$CACHED" -ge $((CONV*9/10)) ]; E2=$?   # riuso quasi-tutta la conversazione (>=90%)
[ "$DELTA" -lt $((PT2/2)) ]; E3=$?        # delta processato << metà del prompt
[ "$E1" -eq 0 ] && [ "$E2" -eq 0 ] && [ "$E3" -eq 0 ]; ck "C7 cache-reuse-switch-drafter (cached=$CACHED/$CONV, delta=$DELTA)" $?
# gate no-cold DEL TARGET: marker reale 'cold fallback:' (verificato nei log fork);
# pipeline SENZA -q al primo grep (il -q sopprime l'output e rende il gate un no-op)
docker logs $NAME > "$OUT/t1-post.log" 2>&1
if grep -iE 'cold fallback' "$OUT/t1-post.log" | grep -v 'spec-route' | grep -q .; then ck "C7b no-cold-target" 1; else ck "C7b no-cold-target" 0; fi

# CHECK 7-INFO (NON-gate, documentativo — emendamento 19/08): resend CON tools.
# Atteso lcp basso (~40: il blocco tools nel system fa divergere il template),
# quindi NESSUN riuso: documenta perché C7 usa l'override e non i tools.
HIST3="$HIST2"
req "$OUT/h3.json" "{\"model\":\"q\",\"max_tokens\":30,\"temperature\":0,\"tools\":$TOOLS,\"messages\":$HIST3}"
INFO_CACHED=$(python3 -c "import json;print(json.load(open('$OUT/h3.json')).get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens',-1))" 2>/dev/null || echo -1)
INFO_LCP=$(docker logs $NAME 2>&1 | grep -oE 'lcp=[0-9]+' | tail -1)
echo "INFO (non-gate) resend-con-tools: cached=$INFO_CACHED $INFO_LCP — atteso riuso ~nullo per divergenza template (lcp atteso ~40)"

docker rm -f $NAME >/dev/null 2>&1
echo "=== T1: PASS=$PASS FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ]
