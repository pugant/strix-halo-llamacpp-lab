#!/bin/bash
# Part of strix-nebulosa — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# test-drafter-routing-t1.sh — T1 smoke dual-load drafter routing (spec §7 T1)
#
# AMENDMENT 2026-08-19 (post T1 review + controller decision; NOT weakenings,
# each gate only changes the measurement instrument or input, never a threshold
# or the semantics):
#  1. --metrics on ALL docker runs: gate C4c requires the /metrics endpoint
#     (infrastructure omission in the plan: without the flag the server answers 501).
#  2. C7 redesigned: the mtp/dflash class alternation happens via the body override
#     "spec_drafter" with an IDENTICAL template (no tools in the body) — that is
#     what isolates the ROUTING effect on the cache. The resend WITH tools diverges
#     in the template (tools block in the system prompt, lcp ~40, see the 19/08
#     checks) and cannot reuse the KV: it stays as an informative not-a-gate
#     check at the end (INFO).
#     The original C7 gate (prompt_tokens<200) measured the TOTAL prompt, not
#     the delta: replaced with cached_tokens >= 90% of the turn-1 conversation AND
#     processed delta < half of the turn-2 prompt (the "prefill delta ~0" semantics
#     of spec §7 T1, correctly measured).
#  3. C4c now naturally 3/3: the spec_route_cache_rebuild_total kinds are
#     pre-registered at 0 at boot (metrics commit 19/08).
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
ck() { # $1 check-name $2 condition(0=ok)
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
last_stat() { LOGS | grep -E "statistics +$1:" | tail -1; } # NB: the real line uses %16s padding (speculative.cpp:3389) — statistics print for ALL impls on every task: activity is read from the cumulative #gen drafts counters

# CHECK 1: boot dual mode
LOGS > "$OUT/t1.log"
grep -q 'spec-route: dual mode active: draft-mtp (nextn' "$OUT/t1.log"; ck "C1 dual-boot-line" $?

# warm-up + CHECK 2: policy default (no tools -> mtp)
req "$OUT/c2.json" '{"model":"q","max_tokens":50,"temperature":0,"messages":[{"role":"user","content":"Rispondi solo OK."}]}'
LOGS | grep 'spec-route: task' | tail -1 | grep -q 'signal=none → drafter=mtp'; ck "C2 policy-default-mtp" $?
last_stat draft-mtp   | grep -qE '#gen drafts = +[1-9]'; ck "C2b mtp-active (#gen drafts>0)" $?
last_stat draft-dflash | grep -qE '#gen drafts = +0';  ck "C2c dflash-inactive (#gen drafts=0 — single-active-impl invariant)" $?

# CHECK 3: tools -> dflash
req "$OUT/c3.json" "{\"model\":\"q\",\"max_tokens\":80,\"temperature\":0,\"tools\":$TOOLS,\"messages\":[{\"role\":\"user\",\"content\":\"Che ore sono? Usa lo strumento.\"}]}"
LOGS | grep 'spec-route: task' | tail -1 | grep -q 'signal=tools → drafter=dflash'; ck "C3 policy-tools-dflash" $?
last_stat draft-dflash | grep -qE '#gen drafts = +[1-9]'; ck "C3b dflash-active (#gen drafts>0)" $?

# CHECK 4: explicit override
req "$OUT/c4.json" '{"model":"q","max_tokens":50,"temperature":0,"spec_drafter":"dflash","messages":[{"role":"user","content":"Conta da 1 a 20."}]}'
LOGS | grep 'spec-route: task' | tail -1 | grep -q 'signal=override:dflash'; ck "C4 override-dflash" $?

# CHECK 4b: tool_choice branch (without a tools array) -> dflash
req "$OUT/c4b.json" '{"model":"q","max_tokens":50,"temperature":0,"tool_choice":"auto","messages":[{"role":"user","content":"Rispondi OK."}]}'
LOGS | grep 'spec-route: task' | tail -1 | grep -q 'drafter=dflash'; ck "C4b policy-tool_choice-dflash" $?

# CHECK 4c: Prometheus counters present (spec §6 — all 3)
curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/c4c.metrics"
grep -q 'spec_route_requests_total' "$OUT/c4c.metrics" && grep -q 'spec_route_cache_rebuild_total' "$OUT/c4c.metrics" && grep -q 'spec_route_override_total' "$OUT/c4c.metrics"; ck "C4c metrics-present (3/3)" $?

# CHECK 5: invalid enum -> 400
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

# CHECK 7 (AMENDED 19/08): conversation with a drafter switch via override
# (IDENTICAL template: no tools — routing is the only variable).
# NB: $NAME was removed in CHECK 6 -> recreate it with the FULL docker run
# (docker restart on a removed container fails and invalidates all the following checks)
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --network host --device /dev/dri --group-add render \
  -v "${LLMODELS_DIR}/models:/llmodels:ro" --entrypoint llama-server "$IMG" \
  -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 --host 127.0.0.1 --port $PORT --metrics \
  --spec-type draft-mtp,draft-dflash --spec-draft-model "$DRAFTER" \
  --spec-draft-ngl all --spec-draft-n-max 7 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
for i in $(seq 1 120); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200 && break; sleep 1; done
HIST='[{"role":"user","content":"Ciao, chi era Giulio Cesare?"}]'
req "$OUT/h1.json" "{\"model\":\"q\",\"max_tokens\":150,\"temperature\":0,\"messages\":$HIST}"
# resend content+reasoning_content VERBATIM (fix 0005 pattern: reasoning must be resent for the hit)
HIST2=$(python3 -c "
import json
h=json.load(open('$OUT/h1.json'))['choices'][0]['message']
a={'role':'assistant','content':(h.get('content') or '')}
if h.get('reasoning_content'): a['reasoning_content']=h['reasoning_content']
msgs=[{'role':'user','content':'Ciao, chi era Giulio Cesare?'}, a,
      {'role':'user','content':'Che ore sono adesso a Roma? Rispondi breve.'}]
print(json.dumps(msgs))")
# turn 2: SAME history, drafter switch via body override (no tools)
req "$OUT/h2.json" "{\"model\":\"q\",\"max_tokens\":150,\"temperature\":0,\"spec_drafter\":\"dflash\",\"messages\":$HIST2}"
read -r PT2 CACHED CONV < <(python3 -c "
import json
h1=json.load(open('$OUT/h1.json')); h2=json.load(open('$OUT/h2.json'))
u1=h1['usage']; u2=h2['usage']
print(u2.get('prompt_tokens',0), u2.get('prompt_tokens_details',{}).get('cached_tokens',0), u1.get('prompt_tokens',0)+u1.get('completion_tokens',0))")
DELTA=$((PT2-CACHED))
echo "C7 evidence: prompt_tokens=$PT2 cached=$CACHED conv_turn1=$CONV processed_delta=$DELTA"
LOGS | grep 'spec-route: task' | tail -1 | grep -q 'signal=override:dflash → drafter=dflash'; E1=$?
[ "$CACHED" -ge $((CONV*9/10)) ]; E2=$?   # reuse almost the whole conversation (>=90%)
[ "$DELTA" -lt $((PT2/2)) ]; E3=$?        # processed delta << half the prompt
[ "$E1" -eq 0 ] && [ "$E2" -eq 0 ] && [ "$E3" -eq 0 ]; ck "C7 cache-reuse-switch-drafter (cached=$CACHED/$CONV, delta=$DELTA)" $?
# no-cold gate FOR THE TARGET: real marker 'cold fallback:' (verified in the fork logs);
# pipeline WITHOUT -q on the first grep (-q suppresses output and makes the gate a no-op)
docker logs $NAME > "$OUT/t1-post.log" 2>&1
if grep -iE 'cold fallback' "$OUT/t1-post.log" | grep -v 'spec-route' | grep -q .; then ck "C7b no-cold-target" 1; else ck "C7b no-cold-target" 0; fi

# CHECK 7-INFO (not a gate, documentary — amendment 19/08): resend WITH tools.
# Low lcp expected (~40: the tools block in the system prompt diverges the
# template), so NO reuse: documents why C7 uses the override instead of tools.
HIST3="$HIST2"
req "$OUT/h3.json" "{\"model\":\"q\",\"max_tokens\":30,\"temperature\":0,\"tools\":$TOOLS,\"messages\":$HIST3}"
INFO_CACHED=$(python3 -c "import json;print(json.load(open('$OUT/h3.json')).get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens',-1))" 2>/dev/null || echo -1)
INFO_LCP=$(docker logs $NAME 2>&1 | grep -oE 'lcp=[0-9]+' | tail -1)
echo "INFO (not a gate) resend-with-tools: cached=$INFO_CACHED $INFO_LCP — ~zero reuse expected due to template divergence (expected lcp ~40)"

docker rm -f $NAME >/dev/null 2>&1
echo "=== T1: PASS=$PASS FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ]
