#!/bin/bash
# Part of strix-nebulosa — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# test-drafter-routing-t3.sh — T3 regression of the sacred paths (0005-0009) in dual mode (spec §7 T3)
# Usage: bash scripts/test-drafter-routing-t3.sh simple|prod
#
# AMENDMENTS 2026-08-19 (same nature as T1/T2; NOT weakenings — rationale in the header):
#  1. Measure reuse via cached=/delta (usage.prompt_tokens_details + the log
#     prompt-eval line), NEVER total usage.prompt_tokens (T1: it is the total, not the delta).
#  2. The plan's "resend WITH tools" variants become not-a-gate INFO echoes (the template
#     divergence of the tools block, lcp~40, would make the exact-hit impossible: it would
#     certify the template, not the routing). The GATE version uses the spec_drafter override.
#  3. DEFAULT SIMILARITY (0.10, NO T2 technique): the sacred paths (trailing rollback,
#     checkpoint salvage 0007) live in the slot-local path of update_slots; with similarity 0
#     + flush the divergent resends take the load()-RAM route which DISCARDS divergent
#     entries (spec_trailing_rm=false) and pays cold — that is not the path to certify here.
#  4. Tag-checkpoint constraint (by design, spec §4.4 + server-context.cpp): the salvage uses
#     the most recent checkpoint <=p0_rm and restores the draft bytes only at a drafter tag
#     compatible with the task class; => S2/S4 align the class of the divergent resends to
#     the class of the turn that created the useful checkpoint. The divergent CROSS-CLASS
#     case is an INFO echo at the end (cold by design, declared limitation). S1 uses turns
#     that end with stop (max_tokens 400) to get deterministic exact round-trips (delta 0)
#     with a switch.
#  5. Long turn-1 prompts (~470 tok): prevent a new scenario from capturing by similarity
#     the slot of a previous conversation (sim<0.10) — otherwise the colds of NEW
#     conversations would be logged as 'cold fallback', polluting the zero-cold gates.
#  6. Pre-existing fix discovered by T3 (commit reasoning-budget-clone): the budget
#     never exhausted on configs with dflash (the sampler clone refilled the counter
#     on every speculative rollback). S1 depends on that fix (marker 'budget exhausted').
#
# Scenario (sources: 15/08 forced-end plan, 16/08 checkpoint-rollback plan + t3-cold-fallback):
#  S1 budget-forced end + verbatim resend, 2 variants (mtp->dflash, dflash->mtp):
#     gate: marker 'budget exhausted, forcing end sequence' + exact-hit (cached>=90% conv,
#     delta<pt/2) + zero cold outside spec-route.
#  S2 altered resend (turn-1 reasoning truncated early => delta beyond ring => salvage):
#     conv mtp,dflash; resend mtp. gate: 200, zero abort, zero cold, marker
#     'prompt cache checkpoint rollback' (regenerates ONLY from the checkpoint).
#  S3 trailing truncation + resend (last word changed, small delta), 2 variants
#     (mtp->dflash, dflash->mtp): gate: marker 'trailing rollback: lcp=' + zero abort + zero cold.
#  S4 checkpoint salvage 0007: conv mtp/dflash/mtp-truncated(max_tokens 60, length) +
#     pure-prefix resume (re-ask turn 3) class mtp: gate: marker 'prompt cache checkpoint
#     rollback' + zero abort + prefill saved >50% (processed < pt/2 from the log) + 200.
#  INFO not a gate: resend WITH tools after S1 and S3 (documents template divergence).
set -u
CONFIG=${1:-simple}
case "$CONFIG" in
  simple|prod) ;;
  *) echo "usage: $0 simple|prod"; exit 2 ;;
esac
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
OUT="${LAB_DIR}/logs/test-drafter-routing/t3${VTAG:-}"
mkdir -p "$OUT"
PORT=8090
MODEL=/llmodels/QWEN3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf
DRAFTER=/llmodels/QWEN3.8/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
IMG=docker-llm-service:vulkan-fork-dflash2-route
NAME=t3-duale-$CONFIG
TOOLS='[{"type":"function","function":{"name":"get_time","description":"Ora corrente","parameters":{"type":"object","properties":{}}}}]'
EXTRA=""
if [ "$CONFIG" = prod ]; then EXTRA="--parallel 4 --kv-unified --cache-ram 65535"; fi
PASS=0; FAIL=0
ck() { if [ "$2" -eq 0 ]; then echo "PASS $1"; PASS=$((PASS+1)); else echo "FAIL $1"; FAIL=$((FAIL+1)); fi; }
BASE="http://127.0.0.1:$PORT/v1/chat/completions"

docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --network host --device /dev/dri --group-add render \
  -v "${LLMODELS_DIR}/models:/llmodels:ro" --entrypoint llama-server "$IMG" \
  -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 --host 127.0.0.1 --port $PORT --metrics \
  --reasoning-budget 128 $EXTRA \
  --spec-type draft-mtp,draft-dflash --spec-draft-model "$DRAFTER" \
  --spec-draft-ngl all --spec-draft-n-max 7 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10
for i in $(seq 1 120); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null | grep -q 200 && break; sleep 1; done

NLOG=0
seg() { docker logs $NAME 2>&1 | tail -n +$((NLOG+1)); } # log from the end of the previous scenario
cursor() { NLOG=$(docker logs $NAME 2>&1 | wc -l); }
longmsg() { # $1 topic -> long ~470-token prompt (new conversation: sim<0.10 vs old slots)
  local M="Contesto lungo per la domanda finale. "
  for i in $(seq 1 30); do M="$M Nota numero $i su $1: il punto e' rilevante per la risposta finale. "; done
  M="$M Alla fine, dopo aver riflettuto, rispondi in una frase breve."
  echo "$M"
}
reqturn() { # $1 outfile $2 hist-file $3 override(mtp|dflash|none) [$4=tools] [$5 max_tokens]
  local OVR="" TOOLSF=""; local MT=${5:-400}
  [ "$3" = dflash ] && OVR='"spec_drafter":"dflash",'
  [ "$3" = mtp ]    && OVR='"spec_drafter":"mtp",'
  [ "${4:-}" = tools ] && TOOLSF="\"tools\":$TOOLS,"
  local H; H=$(cat "$2")
  curl -s -o "$1" -w '%{http_code}' "$BASE" -H 'Content-Type: application/json' \
    -d "{$OVR$TOOLSF\"model\":\"q\",\"max_tokens\":$MT,\"temperature\":0,\"messages\":$H}"
}
usage3() { python3 -c "
import json
u=json.load(open('$1'))['usage']
print(u['prompt_tokens'], u.get('prompt_tokens_details',{}).get('cached_tokens',0), u['completion_tokens'])
"; }
hist_append() { # $1 resp-file $2 hist-file $3 next-user
  python3 -c "
import json
hist=json.load(open('$2'))
h=json.load(open('$1'))['choices'][0]['message']
a={'role':'assistant','content':(h.get('content') or '')}
if h.get('reasoning_content'): a['reasoning_content']=h['reasoning_content']
hist.append(a); hist.append({'role':'user','content':'''$3'''})
json.dump(hist,open('$2','w'))
"
}
proc_tokens() { # $1 seg-file -> tokens processed, from the last prompt-eval line
  grep -oE 'prompt eval time *= *[0-9.]+ ms */ *[0-9]+ tokens' "$1" | tail -1 | grep -oE '[0-9]+ tokens' | grep -oE '[0-9]+' || echo 0
}
zero_cold() { # $1 seg-file -> 0 if zero cold fallback outside spec-route
  if grep -iE 'cold fallback' "$1" | grep -v 'spec-route' | grep -q .; then echo 1; else echo 0; fi
}

# ============ S1: budget-forced end + verbatim resend (2 variants) ============
s1() { # $1 variant(a|mtp->dflash | b|dflash->mtp) $2 turn1-class $3 resend-class $4 topic
  local V=$1 C1=$2 CR=$3 H="$OUT/s1-$1-$CONFIG.hist.json" R
  echo "[{\"role\":\"user\",\"content\":\"$(longmsg "$4")\"}]" > "$H"
  cursor
  R=$(reqturn "$OUT/s1-$V-t1-$CONFIG.json" "$H" "$C1")
  hist_append "$OUT/s1-$V-t1-$CONFIG.json" "$H" "Rispondi ora in una parola."
  R2=$(reqturn "$OUT/s1-$V-t2-$CONFIG.json" "$H" "$CR")
  seg > "$OUT/s1-$V-$CONFIG.seg.log"; cursor
  local PT1 CT1 PT2 CACHED DELTA; read -r PT1 _ CT1 < <(usage3 "$OUT/s1-$V-t1-$CONFIG.json")
  read -r PT2 CACHED _ < <(usage3 "$OUT/s1-$V-t2-$CONFIG.json")
  DELTA=$((PT2-CACHED)); local CONV=$((PT1+CT1))
  local G=0
  grep -q 'budget exhausted, forcing end sequence' "$OUT/s1-$V-$CONFIG.seg.log" || G=1
  [ "$R" = 200 ] && [ "$R2" = 200 ] || G=1
  [ "$CACHED" -ge $((CONV*9/10)) ] && [ "$DELTA" -lt $((PT2/2)) ] || G=1
  [ "$(zero_cold "$OUT/s1-$V-$CONFIG.seg.log")" = 0 ] || G=1
  echo "S1-$V: t1($C1) code=$R t2($CR) code=$R2 | pt1=$PT1 ct1=$CT1 conv=$CONV pt2=$PT2 cached=$CACHED delta=$DELTA"
  ck "S1-$V forced-end+exact-hit ($C1->$CR)" $G
}
s1 a mtp    dflash "Giulio Cesare e la Roma antica"
s1 b dflash mtp    "la fotosintesi delle piante"

# ============ S2: altered resend (turn-1 reasoning truncated) ============
# NB tag-checkpoint (by design §4.4): the divergent resend recovers ONLY via
# checkpoint-salvage, which restores the draft bytes only at a compatible tag => the
# turns of the GATE flow stay class mtp (same as turn-1, creator of the useful
# checkpoint). The divergent cross-class case is an INFO echo at the end (cold by design).
H2="$OUT/s2-$CONFIG.hist.json"
echo "[{\"role\":\"user\",\"content\":\"$(longmsg "le termiche dell atmosfera")\"}]" > "$H2"
cursor
R=$(reqturn "$OUT/s2-t1-$CONFIG.json" "$H2" mtp)
hist_append "$OUT/s2-t1-$CONFIG.json" "$H2" "Ok. Ora dammi un esempio pratico breve."
R2=$(reqturn "$OUT/s2-t2-$CONFIG.json" "$H2" mtp)
# ALTERED resend: turn-1 reasoning truncated early (delta beyond ring) + new turn, class mtp
python3 - "$H2" <<'EOF'
import json,sys
h=json.load(open(sys.argv[1]))
r=h[1].get('reasoning_content') or ''
w=r.split()[:8]
h[1]['reasoning_content']=' '.join(w)+' [reasoning abbreviato dal client]'
json.dump(h,open(sys.argv[1],'w'))
EOF
hist_append "$OUT/s2-t2-$CONFIG.json" "$H2" "Perfetto. Prosegui con un dettaglio in piu."
R3=$(reqturn "$OUT/s2-t3-$CONFIG.json" "$H2" mtp)
seg > "$OUT/s2-$CONFIG.seg.log"; cursor
read -r PT3 CACHED3 _ < <(usage3 "$OUT/s2-t3-$CONFIG.json")
PROC3=$(proc_tokens "$OUT/s2-$CONFIG.seg.log")
echo "S2: codes=$R/$R2/$R3 | t3 pt=$PT3 cached=$CACHED3 processed=$PROC3"
G=0; [ "$R" = 200 ] && [ "$R2" = 200 ] && [ "$R3" = 200 ] || G=1
grep -q 'prompt cache checkpoint rollback' "$OUT/s2-$CONFIG.seg.log" || G=1
[ "$(zero_cold "$OUT/s2-$CONFIG.seg.log")" = 0 ] || G=1
grep -q 'GGML_ABORT' "$OUT/s2-$CONFIG.seg.log" && G=1
ck "S2 altered-resend (no abort, no cold, regenerates from checkpoint: processed=$PROC3/$PT3)" $G

# ============ S3: trailing truncation + resend (ORIGINAL gate restored, RS cycle 20/08) ============
# [RS CYCLE 2026-08-20, plan 2026-08-20-rs-rollback-dflash-experiment] the commit
# 'spec: keep recurrent state rollback with draft-dflash' keeps n_rs_seq>0 even in
# dual mode => the seq_rm of the trailing block now succeeds and the original
# marker 'trailing rollback: lcp=' is reachable again. The 19/08 S3' amendment
# (gate degraded to documented outcome) is OBSOLETE and removed: original gate.
s3() { # $1 variant $2 turn1-class $3 resend-class $4 topic
  local V=$1 C1=$2 CR=$3 H="$OUT/s3-$1-$CONFIG.hist.json"
  echo "[{\"role\":\"user\",\"content\":\"$(longmsg "$4")\"}]" > "$H"
  cursor
  R=$(reqturn "$OUT/s3-$V-t1-$CONFIG.json" "$H" "$C1")
  # append BEFORE the mutation (h[1] must exist): h = [u1,a1,u2]
  hist_append "$OUT/s3-$V-t1-$CONFIG.json" "$H" "Confermi? Una parola."
  # resend with the LAST WORD of the content changed (small delta, within ring)
  python3 - "$H" <<'EOF'
import json,sys
h=json.load(open(sys.argv[1]))
c=(h[1].get('content') or h[1].get('reasoning_content') or 'fine').strip()
w=c.split(); w[-1]='milano' if w[-1].lower()!='milano' else 'torino'
if h[1].get('content'): h[1]['content']=' '.join(w)
else: h[1]['reasoning_content']=' '.join(w)
json.dump(h,open(sys.argv[1],'w'))
EOF
  R2=$(reqturn "$OUT/s3-$V-t2-$CONFIG.json" "$H" "$CR")
  seg > "$OUT/s3-$V-$CONFIG.seg.log"; cursor
  local G=0
  [ "$R" = 200 ] && [ "$R2" = 200 ] || G=1
  grep -q 'trailing rollback: lcp=' "$OUT/s3-$V-$CONFIG.seg.log" || G=1
  [ "$(zero_cold "$OUT/s3-$V-$CONFIG.seg.log")" = 0 ] || G=1
  grep -q 'GGML_ABORT' "$OUT/s3-$V-$CONFIG.seg.log" && G=1
  grep 'trailing rollback: lcp=' "$OUT/s3-$V-$CONFIG.seg.log" | tail -1
  ck "S3-$V trailing-rollback ($C1->$CR)" $G
}
s3 a mtp    dflash "le piramidi d Egitto"
s3 b dflash mtp    "il ciclo dell acqua"

# ============ S4: checkpoint salvage (conv with switch + truncated turn + pure-prefix resume) ============
# Classes: t1=mtp, t2=mtp, t3=dflash TRUNCATED, resume=dflash. The switch stays (t2->t3)
# and every resend with delta<=2 (whitespace trim, topic luck) lands on a checkpoint with
# a COMPATIBLE tag (t2 on the t1 mtp ckpt; resume on the t3 dflash prefill ckpt). The
# resume certifies the salvage on a DFLASH-tagged checkpoint (complementary to S2/mtp-tag).
H4="$OUT/s4-$CONFIG.hist.json"
echo "[{\"role\":\"user\",\"content\":\"$(longmsg "le rotte commerciali medievali")\"}]" > "$H4"
cursor
R=$(reqturn "$OUT/s4-t1-$CONFIG.json" "$H4" mtp)                                # turn 1 mtp
hist_append "$OUT/s4-t1-$CONFIG.json" "$H4" "Dammi un secondo esempio."
R2=$(reqturn "$OUT/s4-t2-$CONFIG.json" "$H4" mtp)                               # turn 2 mtp
hist_append "$OUT/s4-t2-$CONFIG.json" "$H4" "Un terzo esempio, breve."
R3=$(reqturn "$OUT/s4-t3-$CONFIG.json" "$H4" dflash 60)                         # turn 3 dflash TRUNCATED (switch + length)
# resume: the client discards the partial output and resends the turn-3 request VERBATIM
# (H4 is still [u1,a1,u2,a2,u3]: the truncated assistant was never appended) — class dflash (= ckpt t3)
R4=$(reqturn "$OUT/s4-t4-$CONFIG.json" "$H4" dflash)
seg > "$OUT/s4-$CONFIG.seg.log"; cursor
read -r PT4 CACHED4 _ < <(usage3 "$OUT/s4-t4-$CONFIG.json")
PROC4=$(proc_tokens "$OUT/s4-$CONFIG.seg.log")
echo "S4: codes=$R/$R2/$R3/$R4 | resume pt=$PT4 cached=$CACHED4 processed=$PROC4"
G=0; [ "$R" = 200 ] && [ "$R2" = 200 ] && [ "$R3" = 200 ] && [ "$R4" = 200 ] || G=1
grep -q 'prompt cache checkpoint rollback' "$OUT/s4-$CONFIG.seg.log" || G=1
grep -q 'GGML_ABORT' "$OUT/s4-$CONFIG.seg.log" && G=1
[ "$(zero_cold "$OUT/s4-$CONFIG.seg.log")" = 0 ] || G=1
[ "$PROC4" -gt 0 ] && [ $((PROC4*2)) -lt "$PT4" ] || G=1
grep 'prompt cache checkpoint rollback' "$OUT/s4-$CONFIG.seg.log" | tail -1
ck "S4 checkpoint-salvage (marker, no abort, no cold, saved>50%: $PROC4/$PT4)" $G

# ============ INFO not a gate: resend WITH tools (template divergence) ============
cursor
HI="$OUT/s1-a-$CONFIG.hist.json"
RI=$(reqturn "$OUT/info-tools1-$CONFIG.json" "$HI" none tools)
read -r PTI CI _ < <(usage3 "$OUT/info-tools1-$CONFIG.json")
seg > "$OUT/info-$CONFIG.seg.log"
echo "INFO (not a gate) S1-conv resend WITH tools: code=$RI pt=$PTI cached=$CI $(grep -oE 'lcp=[0-9]+' "$OUT/info-$CONFIG.seg.log" | tail -1) — null/partial reuse expected due to template divergence (expected lcp ~40)"

# ============ INFO not a gate: altered resend CROSS-CLASS (by-design limit §4.4) ============
H2X="$OUT/info-xclass-$CONFIG.hist.json"
echo "[{\"role\":\"user\",\"content\":\"$(longmsg "i vulcani islandesi")\"}]" > "$H2X"
R=$(reqturn "$OUT/info-x1-$CONFIG.json" "$H2X" mtp)
hist_append "$OUT/info-x1-$CONFIG.json" "$H2X" "Dammi un dettaglio in piu."
python3 - "$H2X" <<'EOF'
import json,sys
h=json.load(open(sys.argv[1]))
r=h[1].get('reasoning_content') or ''
h[1]['reasoning_content']=' '.join(r.split()[:8])+' [abbreviato]'
json.dump(h,open(sys.argv[1],'w'))
EOF
hist_append "$OUT/info-x1-$CONFIG.json" "$H2X" "Prosegui."
RX=$(reqturn "$OUT/info-x2-$CONFIG.json" "$H2X" dflash)
seg > "$OUT/info-xclass-$CONFIG.seg.log"
echo "INFO (not a gate) altered resend CROSS-CLASS (mtp->dflash): code=$RX — cold expected by design (§4.4: checkpoint salvage at an incompatible tag does not restore the draft bytes)"

# final liveness + global aborts
docker logs $NAME > "$OUT/t3-$CONFIG-full.log" 2>&1
LIVE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)
NABORT=$(grep -c 'GGML_ABORT' "$OUT/t3-$CONFIG-full.log")
NCOLD=$(grep -iE 'cold fallback' "$OUT/t3-$CONFIG-full.log" | grep -cv 'spec-route')
echo "INFO final: health=$LIVE GGML_ABORT(total)=$NABORT cold-fallback(outside-spec-route,total)=$NCOLD"
[ "$LIVE" = 200 ] && [ "$NABORT" -eq 0 ]; ck "G-final server alive + zero aborts" $?

docker rm -f $NAME >/dev/null 2>&1
echo "=== T3[$CONFIG]: PASS=$PASS FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ]
