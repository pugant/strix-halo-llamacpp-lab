#!/bin/bash
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# test-drafter-routing-t3.sh — T3 regressione percorsi sacri (0005-0009) in duale (spec §7 T3)
# Uso: bash scripts/test-drafter-routing-t3.sh simple|prod
#
# EMENDAMENTI 2026-08-19 (stessa natura di T1/T2; NON indebolimenti — motivazione in header):
#  1. Misura riuso via cached=/delta (usage.prompt_tokens_details + riga prompt-eval del
#     log), MAI usage.prompt_tokens totale (T1: e' il totale, non il delta).
#  2. Le varianti "resend CON tools" del piano diventano echo INFO non-gate (la divergenza
#     template del blocco tools, lcp~40, renderebbe l'exact-hit impossibile: certificherebbe
#     il template, non il routing). La versione GATE usa override spec_drafter.
#  3. SIMILARITA' DEFAULT (0.10, NIENTE tecnica T2): i percorsi sacri (trailing rollback,
#     checkpoint salvage 0007) vivono nel path slot-local di update_slots; con similarity 0
#     + flush i resend divergenti prendono la via load()-RAM che SCARTA le entry divergenti
#     (spec_trailing_rm=false) e pagano cold — non e' il percorso da certificare qui.
#  4. Vincolo tag-checkpoint (by design, spec §4.4 + server-context.cpp): il salvage usa il
#     checkpoint piu' recente <=p0_rm e restaura i byte draft solo a tag drafter compatibile
#     con la classe del task; => S2/S4 allineano la classe dei resend divergenti alla classe
#     del turno creatore del checkpoint utile. Il caso divergente CROSS-CLASS e' echo INFO
#     in coda (cold by design, limite dichiarato). S1 usa turni che chiudono con stop
#     (max_tokens 400) per aver round-trip esatti deterministici (delta 0) con switch.
#  5. Prompt-turno-1 lunghi (~470 tok): evitano che un nuovo scenario catturi per similarita'
#     lo slot di una conversazione precedente (sim<0.10) — i cold delle conversazioni NUOVE
#     resterebbero altrimenti loggati come 'cold fallback' inquinando i gate zero-cold.
#  6. Fix pre-esistente scoperto da T3 (commit reasoning-budget-clone): il budget non
#     si esauriva MAI su config con dflash (clone del sampler riempiva il contatore a ogni
#     rollback speculativo). S1 dipende da quel fix (marker 'budget exhausted').
#
# Scenario (fonti: piani 15/08 forced-end, 16/08 checkpoint-rollback + t3-cold-fallback):
#  S1 budget-forced end + resend verbatim, 2 varianti (mtp->dflash, dflash->mtp):
#     gate: marker 'budget exhausted, forcing end sequence' + exact-hit (cached>=90% conv,
#     delta<pt/2) + zero cold non-spec-route.
#  S2 resend alterato (reasoning turn-1 mozzato presto => delta oltre ring => salvage):
#     conv mtp,dflash; resend mtp. gate: 200, zero abort, zero cold, marker
#     'prompt cache checkpoint rollback' (rigenera SOLO dal checkpoint).
#  S3 trailing truncation + resend (ultima parola mutata, delta piccolo), 2 varianti
#     (mtp->dflash, dflash->mtp): gate: marker 'trailing rollback: lcp=' + zero abort + zero cold.
#  S4 checkpoint salvage 0007: conv mtp/dflash/mtp-tronco(max_tokens 60, length) + resume
#     prefisso puro (re-ask turno 3) classe mtp: gate: marker 'prompt cache checkpoint
#     rollback' + zero abort + prefill risparmiato >50% (processed < pt/2 dal log) + 200.
#  INFO non-gate: resend CON tools dopo S1 e S3 (documenta divergenza template).
set -u
CONFIG=${1:-simple}
case "$CONFIG" in
  simple|prod) ;;
  *) echo "uso: $0 simple|prod"; exit 2 ;;
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
seg() { docker logs $NAME 2>&1 | tail -n +$((NLOG+1)); } # log dalla fine dello scenario precedente
cursor() { NLOG=$(docker logs $NAME 2>&1 | wc -l); }
longmsg() { # $1 topic -> prompt lungo ~470 token (nuova conversazione: sim<0.10 con slot vecchi)
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
proc_tokens() { # $1 seg-file -> token processati dall'ultima riga prompt-eval
  grep -oE 'prompt eval time *= *[0-9.]+ ms */ *[0-9]+ tokens' "$1" | tail -1 | grep -oE '[0-9]+ tokens' | grep -oE '[0-9]+' || echo 0
}
zero_cold() { # $1 seg-file -> 0 se zero cold fallback non-spec-route
  if grep -iE 'cold fallback' "$1" | grep -v 'spec-route' | grep -q .; then echo 1; else echo 0; fi
}

# ============ S1: budget-forced end + resend verbatim (2 varianti) ============
s1() { # $1 variante(a|mtp->dflash | b|dflash->mtp) $2 classe-turno1 $3 classe-resend $4 topic
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

# ============ S2: resend alterato (reasoning turn-1 mozzato) ============
# NB tag-checkpoint (by design §4.4): il resend divergente recupera SOLO via
# checkpoint-salvage, che restaura i byte draft solo a tag compatibile => i turni
# del flusso GATE restano classe mtp (stessa del turno-1, creatore del checkpoint
# utile). Il caso cross-class divergente e' echo INFO in coda (cold by design).
H2="$OUT/s2-$CONFIG.hist.json"
echo "[{\"role\":\"user\",\"content\":\"$(longmsg "le termiche dell atmosfera")\"}]" > "$H2"
cursor
R=$(reqturn "$OUT/s2-t1-$CONFIG.json" "$H2" mtp)
hist_append "$OUT/s2-t1-$CONFIG.json" "$H2" "Ok. Ora dammi un esempio pratico breve."
R2=$(reqturn "$OUT/s2-t2-$CONFIG.json" "$H2" mtp)
# resend ALTERATO: reasoning del turno-1 mozzato presto (delta oltre ring) + nuovo turno, classe mtp
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
ck "S2 resend-alterato (no abort, no cold, rigenera dal checkpoint: processed=$PROC3/$PT3)" $G

# ============ S3: trailing truncation + resend (gate ORIGINALE ripristinato, ciclo RS 20/08) ============
# [CICLO RS 2026-08-20, piano 2026-08-20-rs-rollback-dflash-experiment] il commit
# 'spec: keep recurrent state rollback with draft-dflash' mantiene n_rs_seq>0 anche
# col duale => il seq_rm del blocco trailing ora riesce e il marker originale
# 'trailing rollback: lcp=' torna raggiungibile. L'emendamento S3' del 19/08
# (gate degradato a esito-documentato) e' OBSOLETO e rimosso: gate originale.
s3() { # $1 variante $2 classe-turno1 $3 classe-resend $4 topic
  local V=$1 C1=$2 CR=$3 H="$OUT/s3-$1-$CONFIG.hist.json"
  echo "[{\"role\":\"user\",\"content\":\"$(longmsg "$4")\"}]" > "$H"
  cursor
  R=$(reqturn "$OUT/s3-$V-t1-$CONFIG.json" "$H" "$C1")
  # append PRIMA della mutazione (h[1] deve esistere): h = [u1,a1,u2]
  hist_append "$OUT/s3-$V-t1-$CONFIG.json" "$H" "Confermi? Una parola."
  # resend con ULTIMA PAROLA del content mutata (delta piccolo, dentro ring)
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

# ============ S4: checkpoint salvage (conv con switch + turno troncato + resume prefisso puro) ============
# Classi: t1=mtp, t2=mtp, t3=dflash TRONCATO, resume=dflash. Lo switch resta (t2->t3) e
# ogni resend con delta<=2 (trim whitespace, topic-luck) cade su checkpoint a tag
# COMPATIBILE (t2 su ckpt t1 mtp; resume su ckpt prefill t3 dflash). Il resume
# certifica il salvage su un checkpoint DFLASH-tagged (complementare a S2/mtp-tag).
H4="$OUT/s4-$CONFIG.hist.json"
echo "[{\"role\":\"user\",\"content\":\"$(longmsg "le rotte commerciali medievali")\"}]" > "$H4"
cursor
R=$(reqturn "$OUT/s4-t1-$CONFIG.json" "$H4" mtp)                                # turno 1 mtp
hist_append "$OUT/s4-t1-$CONFIG.json" "$H4" "Dammi un secondo esempio."
R2=$(reqturn "$OUT/s4-t2-$CONFIG.json" "$H4" mtp)                               # turno 2 mtp
hist_append "$OUT/s4-t2-$CONFIG.json" "$H4" "Un terzo esempio, breve."
R3=$(reqturn "$OUT/s4-t3-$CONFIG.json" "$H4" dflash 60)                         # turno 3 dflash TRONCATO (switch + length)
# resume: il client scarta l'output parziale e re-invia la request del turno 3 VERBATIM
# (H4 e' ancora [u1,a1,u2,a2,u3]: l'assistant troncato non e' mai stato appeso) — classe dflash (= ckpt t3)
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

# ============ INFO non-gate: resend CON tools (divergenza template) ============
cursor
HI="$OUT/s1-a-$CONFIG.hist.json"
RI=$(reqturn "$OUT/info-tools1-$CONFIG.json" "$HI" none tools)
read -r PTI CI _ < <(usage3 "$OUT/info-tools1-$CONFIG.json")
seg > "$OUT/info-$CONFIG.seg.log"
echo "INFO (non-gate) S1-conv resend CON tools: code=$RI pt=$PTI cached=$CI $(grep -oE 'lcp=[0-9]+' "$OUT/info-$CONFIG.seg.log" | tail -1) — atteso riuso nullo/parziale per divergenza template (lcp atteso ~40)"

# ============ INFO non-gate: resend alterato CROSS-CLASS (limite by-design §4.4) ============
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
echo "INFO (non-gate) resend alterato CROSS-CLASS (mtp->dflash): code=$RX — atteso cold by design (§4.4: checkpoint salvage a tag incompatibile non restaura i byte draft)"

# liveness finale + abort globali
docker logs $NAME > "$OUT/t3-$CONFIG-full.log" 2>&1
LIVE=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health)
NABORT=$(grep -c 'GGML_ABORT' "$OUT/t3-$CONFIG-full.log")
NCOLD=$(grep -iE 'cold fallback' "$OUT/t3-$CONFIG-full.log" | grep -cv 'spec-route')
echo "INFO finale: health=$LIVE GGML_ABORT(totali)=$NABORT cold-fallback(non-spec-route,totali)=$NCOLD"
[ "$LIVE" = 200 ] && [ "$NABORT" -eq 0 ]; ck "G-final server vivo + zero abort" $?

docker rm -f $NAME >/dev/null 2>&1
echo "=== T3[$CONFIG]: PASS=$PASS FAIL=$FAIL ==="
[ "$FAIL" -eq 0 ]
