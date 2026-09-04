#!/bin/bash
# Part of strix-nebulosa — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# bench-routing-vs-mono.sh — T4 A/B: dual routing (policy) vs MONO-MTP6 on Qwen3.8-27B STRIX_LEAN
# Plan T7-f2 Task 9. Derived from bench-dflash-vs-mtp.sh + bench-dflash-df3-agentic.sh (VERBATIM prompts).
# Standard protocol T7/T0: DEDICATED GPU (llm-service STOPPED, managed externally),
# -c 16384, EXPLICIT p_min 0.75, temp 0, warm-up discarded, TREATMENT marker, -lv 3.
# FAIRNESS: single image (vulkan-fork-dflash2-route) for both arms — ONLY the config changes.
#   MONO-MTP6: --spec-type draft-mtp --spec-draft-n-max 6
#   DUALE:     --spec-type draft-mtp,draft-dflash --spec-draft-n-max 7 (+ draft model)
# Prompts: 4 standard T7 (prose×2 WITHOUT tools; det×2 WITH minimal tools for routing) +
#         3 agentic VERBATIM from bench-dflash-df3-agentic.sh (WITH tools).
# Gates (class mean across prompts): R3.1 prose DUAL >= MONO-3%; R3.2 agentic DUAL >= MONO+10%
# (mean of the 3, not 2/3). Det = informational. Acceptance by impl = DELTA of the
# cumulative counters in the 'statistics' lines between consecutive prompts.
set -u
LAB_DIR="${LAB_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
OUT="${LAB_DIR}/logs/bench-routing-vs-mono"
mkdir -p "$OUT"
PORT=1235
MODEL=/llmodels/QWEN3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf
DRAFTER=/llmodels/QWEN3.8/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
IMG=docker-llm-service:vulkan-fork-dflash2-route

# minimal tools payload (T1): activates the 'tools' signal -> policy -> dflash in DUAL;
# present in BOTH arms (same body: fairness, mono ignores it for routing)
TOOLS='[{"type":"function","function":{"name":"get_time","description":"Ora corrente","parameters":{"type":"object","properties":{}}}}]'

# standard T7 prompts (VERBATIM from bench-dflash-vs-mtp.sh)
STD_P1='Scrivi un paragrafo dettagliato sulla storia di Roma.'
STD_P2='Scrivi un saggio breve sulla stampa e il Rinascimento italiano.'
STD_D1='Conta da 1 a 200, un numero per riga, solo i numeri.'
STD_D2='Elenco l alfabeto inglese una lettera per riga, poi ripetilo al contrario.'
# agentic prompts (VERBATIM from bench-dflash-df3-agentic.sh)
AG1='Scrivi dieci funzioni Python molto brevi, una per riga: somma, sottrazione, moltiplicazione, divisione, modulo, potenza, minimo, massimo, valore assoluto, arrotondamento. Ogni funzione su una riga con def e return.'
AG2='Genera un array JSON di trenta oggetti utente: id progressivo da 1, nome user_N, email user_N at example.com, attivo true, punteggio N virgola 5. Solo JSON valido, senza commenti.'
AG3='Scrivi venti righe di log in formato standard: data 2026-08-19, ora progressiva, livello INFO, servizio api, messaggio richiesta N elaborata con stato 200.'

start_server() { # $1=tag $2=type $3=nmax $4=drafter(or empty)
  local tag="$1" tipo="$2" nmax="$3" dft="${4:-}"
  local name="bench-${tag}-q38"
  local extra=()
  [ -n "$dft" ] && extra+=(--spec-draft-model "$dft")
  docker rm -f "$name" >/dev/null 2>&1
  docker run -d --name "$name" --network host \
    --device /dev/dri --group-add render \
    -v "${LLMODELS_DIR}/models:/llmodels:ro" \
    --entrypoint llama-server "$IMG" \
    -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 -lv 3 \
    --host 127.0.0.1 --port "$PORT" \
    --spec-type "$tipo" --spec-draft-ngl all --spec-draft-n-max "$nmax" \
    --spec-draft-p-min 0.75 --spec-draft-p-split 0.10 "${extra[@]}" >/dev/null || return 1
  for i in $(seq 1 120); do
    curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q 200 && return 0
    sleep 1
  done
  echo "FAIL health ${name}:"; docker logs "$name" 2>&1 | tail -20
  return 1
}

# $1=prompt $2=max_tokens $3=tools(1|0) [$4=outfile response] -> echo tok/s
bench_tok() {
  local TL=""
  [ "$3" = 1 ] && TL="\"tools\":$TOOLS,"
  local out="${4:-/dev/null}"
  curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"q\",${TL}\"max_tokens\":${2:-600},\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}" > "$out"
  python3 -c "import json; t=json.load(open('$out')).get('timings',{}); print(round(t.get('predicted_per_second',0),1))" 2>/dev/null || echo "ERR"
}

stat_after() { # $1=container $2=impl -> latest cumulative '#gen drafts' and '#acc tokens' values
  docker logs "$1" 2>&1 | grep -E "statistics +$2:" | tail -1 | sed -E 's/.*#gen drafts = +([0-9]+).*#acc tokens = +([0-9]+).*/\1 \2/'
}
route_after() { # $1=container -> latest routing marker (NB: '.*' absorbs the arrow unicode glyph)
  docker logs "$1" 2>&1 | grep 'spec-route: task' | tail -1 | sed -E 's/.*signal=([^ ]+).*drafter=([a-z]+).*/\1 \2/'
}

run_arm() { # $1=tag $2=type $3=nmax $4=drafter
  local tag="$1" name="bench-$1-q38" tsv="$OUT/arm-$1.tsv"
  echo "== TREATMENT=${tag} tipo=${2} n-max=${3} model=LEAN p_min=0.75 c=16384 vulkan img=route =="
  rm -f "$tsv"
  if ! start_server "$tag" "$2" "$3" "${4:-}"; then
    echo "RESULT TREATMENT=${tag}: SERVER FAIL"
    return 1
  fi
  bench_tok 'Rispondi solo OK.' 50 0 >/dev/null   # warm-up discarded (counter baseline)
  { local bm bd; read -r bm _ <<< "$(stat_after "$name" draft-mtp)"; read -r bd _ <<< "$(stat_after "$name" draft-dflash)"
    [ -z "$bm" ] && bm=-; [ -z "$bd" ] && bd=-
    echo -e "W\twarm\t0\t${bm}\t${bd}\t-\t-" >> "$tsv"; }
  # id class tools max_tokens prompt  (NB: fields separated by a SINGLE space)
  local rows=(
    "P1|prosa|0|600|$STD_P1"
    "P2|prosa|0|600|$STD_P2"
    "D1|det|1|600|$STD_D1"
    "D2|det|1|600|$STD_D2"
    "A1|agentic|1|700|$AG1"
    "A2|agentic|1|700|$AG2"
    "A3|agentic|1|600|$AG3"
  )
  for r in "${rows[@]}"; do
    local id cls tools mt prompt
    IFS='|' read -r id cls tools mt prompt <<< "$r"
    local tok; tok=$(bench_tok "$prompt" "$mt" "$tools" "$OUT/resp-${tag}-${id}.json")
    local mtp df rt sig dr
    read -r mtp _ <<< "$(stat_after "$name" draft-mtp)"
    read -r df _  <<< "$(stat_after "$name" draft-dflash)"
    read -r sig dr <<< "$(route_after "$name")"
    [ -z "$mtp" ] && mtp=-
    [ -z "$df" ] && df=-
    [ -z "$sig" ] && sig=-
    echo -e "${id}\t${cls}\t${tok}\t${mtp}\t${df}\t${sig}\t${dr}" >> "$tsv"
    echo "TREATMENT=${tag} ${id}(${cls}) ${tok} tok/s | stat mtp=${mtp} dflash=${df} | route ${sig} -> ${dr}"
  done
  docker logs "$name" > "$OUT/bench-${tag}.log" 2>&1
  grep -E 'statistics draft' "$OUT/bench-${tag}.log" | tail -2
  docker rm -f "$name" >/dev/null 2>&1
}

# arms (control first)
run_arm MONO-MTP6 draft-mtp 6 ""
run_arm DUALE     draft-mtp,draft-dflash 7 "$DRAFTER"

python3 - "$OUT" <<'EOF'
import sys
def ftok(s):
    try: return float(s)
    except ValueError: return 0.0
def load(tag):
    rows={}; prev_m=prev_d=0
    for line in open(f'{sys.argv[1]}/arm-{tag}.tsv'):
        pid,cls,tok,m,df,sig,dr=line.rstrip('\n').split('\t')
        m=int(m) if m!='-' else prev_m
        df=int(df) if df!='-' else prev_d
        if pid=='W':
            prev_m,prev_d=m,df; continue
        rows[pid]=dict(cls=cls,tok=ftok(tok),dm=m-prev_m,dd=df-prev_d,sig=sig,dr=dr)
        prev_m,prev_d=m,df
    return rows
mono=load('MONO-MTP6'); dual=load('DUALE')
print('== TABLE tok/s (prompt x arm) ==')
print(f"{'prompt':4} {'class':8} {'MONO-MTP6':>10} {'DUALE':>8} {'delta%':>8}  route(DUALE)")
for p in ['P1','P2','D1','D2','A1','A2','A3']:
    m,d=mono[p]['tok'],dual[p]['tok']
    dp=(d-m)/m*100 if m>0 else 0
    print(f"{p:4} {mono[p]['cls']:8} {m:10.1f} {d:8.1f} {dp:+8.1f}%  {dual[p]['sig']} -> {dual[p]['dr']}")
def amean(d,pids):
    v=[d[p]['tok'] for p in pids]; return sum(v)/len(v)
mp,dp_=amean(mono,['P1','P2']),amean(dual,['P1','P2'])
ma,da=amean(mono,['A1','A2','A3']),amean(dual,['A1','A2','A3'])
print(f"\nclass means: prose MONO={mp:.1f} DUALE={dp_:.1f} ({(dp_-mp)/mp*100:+.1f}%) | agentic MONO={ma:.1f} DUALE={da:.1f} ({(da-ma)/ma*100:+.1f}%)")
print('== acceptance by impl DUAL (delta of cumulative counters for each prompt; gen drafts) ==')
print(f"{'prompt':4} {'cls':8} {'mtp gen-dr':>10} {'df gen-dr':>10} {'df active?':>12}")
for p in ['P1','P2','D1','D2','A1','A2','A3']:
    print(f"{p:4} {dual[p]['cls']:8} {dual[p]['dm']:10d} {dual[p]['dd']:10d} {'NO(ok for prose)' if dual[p]['dd']==0 else 'YES'}")
g1 = dp_ >= mp*0.97
g2 = da >= ma*1.10
prosa_routed_ok = all(dual[p]['dr']=='mtp' for p in ['P1','P2'])
agentic_routed_ok = all(dual[p]['dr']=='dflash' for p in ['A1','A2','A3'])
print(f"\nGATE R3.1 prose DUAL>=MONO-3%: {'PASS' if g1 else 'FAIL'} ({dp_:.1f} vs {mp:.1f}, threshold {mp*0.97:.1f}) | prose routed to mtp: {'OK' if prosa_routed_ok else 'NO'}")
print(f"GATE R3.2 agentic DUAL>=MONO+10% (mean of 3): {'PASS' if g2 else 'FAIL'} ({da:.1f} vs {ma:.1f}, threshold {ma*1.10:.1f}) | agentic routed to dflash: {'OK' if agentic_routed_ok else 'NO'}")
print(f"DET (informational): D1 {mono['D1']['tok']:.1f}->{dual['D1']['tok']:.1f} D2 {mono['D2']['tok']:.1f}->{dual['D2']['tok']:.1f}")
print(f"=== T4: R3.1={'PASS' if g1 else 'FAIL'} R3.2={'PASS' if g2 else 'FAIL'} ===")
EOF
echo "=== END (TREATMENT marker on every line) ==="
