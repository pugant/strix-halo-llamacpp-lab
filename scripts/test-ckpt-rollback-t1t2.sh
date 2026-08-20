#!/usr/bin/env bash
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# T1/T2 del test-plan 2026-08-16-spec-cache-checkpoint-rollback-test.md (internal plan, not included in this repo)
# Container dedicato su porta 1235 — NON tocca llm-service.
# T1: smoke 2 turni → atteso "trailing rollback" (o hit esatto), ZERO cold fallback
# T2: correttezza greedy — stesso prompt deterministico con cache fredda vs calda → output identico
set -euo pipefail

IMAGE=${IMAGE:-docker-llm-service:vulkan-fork-ckpt}
PORT=${PORT:-1235}
LLMODELS_DIR="${LLMODELS_DIR:-$HOME/llmodels}"
MODEL=/llmodels/QWEN3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf
CTR=ckpt-rollback-test
LOG=/tmp/ckpt-rollback-test.log

cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

echo "== avvio server test ($IMAGE, porta $PORT) =="
docker run -d --name "$CTR" \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  -v "${LLMODELS_DIR}/models:/llmodels:ro" \
  "$IMAGE" llama-server \
  -m "$MODEL" -ngl 999 -fa on --jinja -c 16384 --host 127.0.0.1 --port "$PORT" \
  --parallel 1 --cont-batching \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --spec-type draft-mtp --spec-draft-ngl all --spec-draft-n-max 6 \
  --spec-draft-p-min 0.75 --spec-draft-p-split 0.10 \
  --spec-draft-type-k f16 --spec-draft-type-v f16 \
  -lv 3 > /dev/null

# attesa avvio (no sleep-loop stretto: poll ogni 5s max 5 min)
for i in $(seq 1 60); do
  if docker logs "$CTR" 2>&1 | grep -q 'server is listening'; then break; fi
  if ! docker inspect -f '{{.State.Running}}' "$CTR" 2>/dev/null | grep -q true; then
    echo "ERRORE: container morto"; docker logs "$CTR" 2>&1 | tail -20; exit 1
  fi
  sleep 5
done
docker logs "$CTR" 2>&1 | tail -3
IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$CTR")
[ -n "$IP" ] || IP=localhost
BASE="http://$IP:$PORT/v1/chat/completions"

chat() { curl -s "$BASE" -H 'Content-Type: application/json' -d "$1"; }

echo
echo "== T1: smoke 2 turni con prompt >64 token (checkpoint creato, pattern client senza reasoning) =="
: > "$LOG"
LONGP='Riassumi in dieci righe la storia dell Impero Romano d Occidente, dalla fondazione di Augusto fino alla caduta di Romolo Augustolo nel 476, includendo le cause della crisi del terzo secolo, le riforme di Diocleziano e Costantino, la divisione dell impero e le invasioni barbariche piu importanti.'
R1=$(chat "{\"messages\":[{\"role\":\"system\",\"content\":\"Sei un assistente conciso.\"},{\"role\":\"user\",\"content\":\"$LONGP\"}],\"max_tokens\":120,\"temperature\":0}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
echo "T1 turn1 (prime 2 righe): [$(echo "$R1" | head -2 | tr '\n' ' ')]"
R2=$(chat "$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"system","content":"Sei un assistente conciso."},{"role":"user","content":sys.argv[1]},{"role":"assistant","content":sys.argv[2]},{"role":"user","content":"Ora la stessa cosa per l Impero Romano d Oriente."}],"max_tokens":120,"temperature":0}))' "$LONGP" "$R1")" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
echo "T1 turn2 (prime 2 righe): [$(echo "$R2" | head -2 | tr '\n' ' ')]"
docker logs "$CTR" --since 10m 2>&1 | grep -E 'trailing rollback|cold fallback|spec-boundary|prompt cache' | tail -8

echo
echo "== T2: correttezza greedy — turno2 via riuso cache (rollback) vs turno2 cold =="
DET1='Scrivi i numeri da 1 a 25, uno per riga, solo i numeri.'
DET2='Ora scrivi le prime 10 lettere dell alfabeto, una per riga.'
# WARM: turno1, poi turno2 riusando la cache (exact o trailing rollback)
W1=$(chat "{\"messages\":[{\"role\":\"user\",\"content\":\"$DET1\"}],\"max_tokens\":200,\"temperature\":0}" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
REQ2=$(python3 -c 'import json,sys; print(json.dumps({"messages":[{"role":"user","content":sys.argv[1]},{"role":"assistant","content":sys.argv[2]},{"role":"user","content":sys.argv[3]}],"max_tokens":200,"temperature":0}))' "$DET1" "$W1" "$DET2")
W2=$(chat "$REQ2" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
# COLD: restart (cache RAM svuotata), stesso identico prompt finale processato da zero
docker restart "$CTR" >/dev/null
for i in $(seq 1 60); do docker logs "$CTR" --since 2m 2>&1 | grep -q 'server is listening' && break; sleep 5; done
C2=$(chat "$REQ2" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])')
if [ "$W2" = "$C2" ]; then echo "T2 PASS: turno 2 identico via-cache vs cold"; else echo "T2 FAIL — via-cache:"; echo "$W2" | head -5; echo "— cold:"; echo "$C2" | head -5; fi

echo
echo "== riepilogo eventi cache =="
docker logs "$CTR" 2>&1 | grep -cE 'cold fallback' || true
docker logs "$CTR" 2>&1 | grep -E 'trailing rollback' | tail -3
