# Spec — grug-35b-v2 → ROCmFP4-STRIX_LEAN per Strix Halo

**Data:** 2026-08-11
**Status:** Draft (in attesa di adversarial review)
**Target hardware:** AMD Ryzen AI MAX+ 395 (Strix Halo, gfx1151, 124 GB unified LPDDR5X-8000 ~270 GB/s)
**Autore:** sessione Claude Code con pugant

---

## 1. Contesto e motivazione

### 1.1 Modello sorgente
`ProCreations/grug-35b-v2` è un fine-tune LoRA di `Ornith-1.0-35B`, a sua volta fine-tune di **Qwen3.5-VL-MoE** (`model_type: qwen3_5_moe`).

Architettura (da `config.json`):
- MoE: **256 expert, 8 attivi + 1 shared → ~3B params attivi/token** (paragonabile a Qwen3.6-35B-A3B)
- 40 layer totali, ibrido **30 linear_attention (Gated DeltaNet) + 10 full_attention** (pattern `[L,L,L,F] × 10`)
- hidden_size 2048, num_attention_heads 16, num_key_value_heads 2, head_dim 256
- vocab 248320, max_position_embeddings 262144, partial_rotary_factor 0.25, full_attention_interval 4
- Vision tower Qwen3.5-VL (`qwen3_5_moe_vision`, depth 27, hidden 1152, patch_size 16)
- **MTP disabilitato**: `mtp_num_hidden_layers: 0`
- License Apache-2.0
- Sorgente: BF16 safetensors, 18 shard, ~70 GB totali

Caratteristiche distintive del modello: token-efficient reasoning (`<think>` in "grug-speak"), tag "agentic" / "tool-use" / "conversational". Riduzione token del 97% su MATH-500 vs base, a parità di accuratezza.

### 1.2 Perché STRIX_LEAN e non Q4_K_M
Il preset `Q4_0_ROCMFP4_STRIX_LEAN` (type 106, ~4.38 bpw) è il target di riferimento per Strix Halo:
- ~4.38 bit per peso → pesi ~4× più piccoli del BF16 → ~4× meno lettura memoria per token → tok/s proporzionale su UMA LPDDR5X (~270 GB/s, memory-bound)
- Protegge **attention K/V + Q5_K token embeddings** (mantiene qualità dove serve)
- FP4 è **software puro** su gfx1151 (RDNA 3.5 non ha unità FP4 in silicio); il vantaggio è solo di banda memoria

Baseline attuale in produzione: `Qwen3.6-35B-A3B-Q4_0_ROCMFP4_STRIX_LEAN.gguf` (19 GB, **63 tok/s plain senza MTP**, su `docker-llm-service` porta :1234).

### 1.3 Paragone corretto
grug-35b-v2 è un MoE 3B-attivi (come 35B-A3B), **non** un denso (come Qwen3.6-27B). Quindi:
- La pipeline target è **plain, senza MTP** (allineata a 35B-A3B produzione)
- Ceiling LPDDR5X: 3B attivi × 0.5 byte (FP4) = 1.5 GB/token → ~180 tok/s teorico
- Plain speed attesa: ~45-65 tok/s
- La mancanza di MTP in grug **non è un problema** — ci allinea alla baseline MoE

### 1.4 Vincoli del container ROCmFPX (verificati in adversarial review)
Il container `docker-llm-service` (Fedora 43 minimale) NON include i deps Python per la conversione HF→GGUF:
- `transformers`, `torch`, `safetensors`, `sentencepiece`: **assenti**
- `pip` / `python3 -m pip`: **assenti**
- Il converter è installato come binario: `/usr/local/bin/convert_hf_to_gguf.py` (NON in `/opt/llama.cpp/examples/`)

Per la Fase 3 (conversione) serve quindi un'immagine derivata con deps Python. Le altre fasi (imatrix, quantize, bench) usano il container base perché non richiedono Python deps.

---

## 2. Obiettivi e non-obiettivi

### Obiettivi
1. Produrre `Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf` (~20 GB) ottimizzato per Strix Halo
2. Produrre `mmproj-grug-35b-v2-f16.gguf` per abilitare la modalità vision (opzionale a runtime)
3. Generare imatrix calibrata sul dominio del modello (grug-think traces)
4. Confrontare A/B vs `ProCreations/grug-35b-v2-Q4_K_M.gguf` (baseline ufficiale)
5. Documentare la pipeline in script shell riutilizzabile
6. Verificare empiricamente che l'arch qwen3_5_moe + Gated DeltaNet funzioni nel container ROCmFPX attuale

### Non-obiettivi
- Aggiornare/rebase del fork ROCmFPX (out of scope)
- Attivare MTP a runtime (modello non ne ha, e allineamento a pipeline MoE plain)
- Fine-tuning, ri-addestramento, applicazione LoRA ex-post
- Multi-GPU / distribuzione (single-node Strix Halo)
- Validazione qualità sistematica (MMLU/SWE-bench) — solo smoke test funzionale

---

## 3. Architettura pipeline

Pipeline a 9 fasi (0-7, con 4a e 4b sottofasi della fase 4) che replica `quantize-qwen36-27b.sh` come template, con tre adattamenti:
- **A1**: parte da safetensors (non GGUF BF16 già pronto) → fase di conversione HF→GGUF (richiede immagine derivata `docker-llm-service-convert`)
- **A2**: imatrix non pre-calcolata → generazione on-prem con dataset di calibrazione custom
- **A3**: niente MTP → nessuna fase di gestione draft layer

```
[Build img] → [Setup+precheck] → [Download 70GB] → [Convert HF→GGUF] → [Prep calibration] → [Gen imatrix] → [Quantize ROCmFP4] → [Bench A/B] → [Cleanup]
    0              1                   2                  3                   4a                  4b                 5               6           7
```

Finestra dedicata: `llm-service` viene fermato durante le fasi 4b-6 (libera RAM e GPU; le fasi 0-4a sono sul disco/CPU e non disturbano il service in produzione). Riavviato a fine fase 6.

---

## 4. Fasi dettagliate

> **Convenzione globale**: tutti i blocchi shell presuppongono `HF_TOKEN` exportato a inizio script (vedi Fase 1 precheck). Se si copia-incolla un blocco in una shell separata, rieseguire `export HF_TOKEN="$(cat ~/.cache/huggingface/token)"`.

### Fase 0 — Build immagine `docker-llm-service-convert` (~10 min, una tantum)
Il container base non ha i deps Python per la conversione HF→GGUF (vedi §1.4). Crea un'immagine derivata riutilizzabile (vale anche per Ornith FASE 2):

```bash
mkdir -p <lab-repo>/docker
cat > <lab-repo>/docker/Dockerfile.convert <<'EOF'
# Derivata da docker-llm-service: aggiunge deps Python per convert_hf_to_gguf.py
FROM docker-llm-service:latest

# Fedora 43 base — dnf per python3-pip, poi pip per i package ML.
# NOTA: --index-url SOSTITUISCE PyPI default (pytorch.org ospita solo torch/torchvision/torchaudio),
# quindi servono DUE pip install separati: torch da pytorch.org, il resto da PyPI.
RUN dnf -y --nodocs --setopt=install_weak_deps=False install python3-pip \
  && dnf clean all && rm -rf /var/cache/dnf/* \
  && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
  && pip install --no-cache-dir transformers==5.14.1 safetensors sentencepiece accelerate
EOF

docker build -t docker-llm-service-convert:latest \
  -f <lab-repo>/docker/Dockerfile.convert \
  <lab-repo>/docker/
```

Size attesa immagine derivata: ~12 GB (base 10.1 + torch CPU ~800 MB + transformers/safetensors/sentencepiece/accelerate + deps ~1 GB). Una tantum, riutilizzabile per Ornith FASE 2.

Verifica:
```bash
docker run --rm docker-llm-service-convert python3 -c \
  "import transformers, torch, safetensors, sentencepiece; print('OK')"
# atteso: OK
docker run --rm docker-llm-service-convert which convert_hf_to_gguf.py
# atteso: /usr/local/bin/convert_hf_to_gguf.py
```

### Fase 1 — Setup + precheck
- Verifica disco libero ≥ 180 GB (picco pipeline)
- Verifica `docker-llm-service` UP
- Verifica immagine `docker-llm-service-convert` presente (se no, vedi Fase 0)
- Verifica HF raggiungibile
- Crea workspace: `~/llmodels/models/GRUG/35B-BF16/`
- Verifica che `~/llmodels/models/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf` non esista già
- Esporta `HF_TOKEN`: `export HF_TOKEN="$(cat ~/.cache/huggingface/token)"`

### Fase 2 — Download BF16 safetensors (~70 GB)
Usare `wget -c` con Bearer (NON `hf download` con hf_xet, che si blocca su file > 1 GB su questa rete — problema già noto). Il repo è pubblico e non gated, ma l'header Bearer è mantenuto per coerenza con il pattern collaudato e per banda autenticata:

```bash
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
REPO="https://huggingface.co/ProCreations/grug-35b-v2/resolve/main"
DLDIR=~/llmodels/models/GRUG/35B-BF16
mkdir -p "$DLDIR"

# 18 shard safetensors (~70 GB totali)
for i in $(seq -w 1 18); do
  wget -c --header "Authorization: Bearer $HF_TOKEN" --tries=20 --timeout=60 --retry-connrefused \
    --progress=dot:giga \
    "${REPO}/model-000${i}-of-00018.safetensors" \
    -O "${DLDIR}/model-000${i}-of-00018.safetensors"
done

# File accessori (~25 MB)
for f in config.json tokenizer.json tokenizer_config.json chat_template.jinja \
         model.safetensors.index.json preprocessor_config.json generation_config.json; do
  wget -c --header "Authorization: Bearer $HF_TOKEN" "${REPO}/${f}" -O "${DLDIR}/${f}"
done
```

Verifica:
- `stat -c%s "${DLDIR}"/model-*.safetensors | awk '{s+=$1} END {printf "%.1f GB\n", s/1024/1024/1024}'`
- Atteso: ~65.4 GiB (~70 GB)
- Verifica integrità: 18 shard + 7 file accessori, niente file `.safetensors.tmp` lasciati da wget interrotti

### Fase 3 — Conversione HF→GGUF BF16
Usa l'immagine derivata `docker-llm-service-convert` (vedi Fase 0). Il path del converter nel container è `/usr/local/bin/convert_hf_to_gguf.py`:

```bash
docker run --rm \
  -v ~/llmodels/models:/llmodels:rw \
  docker-llm-service-convert python3 /usr/local/bin/convert_hf_to_gguf.py \
    /llmodels/GRUG/35B-BF16 \
    --outfile /llmodels/GRUG/grug-35b-v2-BF16.gguf \
    --outtype bf16
```

Output atteso (verificato in adversarial review — il converter riconosce `Qwen3_5MoeForConditionalGeneration` come `MmprojModel`, quindi produce entrambi i file):
- `grug-35b-v2-BF16.gguf` (~70 GB)
- `mmproj-grug-35b-v2-f16.gguf` (~857 MB) estratto automaticamente dalla vision tower

Fallback se il converter NON producesse mmproj: scaricare dal repo GGUF ufficiale:
```bash
wget -c --header "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/ProCreations/grug-35b-v2-gguf/resolve/main/mmproj-grug-35b-v2-f16.gguf" \
  -O ~/llmodels/models/GRUG/mmproj-grug-35b-v2-f16.gguf
```

Verifica `general.architecture` del GGUF prodotto = `qwen35moe` (atteso; arch del fork):
```bash
docker run --rm -v ~/llmodels/models:/llmodels:ro docker-llm-service \
  llama-gguf --verbose /llmodels/GRUG/grug-35b-v2-BF16.gguf 2>&1 | grep "general.architecture"
# atteso: general.architecture = qwen35moe
```

### Fase 4a — Preparazione calibration text (su HOST)
Lo script gira sull'HOST Python 3.12 (ha già `datasets 5.0.0` + `jinja2 3.1.2`). NON usa `transformers` (non serve): applica il chat template con `jinja2.Template` direttamente.

```python
# <lab-repo>/scripts/prep-grug-calibration.py
# Richiede (su HOST): datasets, jinja2  (già presenti)
# Uso: python3 prep-grug-calibration.py
from datasets import load_dataset
from jinja2 import Environment, BaseLoader
from jinja2.exceptions import TemplateError
import os, random, sys

OUT_DIR = os.path.expanduser("~/llmodels/calibration")
TEMPLATE_PATH = os.path.expanduser("~/llmodels/models/GRUG/35B-BF16/chat_template.jinja")
OUT_PATH = os.path.join(OUT_DIR, "grug-calibration.txt")
N_SAMPLE = 800  # target ~4 MB plain text (best practice imatrix: 1-5 MB)

os.makedirs(OUT_DIR, exist_ok=True)
ds = load_dataset("ProCreations/grug-think-v3-10k", split="train")
print(f"[i] Dataset rows: {len(ds)}", file=sys.stderr)

with open(TEMPLATE_PATH) as f:
    chat_template_src = f.read()
# NOTA: trim_blocks/lstrip_blocks lasciati ai default jinja2 (False) per allinearsi
# al behavior di transformers.apply_chat_template. Per imatrix è ininfluente.
template = Environment(loader=BaseLoader()).from_string(chat_template_src)

random.seed(42)
sampled_idx = random.sample(range(len(ds)), min(N_SAMPLE, len(ds)))


def fallback_concat(row):
    """R9 fallback semplice: concatena role:content per ogni message senza template.
    Per imatrix è sufficiente (l'obiettivo è attivare i pesi, non fedeltà chat)."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in row["messages"])


rendered_lines, skipped = [], 0
for i in sampled_idx:
    row = dict(ds[i])  # robusto: ritorna dict standard anche se la lib cambia row type
    messages = row["messages"]
    tools = row.get("tools", [])
    try:
        rendered = template.render(
            messages=messages,
            add_generation_prompt=False,
            tools=tools,
        )
        rendered_lines.append(rendered)
    except TemplateError:
        # R9 fallback: se jinja2 puro non riesce a renderizzare, usa la versione semplice
        rendered_lines.append(fallback_concat(row))
        skipped += 1
        continue

# R9 safety: se TUTTI i render falliscono, alert esplicito (non scrivere file vuoto)
if not rendered_lines:
    sys.exit("[✗] Tutti i render sono falliti — calibration text vuoto. Vedi R9.")

print(f"[i] Rendered OK: {len(rendered_lines)}, di cui fallback: {skipped}", file=sys.stderr)

with open(OUT_PATH, "w") as f:
    f.write("\n\n".join(rendered_lines))

size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
print(f"[✓] Calibration text: {OUT_PATH} ({size_mb:.2f} MB)", file=sys.stderr)
```

Esecuzione su host:
```bash
mkdir -p <lab-repo>/scripts
# (crea lo script qui sopra con l'editor)
python3 <lab-repo>/scripts/prep-grug-calibration.py
```

Size target: ~3-5 MB plain text.

### Fase 4b — Generazione imatrix (nel container base, CPU-only)
Stop `llm-service` prima di questa fase:

```bash
docker compose -f <your-workspace>/workspace/docker/docker-compose.yml stop llm-service

docker run --rm \
  -v ~/llmodels/models:/llmodels:rw \
  -v ~/llmodels/calibration:/calibration:ro \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-imatrix \
    -m /llmodels/GRUG/grug-35b-v2-BF16.gguf \
    -f /calibration/grug-calibration.txt \
    -o /llmodels/GRUG/imatrix-grug-35b-v2.gguf \
    --output-frequency 10 --save-frequency 0 \
    --threads 16 --chunks 256 \
    --no-ppl --parse-special \
    -ngl 0
```

- `-ngl 0`: CPU-only, pesi in system RAM (LPDDR5X). La generazione imatrix è memory-bound su RAM, GPU non aiuta
- `--threads 16`: scelta di compromesso (a 32 si satura banda RAM/LPDDR5X; lezione da `quantize-qwen36-27b.sh`)
- `--chunks 256`: buona copertura del dataset
- `--no-ppl`: salta il calcolo della perplexity (overhead inutile per pura calibration)
- `--parse-special`: tokenizza correttamente i tag speciali del chat template (`<think>`, `<|im_start|>`, ecc.) — critico per dataset reasoning
- Output: `imatrix-grug-35b-v2.gguf` (~150-300 MB)

Tempo atteso: 20-40 min.

### Fase 5 — Quantizzazione ROCmFP4-STRIX_LEAN (nel container base, CPU-only)
`llm-service` resta fermato (già stoppato in Fase 4b).

Dry-run prima (verifica size attesa senza eseguire, ~80ms):
```bash
docker run --rm \
  -v ~/llmodels/models:/llmodels:ro \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-quantize --dry-run \
  /llmodels/GRUG/grug-35b-v2-BF16.gguf Q4_0_ROCMFP4_STRIX_LEAN 2>&1 | tail -10
```

Quantizzazione vera (mount `:rw` perché scrive l'output):
```bash
docker run --rm \
  -v ~/llmodels/models:/llmodels:rw \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-quantize \
    --imatrix /llmodels/GRUG/imatrix-grug-35b-v2.gguf \
    /llmodels/GRUG/grug-35b-v2-BF16.gguf \
    /llmodels/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
    Q4_0_ROCMFP4_STRIX_LEAN 16
```

- `16` è l'ultimo argomento posizionale = `nthreads` (NON un flag) — 32 saturano banda UMA LPDDR5X (lezione Qwen3.6-27B e 35B-A3B)
- Output atteso: ~18-22 GB
- Tempo atteso: 3-5 min (40 layer MoE → più veloce del denso 27B che ha preso 2:43)

### Fase 6 — Bench A/B + smoke test
Stop `llm-service` già fatto in Fase 4b (resta fermato per tutta la finestra).

Download baseline `ProCreations/grug-35b-v2-Q4_K_M.gguf` (21.2 GB, via `wget -c`):
```bash
wget -c --header "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/ProCreations/grug-35b-v2-gguf/resolve/main/grug-35b-v2-Q4_K_M.gguf" \
  -O ~/llmodels/models/GRUG/grug-35b-v2-Q4_K_M.gguf
```

```bash
# Bench ROCmFP4-STRIX_LEAN
docker run --rm \
  -v ~/llmodels/models:/llmodels:ro \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-bench \
    -m /llmodels/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
    -p 512 -n 128 -ngl 999 -fa on --mmap 0 \
    2>&1 | tee docs/benchmarks/bench-grug-rocmfp4.txt

# Bench Q4_K_M baseline
docker run --rm \
  -v ~/llmodels/models:/llmodels:ro \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-bench \
    -m /llmodels/GRUG/grug-35b-v2-Q4_K_M.gguf \
    -p 512 -n 128 -ngl 999 -fa on --mmap 0 \
    2>&1 | tee docs/benchmarks/bench-grug-q4_k_m.txt
```

Smoke test sidecar (porta 8083), `-c 32768` fisso per evitare context-shift (problema noto DeltaNet su Qwen3.5-MoE):
```bash
docker rm -f grug-test 2>/dev/null || true
docker run -d --name grug-test -p 8083:1234 \
  -v ~/llmodels/models:/llmodels:ro \
  --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service llama-server \
    -m /llmodels/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
    --mmproj /llmodels/GRUG/mmproj-grug-35b-v2-f16.gguf \
    -ngl 999 -fa on --jinja -c 32768 \
    --host 0.0.0.0 --port 1234

# Attesa healthy (max 90s)
for i in {1..18}; do
  curl -s -o /dev/null -m 2 http://localhost:8083/health && break
  sleep 5
done

# Test coding
curl -s http://localhost:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"grug","messages":[{"role":"user","content":"Scrivi una funzione Python che calcola Fibonacci iterativamente. Solo codice."}]}' \
  | tee docs/benchmarks/smoke-grug-coding.json

# Test tool-call JSON
curl -s http://localhost:8083/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"grug","messages":[{"role":"user","content":"Dammi le temperature di Roma in JSON con campi city, temp, unit."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}' \
  | tee docs/benchmarks/smoke-grug-toolcall.json

# CLEANUP sidecar (importante — libera 20+ GB RAM e la porta 8083)
docker rm -f grug-test >/dev/null
```

Report risultati in `docs/benchmarks/results-2026-08-11-grug.md`.

Riavvia `llm-service`:
```bash
docker compose -f <your-workspace>/workspace/docker/docker-compose.yml start llm-service
# attendi health su :1234 (max 5 min)
for i in {1..60}; do
  curl -s -o /dev/null -m 2 http://localhost:1234/health && break
  sleep 5
done
```

### Fase 7 — Cleanup
- Verifica che l'output quantizzato sia sano (size nel range, smoke test passato)
- Cancella safetensors (~70 GB)
- Cancella GGUF BF16 (~70 GB)
- Mantiene: output ROCmFP4-STRIX_LEAN, imatrix, mmproj, calibration text
- Verifica disco finale ≥ 100 GB liberi

---

## 5. Parametri tecnici

| Parametro | Valore | Razionale |
|---|---|---|
| Preset quant | `Q4_0_ROCMFP4_STRIX_LEAN` | Default Strix Halo (K/V + Q5_K emb protect, ~4.38 bpw) |
| `llama-quantize` threads (positional) | 16 | 32 saturano banda UMA LPDDR5X (lezione Qwen3.6-27B e 35B-A3B) |
| `llama-imatrix --threads` | 16 | Stessa logica (CPU-only, pesi in system RAM LPDDR5X) |
| `llama-imatrix --chunks` | 256 | Buona copertura del calibration text |
| `llama-imatrix --no-ppl` | (flag presente) | Salta calcolo perplexity (overhead inutile per pura calibration) |
| `llama-imatrix --parse-special` | (flag presente) | Tokenizza correttamente `<think>`, `<|im_start|>` ecc. del chat template |
| Context size smoke test | 32768 fisso | Evita context-shift → single-token spam (problema DeltaNet noto su Qwen3.5-MoE) |
| Calibration dataset | `grug-think-v3-10k` solo | Dominio target perfetto (grug-speak reasoning + tool-call + coding) |
| Calibration size | ~4 MB plain (~800 conv su 10000) | Best practice imatrix: 1-5 MB (vedi [llama.cpp discussion #5263](https://github.com/ggml-org/llama.cpp/discussions/5263)) |
| BF16 download | `wget -c` con Bearer (header opzionale, repo pubblico) | hf_xet si blocca su file > 1 GB su questa rete (problema già noto) |
| Nome mmproj finale | `mmproj-grug-35b-v2-f16.gguf` | Coerente col repo GGUF ufficiale di ProCreations |

---

## 6. Criteri di successo

| Criterio | Target | Misura |
|---|---|---|
| Output size | 18-22 GB | `stat -c%s` |
| Plain tg128 (ROCmFP4) | ≥ Q4_K_M baseline | `llama-bench -n 128` (A/B) |
| Plain pp512 (ROCmFP4) | ≥ Q4_K_M baseline | `llama-bench -p 512` (A/B) |
| Plain tg128 assoluto | ≥ 40 tok/s | Derivazione: 35B-A3B ROCmFP4-STRIX_LEAN in produzione = 63 tok/s (3B attivi × 0.5 B = 1.5 GB/token, ceiling ~180). grug ha stessi 3B attivi e stesso preset → aspetto ~50-65 tok/s. Target ≥ 40 è conservativo (-30% del riferimento per margine: pesi attenzione, MoE router, overhead DeltaNet). |
| Smoke coding test | Funzione Fibonacci iterativa valida (sintassi Python, logica corretta) | ispezione risposta JSON |
| Smoke tool-call | Tool-call ritornato con JSON args ben formato | ispezione risposta JSON |
| No single-token spam | Conversazione normale, niente `/` infiniti o loop | ispezione (DeltaNet context-shift bug) |
| Disco finale | ≥ 100 GB liberi dopo cleanup | `df -h` |

---

## 7. Rischi e mitigazioni

| ID | Rischio | Probabilità | Impatto | Mitigazione |
|---|---|---|---|---|
| R1 | `convert_hf_to_gguf.py` non riconosce `qwen3_5_moe` con DeltaNet | molto bassa | alto (blocca Fase 3) | Già verificato in adversarial review: il converter del fork ROCmFPX registra `Qwen3_5MoeForConditionalGeneration` come `MmprojModel` → `MODEL_ARCH.QWEN35MOE`. La Fase 3 ha la verifica esplicita `general.architecture = qwen35moe`. Se fallisce: log completo, poi escalation (non c'è fallback equivalente via LoRA: il LoRA di grug è merged nel modello pubblicato, NON disponibile come adapter separato). Fallback realistico: fare FASE 2 Ornith prima (stessa arch, BF16 GGUF già pronto) per de-riskare la pipeline, poi ritentare grug. |
| R2 | Path DeltaNet (layer `linear_attention`) non funziona nel container a runtime | molto bassa | alto (inference NaN/spam) | Già dimostrato in produzione: `Qwen3.6-35B-A3B` (stessa famiglia Qwen3.5-MoE, identico pattern `[L,L,L,F]×10`) serve in `llm-service` a 63 tok/s plain. Il container ha `--swa-full` e infrastruttura `llama-memory-hybrid` per recurrent+attention. Smoke test Fase 6 valida empiricamente. |
| R3 | Context-shift corrompe stato DeltaNet → single-token spam (`/` infiniti) | media | medio | Forzare `-c 32768` in tutti i smoke test (Fase 6); evitare `--context-shift`. Il problema è documentato nel README del GGUF ufficiale di ProCreations. |
| R4 | hf_xet stuck sui shard da 3.9 GB durante download | alta | basso | `wget -c` con `--tries=20 --timeout=60 --retry-connrefused` (problema già noto). `-c` riprende se cade la connessione. |
| R5 | Disco saturato durante conversione | bassa | alto | Picco ~180 GB (safetensors 70 + GGUF BF16 70 + output 20 + Q4_K_M bench 21); 257 GB liberi attuali. Cleanup esplicito a fine Fase 7. |
| R6 | **Dipendenze Python mancanti nel container** (transformers/torch/safetensors/sentencepiece) | alta (certo) | alto (blocca Fase 3) | Fase 0 build immagine derivata `docker-llm-service-convert`. Verificato in adversarial review: il container base è Fedora 43 minimale senza pip. Il converter inizia con `from transformers import AutoConfig` quindi NON è indipendente da transformers. |
| R7 | Quality regression inaccettabile vs Q4_K_M | bassa | alto | Smoke test coding + tool-call in Fase 6; se regression evidente (codice non valido, JSON malformed), provare preset `Q4_0_ROCMFP4_COHERENT` (Q6_K token embeddings, massima qualità JSON/tool, bpw ~4.70 vs 4.38 di STRIX_LEAN — size non ancora misurata in questa workspace, attesa leggermente superiore a STRIX_LEAN per i Q6_K emb, ~+200-400 MB su MoE 35B). |
| R8 | Calibration text troppo omogeneo → imatrix overfitting | bassa | medio | `grug-think-v3-10k` ha 6 fonti diverse con conteggi verificati dal README: glaive 1000 / hermes 1000 / nebius 2200 / smith_ticks 2100 / smith_tool 2200 / toolace 1500. Diversità intrinseca assicurata. |
| R9 | Fase 4a script fallisce: chat template grug non renderizza con jinja2 puro | media | medio | Il chat_template.jinja di grug è standard Qwen3.5 (verificato: 7.54 kB); jinja2 dovrebbe renderizzarlo. Fallback semplice se TemplateError: concatenare `role: content` per ogni message senza template (per imatrix è sufficiente — l'obiettivo è attivare i pesi, non fedeltà chat). |
| R10 | Build immagine `docker-llm-service-convert` fallisce (torch CPU wheel pesante, banda rete) | bassa | medio | Pre-build in Fase 0 con margine; wheel `torch` CPU reale ~800 MB via `--index-url https://download.pytorch.org/whl/cpu`. Se fallisce, fallback: installare deps su venv HOST e copiare il converter dal container. |

---

## 8. Deliverable

1. `~/llmodels/models/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf` (~20 GB)
2. `~/llmodels/models/GRUG/mmproj-grug-35b-v2-f16.gguf` (~857 MB)
3. `~/llmodels/models/GRUG/imatrix-grug-35b-v2.gguf` (~150-300 MB)
4. `~/llmodels/calibration/grug-calibration.txt` (~4 MB)
5. `docker/Dockerfile.convert` (immagine derivata per conversione)
6. `scripts/prep-grug-calibration.py` (preparazione calibration text)
7. `scripts/quantize-grug-35b-v2.sh` (replica di `quantize-qwen36-27b.sh` con adattamenti)
8. `docs/benchmarks/results-2026-08-11-grug.md`
9. `docs/benchmarks/bench-grug-{rocmfp4,q4_k_m}.txt`
10. `docs/benchmarks/smoke-grug-{coding,toolcall}.json`

---

## 9. Comando serving finale (per deploy futuro)

```bash
llama-server -m /llmodels/GRUG/Grug-35B-v2-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  --mmproj /llmodels/GRUG/mmproj-grug-35b-v2-f16.gguf \
  -ngl 999 -fa on --jinja -c 32768 \
  --host 0.0.0.0 --port 1234 \
  --reasoning on --reasoning-budget 8192 \
  --temp 0.6 --top-p 0.95 --top-k 20
```

(Senza `--spec-type draft-mtp`: grug non ha MTP, e allineamento a pipeline MoE plain.)

---

## 10. Follow-up

### FASE 2 — Ornith-1.0-35B (spec separata)
Stessa pipeline applicata a `unsloth/Ornith-1.0-35B-GGUF`:
- BF16 GGUF già pronto (69.4 GB, 18 shard) — nessuna conversione HF→GGUF
- imatrix `imatrix_unsloth.gguf_file` già incluso (192 MB)
- mmproj-F16 incluso (899 MB)
- `mtp_num_hidden_layers=1` (MTP presente) → **ma non attivato a runtime** per allineamento a pipeline MoE plain
- Decisione tecnica: mantenere i pesi MTP (`blk.40`) nel GGUF finale (~1-2 GB extra) o escluderli in conversione — da valutare in FASE 2

### Test opzionali (out-of-scope per questa spec)
- Benchmark sistematico qualità (MMLU-Pro, SWE-bench) vs Q4_K_M ufficiale
- Confronto preset `STRIX_LEAN` vs `COHERENT` per qualità tool-call JSON
- Test contesto lungo (>32K) con stress su DeltaNet recurrent state

---

## 11. Riferimenti

- Pipeline di riferimento: `scripts/quantize-qwen36-27b.sh` (pipeline Qwen3.6, non incluso nel repo)
- Report precedente: `docs/benchmarks/results-2026-08-10.md`
- Plan precedente: piano di implementazione interno `2026-08-10-qwen36-27b-rocmfp4-strix-lean-impl.md` (non incluso nel repo)
- Memoria: `rocmfpx-toolchain-strix-halo`, `rocmfpx-strix-lean-pipeline-status`, `hf-download-stuck-xet`
- Modello sorgente: https://huggingface.co/ProCreations/grug-35b-v2
- GGUF ufficiale baseline: https://huggingface.co/ProCreations/grug-35b-v2-gguf
- Dataset calibration: https://huggingface.co/datasets/ProCreations/grug-think-v3-10k
- Base model: https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B (alias: `ornith-ai/Ornith-1.0-35B`)
- Fork ROCmFPX: https://github.com/charlie12345/ROCmFPX (commit `00d5452`, 9 ago 2026)
- Doc imatrix: https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md
