# Spec: Nemotron 3.5 Lightning 30B-A3B → ROCmFP4-STRIX_LEAN + pubblicazione HF

**Data**: 2026-08-12
**Autore**: user pugant
**Stato**: DRAFT → da validare adversarial review (gate 100%)
**Prerequisiti**: fattibilità verificata 2026-08-12. Pipeline grug/Ornith completata come riferimento (vedi `2026-08-11-grug-35b-v2-strix-lean-design.md`).

## 1. Obiettivo e scope

Quantizzare **NVIDIA-Nemotron-3.5-Lightning-30B-A3B** (MoE hybrid Mamba-2 + MoE + Attention, arch `nemotron_h_moe`, 30B totali / 3B attivi) nel preset **`Q4_0_ROCMFP4_STRIX_LEAN`** (type 106) via fork ROCmFPX su Strix Halo, validare il funzionamento runtime (primo modello Mamba-hybrid su questa toolchain), e pubblicare il GGUF su Hugging Face come repo pubblico sotto `pugant/`.

### In scope
- Rebuild del binary `docker-llm-service` da main corrente (prerequisito: arch `nemotron_h_moe` non nel binary `00d5452`)
- Smoke test di load + generate su gfx1151 (validazione Mamba2-on-HIP reale)
- Generazione imatrix (o riuso se compatibile)
- Quantizzazione ROCmFP4-STRIX_LEAN
- Sanificazione metadata GGUF (come grug/Ornith)
- Bench tg128/pp512 (plain + MTP se supportato)
- Pubblicazione HF: 1 repo pubblico `pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN`

### Out of scope
- Pubblicazione script di pipeline (come grug/Ornith: scope = "Core, modello pronto")
- Eval di qualità (perplexity/MMLU) — dichiarata come limitazione
- Modifica pesi oltre la quantizzazione
- Switch del llm-service produzione (separato, decisione utente post-pubblicazione)
- mmproj: il modello NON è vision (text-only) → nessun mmproj necessario (differenza vs grug/Ornith)

## 2. Base di fattibilità (verificata 2026-08-12, non speculativa)

Tutte le dimensioni tecniche verificate con evidenza (3 subagent + grep binario + check config):

| Dimensione | Esito | Evidenza |
|---|:---:|---|
| Fit Strix Halo (memoria/banda) | ✅ | 30B/3B-attivi = classe grug/Ornith. Ceiling ~180 tok/s. Disco 257 GB liberi |
| Enum arch nel fork | ✅ | `LLM_ARCH_NEMOTRON_H_MOE` in `llama-arch.h/.cpp` + gguf-py (main HEAD) |
| Converter | ✅ | `NemotronHModel` auto-detect MoE. N/A: usiamo GGUF BF16 diretto |
| Compute graph | ✅ | `nemotron-h.cpp`: `build_mamba2_layer` + attention + `build_moe_ffn` + shared expert |
| Mamba2/SSM su HIP | ✅ | Kernel `ssm-conv.cu`/`ssm-scan.cu` via `hip.h`. NON CUDA-only |
| ROCmFP4 K/V protection | ✅ | Attention usa tensor name standard `blk.N.attn_k/attn_v` → protezione automatica |
| Regression test fork | ✅ | `Nemotron-Nano-3-30B-A3B` in test suite, `case 52: LLM_TYPE_31B_A3_5B` |
| Bug dim #20570 | ✅(arith) | `n_group=1`, `hidden_size=2688`, `expand=2→d_inner=5376` (verificati da `config.json`) → `5376 % (1×2688)=0` passa. Assert expression da confermare a smoke test |
| Binary produzione `00d5452` | ❌ | grep rodata: ha `qwen35moe` ma ZERO `nemotron_h_moe`/`mamba2` → **rebuild richiesto** |

## 3. Sorgenti artefatti

### 3.1 BF16 GGUF (sorgente quantizzazione)
- **Repo**: `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` (mirror ufficiale ggml)
- **File**: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf` (65.85 GB, file singolo)
- **Pattern**: come Ornith (BF16 GGUF diretto, salta conversione HF→GGUF). Diverso da grug (che convertiva da safetensors).
- **Download**: `wget -c --header="Authorization: Bearer $TOKEN"` in `~/llmodels/models/NEMOTRON/` (già avviato 2026-08-12, ~13-14 MB/s, ETA ~80 min)
- **NOTA unsloth**: il repo `unsloth/...-GGUF` NON contiene BF16 (verificato via API: solo branch main, solo quant compressi). Fonte autorevole = ggml-org.

### 3.2 imatrix
- **Opzione A (preferita, rapida)**: riusare `bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-imatrix.gguf` (55.3 MB, size verificata via API). Da verificare compatibilità tensor-count vs il BF16 ggml-org (se mismatch → Opzione B).
- **Opzione B (fallback, come grug)**: generare imatrix nostra via `llama-imatrix` (CPU, 16 thread, ~256 chunks). Calibration: dataset generale (wikitext/fineweb sample) o dominio Nemotron (reasoning/coding).
- **Decisione**: provare A, fallback B.

### 3.3 Tokenizer / chat template
- Inclusi nel GGUF BF16 (ggml-org conversione standard). chat_template.jinja presente nel repo nvidia base. Verifica post-download: `llama-server` carica il template senza errori.

## 4. Verifica legale (licenza)

### Licenza modello
- **OpenMDW-1.1** (Linux Foundation, permissiva per distribuzioni ML). NVIDIA adotta per Nemotron family. Fonte: [openmdw.ai/license/1-1/](https://openmdw.ai/license/1-1/), [LF press release](https://www.linuxfoundation.org/press/linux-foundation-releases-openmdw-1-1-nvidia-adopts-openmdw-for-cosmos-isaac-gr00t-ising-and-nemotron-ai-model-families).
- **Ridistribuzione derivative (GGUF quantizzato)**: **consentita** con obblighi: (1) includere copia dell'accordo OpenMDW-1.1, (2) mantenere tutti i copyright/origin notices.
- → Pubblicazione HF OK. File `LICENSE` (OpenMDW-1.1 full text) + `NOTICE` (attribution NVIDIA + ggml-org + ROCmFPX + bartowski per imatrix).

### Catena di derivazione
```
nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16  (OpenMDW-1.1)  [base]
    └── ggml-org/...-GGUF (BF16 GGUF conversione)  [sorgente nostra]
            └── pugant/...-ROCmFP4-STRIX_LEAN  (OpenMDW-1.1 derivative)  [output]
```

### Componenti terzi
| Componente | Licenza | Ridistribuzione GGUF? |
|---|---|---|
| nvidia base model | OpenMDW-1.1 | Sì, con license + notices |
| ggml-org BF16 GGUF | OpenMDW-1.1 (derivative of nvidia) | Sì, attribution |
| charlie12345/ROCmFPX (fork quant) | MIT | Sì (codice MIT; non impone termini sul GGUF output) |
| bartowski imatrix (se usata) | OpenMDW-1.1/mit (verificare) | Sì, attribution NOTICE |
| kyuz0 toolbox (container) | Nessuna LICENSE (404 verificato per grug) | N/A (solo attribution come link) |

## 5. Rebuild binary `docker-llm-service` (PREREQUISITO)

### Perché
Il binary produzione (commit `00d5452`, build 9 ago) NON ha `LLM_ARCH_NEMOTRON_H_MOE`. grep rodata: zero match `nemotron_h`/`mamba2`. Serve main corrente.

### Come — `docker build` diretto, NON compose
- **Dockerfile**: `<your-workspace>/workspace/docker/amd-strix-halo-toolboxes/toolboxes/Dockerfile.rocm-7.2.4-rocmfpx` (verificato esistente, 3676 B).
- Il Dockerfile clona `charlie12345/ROCmFPX@main` (`ARG BRANCH=main`, riga 38). **Rebuildare l'immagine = pesca automaticamente main HEAD** con nemotron_h_moe. Nessun edit Dockerfile.
- Build context = la dir `toolboxes/` (contiene i COPY source `llama-grammar.patch` 377 B + `gguf-vram-estimator.py` 7885 B, entrambi verificati presenti).
- Target: gfx1151, ROCm 7.2.4, `GGML_HIP_FORCE_MMQ=ON`.

**Comando build (tag separato, bypassa compose)**:
```bash
docker build --network host -t docker-llm-service:nemotron \
  -f <your-workspace>/workspace/docker/amd-strix-halo-toolboxes/toolboxes/Dockerfile.rocm-7.2.4-rocmfpx \
  <your-workspace>/workspace/docker/amd-strix-halo-toolboxes/toolboxes
```
Tempo stimato: ~20-40 min (compilazione HIP gfx1151 con nproc=32, kernel SSM aggiuntivi).

### Coordinamento produzione — SAFETY (verificato)
- Il container `docker-llm-service:latest` **serve in produzione** (gestito da compose service `llm-service` con `restart: unless-stopped`).
- **⚠️ NON usare `docker compose build llm-service` / `docker compose up -d` per questo lavoro**: il service compose ha `build:` **senza `image:`** (verificato nel deployment locale) → compose build **sovrascriverebbe `:latest`** e `up -d` **ricreerebbe il container produzione** interrompendo il servizio — adjust your local llama.cpp service deployment di conseguenza.
- **Strategia**: buildare solo il tag `docker-llm-service:nemotron` (comando sopra). Smoke test e quantizzazione girano in `docker run docker-llm-service:nemotron`. `:latest` e il container produzione **non vengono toccati**. Promozione a `:latest` = decisione utente separata, fuori scope.
- (Nota: eventuali regole locali di restart "kill+start" si riferiscono a un server dflash secondario su `<llm-service-port>`, non al llm-service compose-managed — adjust your local llama.cpp service deployment.)

### Verifica post-rebuild
- grep rodata nuovo binary per `nemotron_h_moe` E `mamba2` (devono ora matchare).
- `llama-server --version`: commit più recente di `00d5452`.

## 6. Smoke test (validazione runtime Mamba2-on-HIP)

Dopo rebuild, **prima** di quantizzare, validare che il modello giri (tutto in container `docker-llm-service:nemotron` — **mai `:latest`/produzione**, che non riconosce l'arch). Test economico per de-risking.

### ⚠️ Prerequisito memoria (verificato 2026-08-12)
Strix Halo è APU: `rocm-smi` riporta **VRAM partition = 512 MB** soltanto (no VRAM dedicata significativa). La memoria reale è la LPDDR5X unificata (124 GB), accessibile a ggml solo con **`-e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`** (è ciò che la produzione setta, compose riga 364). **Senza questa env var, nessun modello carica** su questa GPU. → Obbligatoria in OGNI `docker run` che fa inference GPU (smoke §6, bench §10). La quantizzazione (§8) è CPU-only → non serve.

### Procedura
1. **Verifica arch + metadata** del BF16 GGUF: `docker run --rm -v ~/llmodels/models:/llmodels docker-llm-service:nemotron llama-gguf /llmodels/NEMOTRON/<BF16>.gguf r` → `general.architecture = nemotron_h_moe`, 52 layer, layer types misti (mamba/moe/attention).
2. **Smoke load + generate** — due opzioni:

   **Opzione 1 (preferita, valida la sorgente quant reale)**: fermare brevemente la produzione per isolare la GPU, caricare il BF16 (65 GB in UMA unificata, ci sta ampiamente):
   ```bash
   docker stop llm-service                                          # ~10s, interruzione breve
   docker run --rm -d --name nemotron-smoke \
     --device /dev/kfd --device /dev/dri --group-add video --group-add render \
     -e HSA_OVERRIDE_GFX_VERSION=11.5.1 -e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 \
     -p 1235:1234 -v ~/llmodels/models:/llmodels \
     docker-llm-service:nemotron llama-server \
       -m /llmodels/NEMOTRON/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf \
       -ngl 999 -c 4096 --host 0.0.0.0 --port 1234
   # poll health finché "ready", poi curl /v1/chat/completions, poi cleanup:
   docker stop nemotron-smoke && docker start llm-service           # ripristina produzione
   ```
   **Opzione 2 (nessuna interruzione prod)**: scaricare un quant leggero (es. unsloth `UD-IQ2_M` 19 GB, stessa arch + Mamba path), smoke-test alongside produzione (19 + ~20 GB prod = 39 GB ≪ 124 GB UMA). Più economico in RAM, costoso in tempo download (~25 min). Valida arch+Mamba ma NON la sorgente BF16.

3. **Verifiche** (entrambe le opzioni):
   - Poll `curl localhost:1235/health` finché `{"status":"ok"}` (modello caricato)
   - curl `/v1/chat/completions` con prompt semplice → testo coerente
   - Assenza assert `d_inner % (n_group*n_embd)` #20570 nei log (Mamba path OK su HIP)
4. **Cleanup OBBLIGATORIO**: `docker stop nemotron-smoke` (il container ha `--rm` → auto-rimosso allo stop) + riavvio produzione se Opzione 1.
5. **Se fallisce**: diagnosticare (bug #20570, kernel SSM gfx1151, arch non riconosciuta). Fermarsi e ri-valutare PRIMA di quantizzare.

> Solo se il smoke test passa si procede a quantizzare. Risparmia ore se ci sono blocchi runtime imprevisti sui kernel Mamba.

## 7. Verifica MTP / nextn

- Config modello: `num_nextn_predict_layers: 1`, `mtp_layers_block_type: ["attention", "moe"]`.
- **Verifica post-download**: il BF16 ggml-org contiene il blocco nextn/MTP? Per Nemotron-H il naming GGUF dei layer MTP va verificato (diverso da Qwen `blk.64`). Cercare tensori `nextn.*` o un blocco layer extra oltre i 52 dichiarati.
- **⚠️ Convenzione split-file osservata**: bartowski pubblica un file **separato** `mtp-NVIDIA-Nemotron-…-Q4_0.gguf` (~1.15 GB) come drafter, accanto al GGUF principale. Questo suggerisce che Nemotron-H MTP segua la convenzione **split-file / draft-model separato** (caricato via `--spec-model` o `-md`), NON l'MTP embedded `blk.N` + `--spec-type draft-mtp` usato da Qwen3.6. Quindi: anche se il blocco nextn è presente, `--spec-type draft-mtp` potrebbe non applicarsi.
- **Se presente + convenzione embedded**: testare `--spec-type draft-mtp`. Misurare acceptance + tok/s.
- **Se convenzione split-file**: scaricare/estrarre il drafter MTP e testare speculative decoding con draft-model.
- **Se assente o non funzionante**: plain inference (comunque valido; il modello è già veloce come 3B-attivi MoE). MTP = bonus dichiarato come work-in-progress.

## 8. Quantizzazione ROCmFP4-STRIX_LEAN

```bash
docker run --rm -v ~/llmodels/models:/llmodels \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  docker-llm-service:nemotron llama-quantize \
    --imatrix /llmodels/NEMOTRON/<imatrix> \
    /llmodels/NEMOTRON/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf \
    /llmodels/NEMOTRON/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf \
    Q4_0_ROCMFP4_STRIX_LEAN 16
```
- `nthreads=16` (ottimo su Ryzen AI Max+ 395).
- `--dry-run` prima per stimare size output.
- **Filename output**: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf` (coerente col nome repo HF e upload §11).
- **Size attesa**: ~30B × 4.38 bpw / 8 ≈ **16.4 GB** (range 16-17 GB, comparabile grug/Ornith 17.32 GiB).
- **Tempo atteso**: pochi minuti (27B denso = 2:43; MoE 30B simile, la quantizzazione è CPU-bound sui **parametri totali 30B** — tutti gli expert + mamba + attention, non i 3B attivi).

### Validazione post-quant
- `llama-server` carica il quant senza errori.
- `llama-bench -ngl 999 -fa on -p 512 -n 128 -mmap 0` → size confermata, tg128/pp512.

## 9. Sanificazione metadata GGUF (pre-upload, OBBLIGATORIA)

Come grug/Ornith (vedi `reference-gguf-sanitization-bugs` per la SOLUZIONE API low-level `add_tensor_info` + `write_tensor_data` — NON `add_tensor` che è rotta per dtype custom).

**Operazioni** (script Python con gguf-py nel container `docker-llm-service-convert`):
1. Leggere KV metadata esistenti
2. Modificare/aggiungere:
   - `general.name`: `'NVIDIA-Nemotron-3.5-Lightning-30B-A3B'`
   - `general.basename`: `'NVIDIA-Nemotron-3.5-Lightning-30B-A3B'`
   - `general.size_label`: `'30B-A3B'`
   - `general.organization`: `'NVIDIA'`
   - `general.repo_url`: `'https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16'`
   - `general.base_model.0.name`: `'NVIDIA Nemotron 3.5 Lightning 30B A3B'`
   - `general.base_model.0.repo_url`: (come sopra)
   - `general.license`: `'openmdw-1.1'`
   - `general.quantized_by`: `'pugant'`
   - `general.tags`: `[gfx1151, rocmfpx, strix-halo, rocm, amdgpu, nemotron, mamba2, moe]`
3. Sostituire path locali (`quantize.imatrix.file`/`.dataset`) con basename neutri
4. Copiare tensori 1:1 invariati, scrivere `*.sanitized.gguf`
5. Verifica: SHA256, `llama-bench` smoke (load + 1 token), dump KV conferma valori nuovi

**Costo**: ~10 min (copia ~16 GB a ~1 GB/s NVMe). Riusa `scripts/sanitize-gguf-v2.py` (non incluso nel repo; adattando config).

## 10. Bench

- **Metodologia**: `llama-bench -ngl 999 -fa on -p 512 -n 128 -mmap 0` (in container `docker-llm-service:nemotron`, con `-e GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` — vedi §6 prerequisito memoria) + (per MTP) `llama-server` + curl `/v1/chat/completions` leggere `timings.predicted_per_second` (llama-bench NON esercita MTP — osservazione interna).
- **Configurazione sistema dichiarata** (come grug/Ornith): Bosgame BeyondMax, Ubuntu 24.04.4, kernel 7.0.0-28, power profile `balanced` (non forzato performance), amd-pstate-epp.
- **Range atteso tg128 plain**: ~50-70 tok/s (classe 3B-attivi MoE; grug 70.92, Ornith 66.68). Incertezza: maturità kernel SSM su gfx1151. Il Mamba (KV fisso) dovrebbe aiutare a context lungo.
- **Baseline comparativa**: Q4_K_M (scaricato da unsloth o ggml-org) stesso bench.
- **Output**: `docs/benchmarks/bench-nemotron-{rocmfp4,q4_k_m}.txt`, report `docs/benchmarks/results-2026-08-12-nemotron.md`.

## 11. Pubblicazione HF

### Repo
- **Path**: `pugant/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN`
- **Visibility**: public
- **License**: openmdw-1.1
- **Files**:
```
README.md                                              # model card EN
LICENSE                                                # OpenMDW-1.1 full text
NOTICE                                                 # attribution chain (NVIDIA + ggml-org + ROCmFPX + bartowski)
.gitattributes                                         # LFS config (*.gguf)
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-ROCmFP4-STRIX_LEAN.gguf   # ~16.4 GB (main, LFS)
NVIDIA-Nemotron-3.5-Lightning-30B-A3B-imatrix.gguf     # imatrix (LFS) — nostra o bartowski (nome coerente §3.2)
```
(NESSUN mmproj — text-only model.)

### Model card (README.md, EN)
Struttura su modello grug/Ornith, sezioni:
1. HF YAML frontmatter: `library_name: llama.cpp`, **`license: other` + `license_name: openmdw-1.1` + `license_link: https://openmdw.ai/license/1-1/`** (HF sidebar non riconosce `openmdw-1.1` come valore `license:` bare — pattern `other`+`license_name` come da card NVIDIA), `base_model: nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, `tags: [rocmfpx, gfx1151, strix-halo, nemotron, mamba2, moe, rocm, amdgpu]`, `pipeline_tag: text-generation`
2. TL;DR + ⚠️ Critical warnings (type 106 = invalid in stock llama.cpp; richiede ROCmFPX fork; FP4 software su RDNA 3.5; **a nostra conoscenza, primo modello Mamba-hybrid pubblicato in questo formato ROCmFP4**)
3. Benchmarks (tg128/pp512 ROCmFP4 vs Q4_K_M, sistema dichiarato)
4. Quantization details (STRIX_LEAN, 4.38 bpw, K/V protect + Q5_K emb)
5. **Architettura** (sezione extra vs grug/Ornith): spiegare nemotron_h_moe = Mamba-2 + MoE + sparse Attention hybrid, perché interessante su Strix Halo (KV fisso → long-context)
6. imatrix methodology
7. Usage (comandi Strix Halo, nota MTP status)
8. Attribution & model tree
9. License (OpenMDW-1.1), Acknowledgements, Limitations, Disclaimer

### Upload
- `huggingface_hub` 1.22, token write-mode `~/.cache/huggingface/token` (user `pugant`).
- Riusa `scripts/upload-to-hf.py` (adattando repo_id + file list).
- Account free: 500 GB storage, 50 GB/file. Totale ~17 GB → OK.
- Ordine: LICENSE/NOTICE/.gitattributes → GGUF + imatrix → README.md (live card per ultimo).

## 12. Criteri di successo

1. Binary rebuildato (tag `:nemotron`) con `nemotron_h_moe` + `mamba2` in rodata
2. Smoke test BF16: load + generate su gfx1151 senza assert falliti
3. GGUF ROCmFP4-STRIX_LEAN prodotto (~16-17 GB, carica in llama-server)
4. Metadata sanificati (verify post-sanitization)
5. Bench documentato (tg128/pp512, plain; + MTP se disponibile)
6. Repo HF pubblico creato con tutti i file, card renderizzata, license visibile, model tree popolato
7. Nessun dato sensibile pubblicato (re-check grep path locali)

## 13. Rischi e mitigation

| Rischio | Probabilità | Mitigazione |
|---|:---:|---|
| Kernel SSM gfx1151 immaturi → smoke test BF16 fallisce/lento | Media | Smoke test PRIMA di quantizzare. Se fallisce, stop e diagnosi. Modello resta valido come esperimento documentato |
| Bug dim #20570 si triggera sul nostro modello a runtime | Bassa | `n_group=1` config → aritmetica passa. Confermato a smoke test |
| MTP/nextn non supportato a runtime su nemotron_h_moe | Media | Plain inference resta valore. MTP = bonus, dichiarato come work-in-progress |
| Rebuild rompe produzione (:1234) | Bassa | Nuovo tag `:nemotron`, non toccare `:latest` |
| imatrix bartowski incompatibile (tensor mismatch) | Bassa-Media | Fallback genera nostra (Opzione B) |
| ggml-org BF16 non include MTP/nextn block | Bassa | Plain inference; oppure valuta repo `h1st0ry3D/...-MTP-GGUF` |
| License metadata OpenMDW errata nel frontmatter | Bassa | Verifica pre-upload pattern HF `license: other` + `license_name: openmdw-1.1` + `license_link` (NON bare `license: openmdw-1.1`, HF sidebar non lo renderizza) |

## 14. Decisioni differite / out-of-scope

- **Promozione rebuild a `:latest` produzione**: separato, decisione utente post-validazione
- **Eval qualità (perplexity/MMLU)**: lavoro futuro su input community
- **Comparazione preset COHERENT vs STRIX_LEAN** per tool-call: futuro
- **Pubblicazione di un quant Q4_K_M "sicuro" stock-llama.cpp** come fallback per non-ROCmFPX utenti: futuro
