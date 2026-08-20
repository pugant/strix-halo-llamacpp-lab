# Spec: Pubblicazione HF grug + Ornith ROCmFP4-STRIX_LEAN

**Data**: 2026-08-11
**Autore**: user pugant
**Stato**: DRAFT → da validare adversarial review
**Prerequisiti**: FASE 1 grug + FASE 2 Ornith completate (T4-T9 ✓). Vedere `docs/design/2026-08-11-grug-35b-v2-strix-lean-design.md`.

## 1. Obiettivo e scope

Pubblicare su Hugging Face due repository pubblici con i GGUF quantizzati `Q4_0_ROCMFP4_STRIX_LEAN` di **grug-35b-v2** e **Ornith-1.0-35B**, prodotti dalla pipeline ROCmFPX su Strix Halo, per condivisione con la community (in particolare altri utenti Strix Halo / gfx1151).

### In scope
- 2 repo HF pubblici sotto `pugant/`
- GGUF quantizzato + mmproj + imatrix (grug) per ciascun modello
- Model card EN con benchmark, metodologia, usage, attribution
- File LICENSE + NOTICE per ciascun repo
- Upload via `huggingface_hub` (token in `~/.cache/huggingface/token`, user `pugant`)

### Out of scope
- Pubblicazione degli script di pipeline (scelta utente: "Core, modello pronto")
- Pubblicazione del calibration text integro (contiene codice OS di terzi)
- Eval di qualità (perplexity/MMLU) — dichiarata come limitazione, invito al feedback community
- Modifiche ai pesi o ai metadata del GGUF (pubblichiamo così come generati)
- Riavvio del llm-service produzione (separato)

## 2. Verifica legale (licenze)

### Catena di derivazione (model tree)
```
Qwen3.5-VL-MoE (arch base)
    └── ornith-ai/Ornith-1.0-35B (alias `deepreinforce-ai/Ornith-1.0-35B`, MIT)  [base]
            └── ProCreations/grug-35b-v2 (Apache-2.0) [fine-tune]
```
Model tree verificato via HF API: `ProCreations/grug-35b-v2` cardData.base_model dichiara `deepreinforce-ai/Ornith-1.0-35B` (namespace legacy). Quel repo fa redirect 307 → `ornith-ai/Ornith-1.0-35B` (canonical). Nel YAML frontmatter dei nostri repo useremo `ornith-ai/Ornith-1.0-35B` (canonical) per allineamento col model tree HF.

### Licenze componenti
| Componente | Licenza | Fonte verifica | Permette ridistribuzione GGUF derivato? |
|---|---|---|---|
| `ProCreations/grug-35b-v2` (modello base grug) | **Apache-2.0** | HF metadata `license: apache-2.0` | Sì, con attribution + license inclusion + NOTICE |
| `ornith-ai/Ornith-1.0-35B` (modello base Ornith) | **MIT** | HF metadata `license: mit` + unsloth card | Sì, con attribution + license inclusion |
| `charlie12345/ROCmFPX` (fork quantizzazione) | **MIT** | README repo GitHub | Sì (MIT sul codice; non impone termini sui GGUF output) |
| `kyuz0/amd-strix-halo-toolboxes` (container) | **Nessuna LICENSE** (GitHub API 404 verificato 2026-08-11) | GitHub repo | N/A (non ridistribuiamo il container; usiamo solo attribution come riferimento link) |
| `ProCreations/grug-think-v3-10k` (dataset calibration) | Apache-2.0, **pubblico non gated** | HF API verificata | Non ridistribuiamo il dataset, solo riferimento + attribution. Community può scaricarlo liberamente. |
| Qwen3.5-VL-MoE (arch base) | non direttamente nel model tree HF di grug (base dichiarata = Ornith) | — | Licenza efectiva = quella del model tree HF |

### Obblighi di attribuzione
- **Apache-2.0** (grug): includere license text + NOTICE file + dichiarazione "derivative work". Apache-2.0 sezione 4 impone di mantenere tutte le attribution/NOTICE.
- **MIT** (Ornith): includere license text + copyright notice.
- Entrambi i repo avranno `LICENSE` (full text) + `NOTICE` (attribution chain).

### Caveat legale (da dichiarare nel model card)
- "No affiliation with AMD, Qwen, ProCreations, DeepReinforce, unsloth, kyuz0, or charlie12345."
- "Provided as-is, without warranty."
- "Users must comply with the base model license (Apache-2.0 / MIT)."

## 3. Verifica sicurezza (dati sensibili)

### Verifiche eseguite empiricamente (2026-08-11, adversarial review)

| Check | Risultato |
|---|---|
| Token HF negli script pubblicati | N/A (scope = "Core, modello pronto" → nessuno script pubblicato) |
| **Path locali embedded come metadata GGUF** | **CONFERMATI via `grep -aoE` su file intero** (NON tramite `strings` che ha soglia minima e li ha mancati):<br>• grug: `quantize.imatrix.file = '/llmodels/GRUG/imatrix-grug-35b-v2.gguf'`, `quantize.imatrix.dataset = '/calibration/grug-calibration.txt'`<br>• Ornith: `quantize.imatrix.file = '/llmodels/ORNITH/imatrix_unsloth.gguf_file'`, `quantize.imatrix.dataset = 'unsloth_calibration_Ornith-1.0-35B.txt'`<br><br>Nessun path nella home utente o token HF. I path `/llmodels/` non sono credenziali ma rivelano convenzioni interne → **sanificazione richiesta** (vedi §3.1). |
| **Metadata GGUF errati o generici** | **CONFERMATI via parser GGUF Python**:<br>• grug: `general.name = '35B BF16'` (generico, nessuna menzione "grug")<br>• Ornith: `general.name = 'Ornith-1.0-9B'` (**ERRATO: dice 9B invece di 35B**, errore ereditato da unsloth)<br><br>→ **sanificazione richiesta** (vedi §3.1). |
| Calibration text (non pubblicato) | 140 match keyword "sensibili" = **tutti falsi positivi**: codice Python OS (paramiko, marshmallow) con nomi attributi `password`/`api_key` + email pubbliche autori OS in copyright notice. Nessun dato sensibile reale. |
| Token HF negli script interni | Presenti negli script (`scripts/*.sh`) ma NON pubblicati (scope = Core, no script). |

### 3.1 Sanificazione metadata GGUF (pre-upload, **OBBLIGATORIA**)

Prima dell'upload HF, entrambi i GGUF vengono sanificati via script Python con libreria `gguf` (path riutilizza `/usr/local/bin/convert_hf_to_gguf.py` deps installati nel container `docker-llm-service-convert`).

**Operazioni** (per entrambi i GGUF):
1. Leggere tutti i KV metadata esistenti
2. Modificare / aggiungere:
   - `general.name`: `'grug-35b-v2'` (grug) / `'Ornith-1.0-35B'` (Ornith) — fix C1/C2
   - `general.basename`: `'grug'` / `'Ornith-1.0-35B'`
   - `general.size_label`: `'35B'`
   - `general.finetune`: `'grug-v2'` (grug) / `'1.0'` (Ornith)
   - `general.organization`: `'ProCreations'` / `'DeepReinforce'`
   - `general.repo_url`: `'https://huggingface.co/ProCreations/grug-35b-v2'` / `'https://huggingface.co/ornith-ai/Ornith-1.0-35B'`
   - `general.base_model.0.name`: `'Ornith 1.0 35B'` (entrambi, Ornith come base)
   - `general.base_model.0.repo_url`: `'https://huggingface.co/ornith-ai/Ornith-1.0-35B'`
   - `general.license`: `'apache-2.0'` (grug) / `'mit'` (Ornith)
   - `general.quantized_by`: `'pugant'`
   - `general.tags`: aggiungere `[gfx1151, rocmfpx, strix-halo, rocm, amdgpu]`
3. Rimuovere o sostituire con placeholder neutri:
   - `quantize.imatrix.file`: `'grug-calibration.gguf'` (grug) / `'unsloth-imatrix.gguf'` (Ornith) — solo basename, no path — fix C3
   - `quantize.imatrix.dataset`: `'grug-think-v3-10k'` (grug) / `'unsloth-calibration'` (Ornith)
4. Copiare tensori 1:1 invariati
5. Scrivere nuovo file GGUF su path temporaneo (es. `*.sanitized.gguf`)
6. Verifica post-sanificazione: ricalcolo SHA256, `llama-bench` smoke (caricamento + 1 token), `llama-gguf ... r` per dump KV e verifica valori nuovi
7. Sostituire il file originale col sanificato (rinomina)

**Costo**: ~10 min per GGUF (copia 17 GB con metadata nuovi, su NVMe interno ~1 GB/s).
**SHA256**: cambia post-sanificazione → ricalcolare per il report.

### Conclusione sicurezza (post-sanificazione)
Post-sanificazione, i GGUF pubblicati avranno:
- ✓ Nome modello corretto e identificabile
- ✓ Attribution completa (organizzazione, repo URL, base model)
- ✓ License embedded nel metadata
- ✓ Nessun path locale embedded (solo basename neutri)
- ✓ Tags per discoverability HF

## 4. Struttura repo

### 4.1 Repo grug
**Path**: `pugant/grug-35b-v2-ROCmFP4-STRIX_LEAN`
**Visibility**: public
**Files**:
```
README.md                                  # model card EN
LICENSE                                    # Apache-2.0 full text
NOTICE                                     # attribution chain + ROCmFPX commit
.gitattributes                             # LFS config (*.gguf, *.png)
grug-35b-v2-ROCmFP4-STRIX_LEAN.gguf        # 17.32 GiB (main, LFS)
imatrix-grug-35b-v2.gguf                   # 183 MB (nostra, LFS)
mmproj-grug-35b-v2-f16.gguf                # 857 MB (vision projector, LFS)
```

### 4.2 Repo Ornith
**Path**: `pugant/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN`
**Visibility**: public
**Files**:
```
README.md                                  # model card EN
LICENSE                                    # MIT full text
NOTICE                                     # attribution chain (ornith-ai + unsloth + ROCmFPX)
.gitattributes                             # LFS config
Ornith-1.0-35B-ROCmFP4-STRIX_LEAN.gguf     # 17.32 GiB (main, LFS)
imatrix.dat                                # 183 MB (copia imatrix unsloth, LFS) — attribution nel NOTICE
mmproj-F16.gguf                            # 857 MB (LFS)
```

**Nota imatrix Ornith**: l'imatrix deriva da `unsloth/Ornith-1.0-35B-GGUF/imatrix_unsloth.gguf_file`. La ridistribuiamo come `imatrix.dat` con attribution esplicita a unsloth nel NOTICE (licenza MIT di unsloth lo permette).

## 5. Model card (README.md, EN)

Struttura sezioni (ispirata a unsloth, adattata Strix-specific):

1. **HF metadata YAML** (frontmatter)
   - `library_name: llama.cpp`
   - `license: apache-2.0` (grug) / `mit` (Ornith)
   - `base_model: ProCreations/grug-35b-v2` / `ornith-ai/Ornith-1.0-35B`
   - `tags: [rocmfpx, gfx1151, strix-halo, qwen35moe, moe, rocm, amdgpu, ROCmFP4]`
   - `pipeline_tag: image-text-to-text` (modello vision + text)
   - `language: [en, multilingual]`
   - `version: 1.0` + `date: 2026-08-11`

2. **Version header**: `> Version 1.0 — 2026-08-11`

3. **TL;DR** (2 righe): modello + quant + target hardware

4. **⚠️ Critical warnings** (top, callout visibile):
   - Richiede **kyuz0 Strix Halo toolbox** (che builda `charlie12345/ROCmFPX`). Type 106 (Q4_0_ROCMFP4_STRIX_LEAN) = **INVALID in stock llama.cpp**.
   - Profilato per **gfx1151** (Strix Halo / RDNA 3.5). Altrove non testato.
   - FP4 è **software** su RDNA 3.5 (nessuna unità FP4 in silicio) — beneficio è bandwidth memory, non compute.

5. **Benchmarks** — tabella tg128/pp512 (grug ROCmFP4: 70.92 vs grug Q4_K_M 61.18, +16%; Ornith ROCmFP4: 66.68; grug size 17.31 GiB vs Q4_K_M 19.70 GiB, -12%). Su Strix Halo Ryzen AI Max+ 395, 128 GB LPDDR5X. Metodologia: `llama-bench -ngl 999 -fa on -p 512 -n 128 -mmap 0`. Riferimento produzione: Qwen3.6-35B-A3B ROCmFP4-STRIX_LEAN 63 tok/s. Fonti: `docs/benchmarks/bench-grug-{rocmfp4,q4_k_m}.txt`, `logs/bench-ornith-vs-grug.log`.

   **Configurazione sistema al momento del bench** (dichiarata per riproducibilità):
   - **Bare metal host**: Bosgame BeyondMax Series (`bosgame-m5`), Ubuntu 24.04.4 LTS, kernel 7.0.0-28-generic
   - **CPU power profile**: `balanced` (`powerprofilesctl get`) — configurazione di default, **NON forzata a `performance`**. Rappresentativa di un setup out-of-the-box.
   - **CPU scaling driver**: `amd-pstate-epp`, scaling_governor `performance` (default amd-pstate-epp), EPP `performance`
   - **IOMMU/iGPU power**: auto (nessun tuning manuale)

   Nota: i tok/s sono misurati su sistema NON ottimizzato (power profile balanced). Utenti che impostano `powerprofilesctl set performance` possono ottenere valori leggermente superiori.

6. **Quantization details** — preset `Q4_0_ROCMFP4_STRIX_LEAN` (type 106, 4.29 BPW). Cosa protegge: K/V attention (`attn_qkv`/`attn_v` → q4_0_rocmfp4) + Q5_K token embeddings. Expert (`ffn_*_exps`) in `q4_0_rocmfp4_fast` (max speed). Riferimento fork: `charlie12345/ROCmFPX` commit `00d5452`.

7. **imatrix methodology**
   - **grug**: generata con `llama-imatrix` 256 chunks, 16 thread, CPU-only. Calibration da `ProCreations/grug-think-v3-10k` (**dataset pubblico Apache-2.0, non gated** — chiunque può scaricarlo per replicare). Ringraziamento esplicito al grug team. 510 entries su 733 tensori. Warning `partial data 99.61%` = 1/256 expert non attivato nel calibration (normale per MoE, vedi codice llama.cpp `tools/imatrix/imatrix.cpp`).
   - **Ornith**: imatrix precomputata da unsloth (46 chunks). Inclusa come `imatrix.dat`.

8. **Usage** — comandi Strix Halo:
   ```bash
   # Richiede container kyuz0/charlie12345 ROCmFPX
   llama-server -m <model>.gguf --mmproj <mmproj>.gguf \
     -ngl 999 -fa on --jinja -c 32768 --host 0.0.0.0 --port 1234
   ```
   (Nota: MTP non attivato per grug — assente nel modello. Per Ornith — presente `mtp_num_hidden_layers=1` ma non attivato a runtime per allineamento MoE plain.)

9. **How to replicate** (testo, no script pubblicati) — pipeline a parole: builda container ROCmFPX → download BF16 → `convert_hf_to_gguf.py` (grug) / usa BF16 GGUF unsloth (Ornith) → `llama-imatrix` (grug, dataset grug-think-v3-10k) → `llama-quantize ... Q4_0_ROCMFP4_STRIX_LEAN 16`.

10. **Attribution & model tree** — base model link, chain (Qwen3.5-VL-MoE → Ornith → grug), commit ROCmFPX, tag kyuz0 toolbox, unsloth credit.

11. **License** — Apache-2.0 (grug) / MIT (Ornith). "Derivative work: original model and its license are preserved. See LICENSE and NOTICE."

12. **Acknowledgements** — sezione dedicata (vedi §5.1).

13. **Limitations & community feedback** — speed benchmark only, no quality eval. Invito esplicito: "We invite the community — especially fellow Strix Halo owners — to test and share quality results. Open a Discussion."

14. **Citation** — bibtex placeholder (grug: ProCreations; Ornith: DeepReinforce Team, URL `deep-reinforce.com/ornith_1_0.html`).

15. **Disclaimer** — no affiliation, as-is.

### 5.1 Sezione Acknowledgements
```markdown
## Acknowledgements
Built on the shoulders of giants:
- [kyuz0/amd-strix-halo-toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes) — Strix Halo container runtime
- [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) — llama.cpp fork with ROCmFP4 presets (type 106)
- [unsloth](https://huggingface.co/unsloth) — BF16 GGUF + imatrix for Ornith
- [ProCreations](https://huggingface.co/ProCreations) — grug-35b-v2 + grug-think-v3-10k calibration dataset
- [ornith-ai / DeepReinforce Team](https://huggingface.co/deepreinforce-ai) — Ornith-1.0-35B
- [llama.cpp](https://github.com/ggerganov/llama.cpp) community + Kawrakow (imatrix methodology)
- Hardware: Bosgame BeyondMax Series (Strix Halo bare metal host)
```

## 6. Upload mechanism

### Tool
`huggingface_hub` Python lib v1.22.0 (verificata su host). Token: `~/.cache/huggingface/token`. User confermato: `pugant`, nessuna org.

### Verifica preventiva storage HF (2026-08-11)
- `pugant` è account **free** (`isPro=False, plan=None, canPay=False` via `hf whoami-v2`)
- Limiti HF free: 500 GB total storage, 50 GB per singolo file LFS ([docs](https://huggingface.co/docs/hub/en/storage-limits))
- Pubblicazione totale: ~36 GB (grug ~18 GB + Ornith ~18 GB) → **OK entro soglia**
- Per-file: GGUF 17 GB < 50 GB limit ✓

### Procedura (per repo)
1. `HfApi().create_repo(repo_id="pugant/<name>", repo_type="model", private=False, exist_ok=False)`
2. Upload `.gitattributes` + `LICENSE` + `NOTICE` (file piccoli)
3. `upload_large_folder()` o `upload_file()` per GGUF (17 GB), mmproj (857 MB), imatrix (183 MB) — LFS automatico, resume on failure
4. `upload_file()` per `README.md` (per ultimo, così il repo è "completo" quando la card va live)
5. sha256 check post-upload (opzionale, `huggingface_hub` non lo fa nativamente; confronto `hf_hub_url` size vs locale)

### Ordine upload
1. LICENSE, NOTICE, .gitattributes (setup)
2. File grandi (GGUF, mmproj, imatrix)
3. README.md (live card)

### Resume
`huggingface_hub` handle resume automatico per LFS. Se upload interrompe, re-run stesso comando riprende.

## 7. Criteri di successo

1. Entrambi i repo creati `public` sotto `pugant/`
2. Tutti i file presenti con size corretta (verifica via HF API `repo_info`)
3. README.md si renderizza correttamente (preview HF)
4. License visibile nel sidebar HF (metadata `license` nel frontmatter)
5. Model tree HF popolato (`base_model` link funzionante)
6. Nessun dato sensibile pubblicato (re-check post-upload: grep path locali nel README/NOTICE)
7. Repo trovabile via tag `rocmfpx`, `gfx1151`, `strix-halo`

## 8. Limitazioni dichiarate (nel model card)

- Speed benchmark only (no quality eval)
- Profilato per gfx1151 solo (non testato su altre GPU)
- MTP non attivato (plain inference). Eventuale attivazione MTP = lavoro futuro
- imatrix grug: 1/256 expert non coperto (normale MoE, impatto trascurabile)

## 9. Rischi e mitigation

| Rischio | Mitigazione |
|---|---|
| Upload fallisce per size | LFS resume; re-run |
| License metadata errata nel frontmatter | Verifica pre-upload: `license: apache-2.0` (grug), `license: mit` (Ornith) |
| Community confonde con stock-llama.cpp quant | Warning critico top + tag espliciti |
| Violazione attribution Apache-2.0 NOTICE | File NOTICE completo + dichiarazione derivative work |
| grug-think-v3-10k gated: utente non può replicare imatrix | **Non applicabile**: dataset pubblico Apache-2.0 non gated (verificato HF API). Community può scaricarlo liberamente per replicare la imatrix. Punto positivo per la replicabilità, dichiarato nel card. |

## 10. Artefatti sorgente (post-completamento, riferimento)

- grug: `~/llmodels/models/GRUG/grug-35b-v2-ROCmFP4-STRIX_LEAN.gguf` (17.32 GiB), `imatrix-grug-35b-v2.gguf` (183 MB), `mmproj-grug-35b-v2-f16.gguf` (857 MB)
- Ornith: `~/llmodels/models/ORNITH/Ornith-1.0-35B-ROCmFP4-STRIX_LEAN.gguf` (17.32 GiB), `imatrix_unsloth.gguf_file` (183 MB, → `imatrix.dat`), `mmproj-F16.gguf` (857 MB)
- Bench logs: `<lab-repo>/logs/bench-grug-t9.log`, `bench-ornith-vs-grug.log`

## 11. Apertura / decisioni differite

- **License kyuz0 toolbox**: **verificato 2026-08-11** — il repo `kyuz0/amd-strix-halo-toolboxes` NON ha file LICENSE (HTTP 404 su GitHub API sia su `/license` che su `LICENSE` raw). Default copyright "All rights reserved" si applica. **Decisione**: nel model card usiamo solo attribution come riferimento link GitHub (fair use), senza clausole che implicano ridistribuzione del container. Se kyuz0 aggiunge LICENSE in futuro, allineare NOTICE.
- **eval quality**: lavoro futuro, su input community
- **Riavvio llm-service produzione**: separato, decisione utente post-pubblicazione
