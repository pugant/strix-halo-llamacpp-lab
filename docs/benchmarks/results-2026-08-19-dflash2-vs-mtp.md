# Risultati — T7 A/B: DFlash2 (draft-dflash) vs MTP n6 · Qwen3.8-27B STRIX_LEAN

**Data:** 2026-08-19 · **Piano:** piano di esperimento interno `2026-08-19-dflash2-vs-mtp-ab.md` (non incluso nel repo)
**Setup:** Vulkan RADV gfx1151, GPU dedicata (llm-service fermo, restart+health a fine run),
LEAN 13.8 GB, `-c 16384 -fa on --jinja`, temp 0, p_min 0.75, p_split 0.10, warm-up scartato,
marker TREATMENT nei log (`logs/bench-dflash-vs-mtp/`). Script: `scripts/bench-dflash-vs-mtp.sh`.
**Porting:** branch `dflash2` (cherry-pick PR #27342 `ba2485545` + 4 fix `ebf1cc855`),
immagine `docker-llm-service:vulkan-fork-dflash2`; drafter `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`
(1.9B, 1.1 GB, block_size 8 → n-max 7).

## Tok/s (2 prosa it / 2 deterministici)

| Braccio | prosa 1/2 | det 1/2 | Δ det vs MTP6 |
|---|---|---|---|
| MTP6 (controllo, ckpt7) | 19.6 / 20.2 | 45.2 / 26.1 | — |
| **DF7** | 14.2 / 15.0 | **57.4 / 36.3** | **+27% / +39%** |
| DF5 | 17.5 / 15.9 | 52.2 / 39.5 | +16% / +51% |

**Det 57.4 tok/s = nuovo record per il LEAN su questo hw** (precedente 45.4).

## Acceptance (dal log, per prompt)

| Braccio | prosa (mean len) | det (mean len) |
|---|---|---|
| MTP6 | 2.48 (per-pos 0.93, 0.33, 0.13, …) | 6.26 / 3.94 |
| DF7 | 2.43 (per-pos 0.62, 0.33, 0.20, …) | **7.59** (0.99, 0.97, 0.96, 0.94, 0.92, 0.91, **0.90**) / 6.03 |
| DF5 | 2.29 | 5.80 (1.00, 0.98, 0.97, 0.93, 0.92) / 5.23 |

**Firma DFlash2 chiara**: su contenuto predittibile la acceptance per-posizione resta
≥0.90 fino alla posizione 7 (l'MTP crolla dopo la 1ª: 0.93→0.33). Su prosa italiana
l'accettanza è PARI all'MTP (2.43 vs 2.48) ma il round costa di più → tg −26/28%.

## Interpretazione fisica

- det: acceptance ~7.5 token/round × verify batch → tg quasi lineare nel numero di
  token accettati (SSM-dominato, KV 69.6 KB/token) → +27-39%.
- prosa: acceptance ~2.4 → il costo fisso del round DFlash (decode del noise block
  INTERO a 8 posizioni sempre + encoder/iniezione KV per token) non è ammortizzato
  → penalità −26/28% vs MTP che ferma il drafting AR quando la confidenza cala.

## Spot-check "lossless" (greedy, stesso prompt)

| Confronto | Esito |
|---|---|
| NONE vs MTP6 | divergenza reasoning a 422 char (comune) |
| NONE vs DF7 | divergenza reasoning a 422 char |
| MTP6 vs DF7 | coincidono fino a 1497 char |

La divergenza NONE-vs-spec allo STESSO punto per entrambi i drafter = **numerica batched
del verify** (target decodifica batch multi-token vs singolo → argmax flip su token quasi
pari), caratteristica PREESISTENTE dello stack (produzione MTP inclusa), NON un difetto
del porting DFlash2. Claim "lossless" di DFlash2 valido solo entro questo caveat.

## Verdetto (criteri piano)

**NO-GO alla sostituzione produzione** (prosa −26% >> soglia −3%), **MA**:
- record det assoluto (57.4), dominante su tutto il workload deterministico;
- filone aperto: drafter adattivo (DFlash per coding/counting/agentic-det, MTP per
  prosa) o n-max workload-dipendente; eventuale PR del porting a charlie.

---

## Addendum 1 — curva n-max DF3 + workload agentic (19/08 pomeriggio)

Piano di esperimento interno `2026-08-19-dflash2-df3-agentic.md` (non incluso nel repo), script
`scripts/bench-dflash-df3-agentic.sh` (non incluso), log `logs/bench-dflash-df3-agentic/`.
Stesso protocollo (GPU dedicata, temp 0, p_min 0.75, marker TREATMENT).

### Workload agentic (MTP6 vs DF7, prompt coding/JSON/log)

| Prompt | MTP6 | DF7 | Δ |
|---|---|---|---|
| coding (10 funzioni) | 27.5 | 33.8 | **+23%** |
| JSON (30 oggetti) | 35.0 | 36.2 | +3% |
| log (20 righe formato fisso) | 28.0 | 35.8 | **+28%** |

Acceptance mean: MTP 3.97-4.98 vs **DF7 5.63-5.90**; per-pos DF7 su agentic
(0.92, 0.77, 0.71, 0.65, 0.56, 0.52, 0.51) — resta >0.50 fino a pos 7 dove
MTP scende a 0.25-0.50. Il JSON è il caso già quasi saturo per MTP (floor).
**Criterio ≥+10% su 2/3 prompt: PASS → routing per workload confermato.**

### Curva n-max DFlash (prompt standard)

| Braccio | prosa 1/2 | det 1/2 |
|---|---|---|
| MTP6 | 19.6 / 20.2 | 45.2 / 26.1 |
| DF3 | 17.6 / 18.2 | 39.5 / 34.5 |
| DF5 | 17.5 / 15.9 | 52.2 / 39.5 |
| DF7 | 14.2 / 15.0 | 57.4 / 36.3 |

DF3 porta la prosa al quasi-pareto (−10%) MA il det1 scende sotto MTP (39.5 vs
45.2): il compromesso ottimale **singolo** drafter è **DF5** (det +16/+51%,
prosa −11/−21%). DF3 è dominato da DF5 (prosa pari, det molto inferiore).

### Conclusione T7 (finale)

La risposta alla "prosa" non è n-max ma **routing per workload**:
- prosa/chat → MTP6 (19.6-20.2)
- agentic/coding/det → DFlash2 (33.8-57.4, +23-39%)
- eventuale profilo unico senza routing → DF5.

Implementazione futura possibile: flag per-request nel fork o doppia istanza;
alternativa leggera: parametri server diversi per slot/servizio.
