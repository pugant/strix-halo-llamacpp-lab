# Spec — T7-f2: Routing drafter per workload (MTP ↔ DFlash2) nel fork ROCmFPX

**Data:** 2026-08-19 · **Stato:** approvata in co-design con l'utente (sessione brainstorming)
**Contesto:** T7 DFlash2 (report `docs/benchmarks/results-2026-08-19-dflash2-vs-mtp.md` +
Addendum 1) ha chiuso: DFlash2 vince su workload deterministico/agentic (+23-39%),
perde su prosa libera (−26%). Questa spec definisce il routing: **un solo server, un
solo modello target, due drafter attivi, scelta per request**.

## 1. Obiettivo e requisiti (bloccati con l'utente)

Far girare la produzione (llm-service) con il drafter giusto per workload:
- prosa/chat → **MTP n6** (19.6-21.0 tok/s; penalità DFlash −26%)
- agentic/det → **DFlash2 n7** (33.8-57.4 tok/s, +23-39%)

**Requisiti vincolanti:**
- R1 — **Client-agnostic**: qualsiasi client OAI (curl, webui, pi) funziona senza campi
  custom. Ripiego accettato ma non richiesto: flag pi-only. Campo body opzionale
  `spec_drafter` come override esplicito per power-user.
- R2 — **Un solo modello target in esecuzione** (niente doppia istanza/due pesi).
- R3 — Criteri di successo produzione: (1) prosa ≥ −3% vs MTP6; (2) agentic ≥ +10%;
  (3) **zero regressioni cache** su fix 0005-0009 (cold-fallback/rollback/checkpoint);
  (4) curl senza campi custom = stesso comportamento; (5) rollback = switch immagine,
  nessuna ritrattazione di codice.
- R4 — Destinazione: produzione llm-service (switch con approvazione finale utente).
- R5 — Scopo interno all'harness: NON materiale da condividere in community.

## 2. Evidenza sperimentale (a supporto delle decisioni)

**T0 (gating policy, eseguito 19/08)** — report
`docs/benchmarks/results-2026-08-19-t0-reasoning-acceptance.md`, log
`logs/bench-t0-thinking/`:
- Il **thinking NON è classe-prosa**: acceptance DF7 su reasoning 4.06/4.85/5.27
  (prosa 2.39, det 7.59); tok/s quasi-parità (−2%/+14%/−5%, media +2.4%).
- → **Politica a request intera**; phase-aware (switch a end_tag) SCARTATA su dati
  (comprerebbe ~0-4% al costo di switch mid-generation + re-encode + tagging
  checkpoint). Fallback documentato, non implementato.

**T7 A/B** — DF7 agentic: coding +23%, log +28%, json +3%; det record 57.4.
**Community** — NESSUN multi-drafter per-request pronto: verificato in codice
llama.cpp master (un solo `mparams`/`ctx_dft`; `common_speculative_init_result` =
RAII factory per UN draft model), vLLM/SGLang (un drafter per deployment), SpecForge
(arXiv 2603.18567 = training framework EAGLE-3, nessun routing).

## 3. Architettura — dual-drafter in un solo `common_speculative`

Il framework impl del fork già contiene: vettore di impl per `common_speculative`,
stato per-seq (`dparams` + vettori `[n_seq]`), chain per priorità con flag `drafting`.
La surgery aggiunge il plumbing per due impl **draft-model** (MTP + DFlash) e la scelta
per-seq. **Nessun drafter nuovo**: gli impl MTP e DFlash sono quelli esistenti e testati.

**Attivazione CLI** (nessun vettorizzare mparams — il nostro caso è esattamente
"un draft model esterno + MTP-nextn nel GGUF target"):
```
--spec-type draft-mtp,draft-dflash --spec-draft-model /llmodels/QWEN3.8/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
--spec-draft-n-max 7 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10 --spec-draft-ngl all
```
Regola: il tipo `draft-mtp` NON consuma file (usa i nextn del target); il primo tipo
con draft-model esterno consuma `--spec-draft-model`. Un solo `--spec-type` = modalità
mono → policy identità, **zero cambiamenti di comportamento** (R3.4, R3 rollback).

### 3.1 I sei tocchi chirurgici (~250-400 LOC totali)

**T1. Context — `tools/server/server-context.cpp` `load_model` (~80 LOC).**
Oggi if/else-if mutualmente esclusivo (915-961) crea un solo ctx_dft (membri singleton
731-732). In duale: due cparams e due ctx:
- `ctx_dft_mtp`: sul **modello target**, `ctx_type=LLAMA_CONTEXT_TYPE_MTP`,
  `ctx_other=ctx_tgt` (ramo esistente 939-961 per nextn);
- `ctx_dft_dflash`: sul **modello DFlash** (load dal file, ramo esistente 872-933),
  `ctx_other=ctx_tgt`, causal-off + estrazioni layer (flag del ctor DFlash,
  speculative.cpp:1121-1128).
Le estrazioni sul target (pre_norm MTP a 1557-1558, layer_inp DFlash) sono **additive**
e coesistono. Membri aggiuntivi: `model_dft_dflash`, `ctx_dft_mtp`, `ctx_dft_dflash`.

**T2. Config per-impl — `common/speculative.cpp` `common_speculative_init` (2941-3010,
~40 LOC).** Il config copiato per impl (73-79) oggi porta l'unico `ctx_dft`; diventa:
config MTP → `ctx_dft_mtp`, config DFlash → `ctx_dft_dflash` (campo nuovo o
`ctx_dft_by_type`). Impl istanziati come oggi (3029, 3033). **Guard per-tipo
(obbligatorio, trappola verificata in 2951)**: l'impl DFlash si costruisce SOLO se
`ctx_dft_dflash != nullptr` (oggi `has_draft_dflash = enabled && ctx_dft != nullptr`
costruirebbe un impl DFlash sul ctx MTP del target se il path del modello manca);
simmetricamente l'impl MTP solo se il proprio ctx esiste. Un tipo il cui modello/ctx
non è disponibile viene **droppato al boot con WARNING** e il server parte mono
(vedi §5 riga 1: questo fallback è codice NUOVO).

**T3. Gating per-seq — `common/speculative.h:32-69` + `speculative.cpp` (~60 LOC).**
Nuovo campo `common_speculative_type drafter` in `common_speculative_draft_params`.
Gating asimmetrico, motivato dal funzionamento della cache (vedi §4):
- **`draft()` gated per entrambi** (loop 3173-3250): solo l'impl attivo per quella seq;
- **`process()` DFlash gated** (3131-3143): l'encoder + iniezione KV è il costo reale
  e inquina il ctx DFlash — mai per seq MTP;
- **`process()` MTP UNGATED per tutte le seq**: la cattura della boundary row
  (`pending_h`) è economica ed è ciò che rende disponibile lo stato richiesto dal
  save della prompt cache (server-context.cpp:149-152; `state_required` è globale e
  MTP è l'unico stateful). Gatarlo per seq DFlash-routed → stato mai catturato →
  `prompt_save` salta l'entry intera → le conversazioni agentic perdono la cache
  (violazione R3.3, trovato in review).
`impl_last[seq_id]` (3211-3225) attribuisce `accept()` all'impl attivo.
**Switch SOLO a confine task** (mai mid-generation): il valore si setta all'avvio del
task e non cambia. Effetto collaterale noto e accettato (da review it.2): col process
MTP ungated, le batch delle seq DFlash-routed vengono specchiate anche nel KV di
`ctx_dft_mtp` — neutro-positivo (il ctx MTP resta caldo per futuri task MTP sullo
slot; le righe stale sono coperte dal trim T6).

**T4. Sizing a max (~10 LOC).** `n_rs_seq` e `n_batch`/`n_ubatch` calcolati con
`n_draft = max` = 7 (il commento upstream in codice anticipa esattamente questo).
Per-impl gli n-max effettivi restano auto-clampati (MTP→6 via chain_heads 1566-1567;
DFlash→7 via block_size 1082-1088). I check `n_rs_seq` (server-context 862, 1032-1037)
leggono il globale → passano.

**T5. Politica — `tools/server/server-task.cpp` (~30 LOC).** Al parse del body
(accanto a `speculative.n_min/n_max/p_min` esistenti a 313-322; embrione `#if 0` a
324-336): `tools`/`tool_choice` non-vuoti → `drafter=DFLASH`, altrimenti `MTP`
(default conservativo: misroute prosa→DFlash costa −26%, l'inverso zero).
Override esplicito opzionale `spec_drafter` ∈ {`mtp`,`dflash`,`auto`} dove
**`auto` ≡ assente** (= decide la policy). Valore sconosciuto → 400 chiaro.

**T6. Trim/checkpoint entrambi i ctx (~10 LOC).** Il punto che trimma `ctx_dft`
(server-context.cpp:2583, `common_context_seq_rm` oltre `ckpt.pos_max`) trimma anche
l'altro ctx: le righe stale oltre il confine accettato non devono sopravvivere a un
futuro switch (collisioni di posizione).

### 3.2 Flusso dati

```
POST /v1/chat/completions (body standard OAI)
  → server_task_params_from_json: drafter = override | policy(tools)
  → get_available_slot: INVARIATO (LCP/LRU sul KV target — drafter-agnostic)
  → update_slots: dp per-slot {n_max, n_min, p_min, drafter} (zona 2535-2551)
  → common_speculative_draft: SOLO l'impl attivo per quella seq
  → common_speculative_process: MTP sempre (boundary per la cache), DFlash solo se attivo
  → verify batch: invariato — l'argomento di invarianza numerica è PER-SEQ
    (ogni seq è verificata contro i token del proprio singolo drafter)
  → checkpoint/trim: entrambi i ctx_dft; spec_state = blob MTP come oggi
```

**n_parallel**: produzione gira `--parallel 4 --kv-unified` (start-llama-server.sh,
ramo Qwen3.8-27B) → slot su drafter diversi nello stesso round è scenario vivo. Il
meccanismo per-seq di T3 lo copre. Il check strict-qwen richiede np=1 e in produzione
NON è usato: non attivarlo.

### 3.3 YAGNI deliberato (fuori scope, documentato)

- Generalizzazione a N draft model esterni; porting di `common_speculative_init_result`
  da master; slot-affinity per drafter in `get_available_slot` (1281-1330; solo se la
  telemetria mostra switch-cost rilevante); routing adattivo su acceptance; phase-aware
  (scartato su dati T0).

## 4. Cache e checkpoint (criterio R3.3)

Principio: **il KV target è drafter-indipendente** ed è la parte che vale il 91% di
prefill risparmiato; lo stato lato draft ha già path di degradazione gradata.

1. **Tag dell'entry** (`prompt_save`, server-context.cpp:131-155): campo `drafter`
   nell'entry della prompt cache RAM. La parte "dft" dell'entry = stato KV del ctx
   del **drafter attivo** (la chiamata `prompt_cache.save(prompt, ctx_tgt, ctx_dft, …)`
   a 154 passa oggi il singleton: passa il ctx del drafter attivo). **Nota (da review,
   NON "pulire")**: le entry taggate dflash porteranno comunque il blob spec MTP
   (peso morto, ~KB) perché il gate di save richiede lo stato dell'impl stateful —
   MTP process è ungated proprio per questo (§3.1 T3).
2. **Load** (`prompt_load`, 157-186): KV target ripristinato **sempre**; parte dft +
   blob spec solo a tag corrispondente.
3. **Cambio semantico chirurgico** (l'UNICA modifica alla semantica delle fix
   0005-0009, in direzione MENO punitiva): oggi "entry senza spec su server
   MTP-required → `res=false` → rigenerazione fredda" (163-169). Con il routing il
   full-cold si restringe alla sola mancata corrispondenza del **target**; mismatch
   drafter → target ok, draft ricostruito: MTP = resync con boundary nulla (un draft
   scartato, path esistente 1716-1758); DFlash = nessun re-encode del prefix esiste
   nell'impl (verificato in implementazione 19/08: `begin()` ha solo il warning
   pos_max<N-1, il ctx riceve solo i delta) → prefix mancante = draft degradati,
   output corretto (verify sul target). [Emendamento 19/08 sera: il kind di log
   onesto è `dflash-prefix-miss`, vedi §6.]
4. **Checkpoint-salvage 0007/0009**: MTP resta l'unico impl stateful (DFlash non fa
   override di `get_state` — default 190-192) → `common_speculative_get_state`
   first-wins (3284-3296) resta corretto. Il checkpoint (`data_spec`, 2020-2061;
   snapshot 2520-2529) si etichetta col drafter attivo (con switch solo a confine
   task, un task = un drafter, nessun caso mid-flight). Trailing-rollback (2816-2907):
   trim tgt + entrambi i ctx (2833-2838) + rollback_state come oggi.
5. **Cache idle (2113-2125) e disk cache (server-task.cpp:2447-2455, 3016-3105)**:
   stessa regola del tag (disk cache non abilitata in produzione: coperta per
   correttezza, test leggeri).
6. **NON cambia**: LCP/LRU, context-shift (2419-2424), forced-end/soft-wrap 0008/0009.

## 5. Error handling (ogni fallimento degrada allo status quo, mai crash)

| Fallimento | Comportamento |
|---|---|
| Drafter esterno manca/corrotto al load | **Fallback di boot NUOVO**: drop del tipo dflash + WARNING + partenza mono MTP-nextn. NB: oggi il comportamento è `SRV_ERR` + server che NON parte (server-context.cpp:886-890) — questo fallback è codice aggiuntivo volontario, col guard per-tipo di §3.1 T2 |
| Modalità mono (un `--spec-type`) | Policy identità, zero path nuovi |
| `spec_drafter` con valore non in enum | 400 con enum chiaro |
| `spec_drafter=dflash` su server mono-MTP (dflash droppato al boot) | 400 esplicito "drafter dflash not loaded; active: mtp" — MAI fallback silenzioso |
| Mismatch tag al load cache | Target ripristinato, draft ricostruito, log INFO conteggiato |
| DFlash caricato, 0 request lo usano | ~1.1 GB RAM, zero costo per round (gating salta process/draft) |
| Misroute tools→prosa | −26% su quella response sola; cache target intatta |
| Guasto impl DFlash mid-task | Come oggi un drafter rotto: abort task, slot reset |

**Invarianti (asserzioni + test T1-T3)**: un solo impl attivo per seq per round;
`impl_last[seq]` = impl attivo; switch solo tra task; trim di entrambi i ctx a ogni
checkpoint.

## 6. Osservabilità (richiesta esplicita utente per troubleshooting)

Marker grep-abili `spec-route:` a livello INFO (3):
```
spec-route: dual mode active: draft-mtp (nextn, n_max=6) + draft-dflash (<file>, n_max=7)
spec-route: task <id> seq <n>: signal=tools|none|override:<val> → drafter=<mtp|dflash>
spec-route: cache tag mismatch (entry=<mtp>, active=<dflash>) seq <n>: target restored, draft rebuild=dflash-prefix-miss (<P> tok)
```
[Emendamento 19/08 sera: kind `dflash-prefix-miss` al posto di `dflash-reencode` —
nessun re-encode del prefix esiste nell'impl; P = righe KV mancanti nel ctx attivo.
`mtp-resync` invariato.]
```
Gratis col dual-load: `statistics draft-mtp:` e `statistics draft-dflash:` per-impl
(acceptance/calls separati) + `slot print_timing` per-task. Contatori Prometheus su
`/metrics`: `spec_route_requests_total{drafter}`, `spec_route_override_total`,
`spec_route_cache_rebuild_total{kind}` (cumulativi: delta tra letture). **In mono**
(incluso boot con fallback): i contatori restano registrati con label `drafter`
fissata all'unico tipo attivo — la policy è identità e ogni request conta lì.
Nessun log nuovo oltre il livello 3; body resta a 5 in finestre brevi (regola nota).

## 7. Testing (gate in sequenza)

- **T1 smoke dual-load** (container :8090): entrambi gli impl caricati; request senza
  tools → marker acceptance `draft-mtp`; con tools → `draft-dflash`; conversazione con
  classi alternate → nessun abort, prefill delta ~0.
- **T2 cache round-trip drafter alternato**: 4 turni con switch di classe a ogni
  turno; gate: zero cold-fallback del target, rebuild draft attesi, prefill salvato ≥
  soglia. **Certifica R3.3.** Da eseguire sia su container :8090 semplice SIA con la
  config di produzione completa (`--parallel 4 --kv-unified --cache-ram 65535`,
  start-llama-server.sh ramo Qwen3.8-27B) — il parallelismo è scenario vivo, non
  teorico.
- **T3 regressione percorsi sacri**: scenari 0006-0009 (budget-forced end, resend
  alterato, trailing rollback, checkpoint restore) in duale con entrambe le classi;
  esiti = mono. Stessa doppia configurazione di T2.
- **T4 A/B routing vs mono** (GPU dedicata, piano .md dedicato): 6 prompt T7 + 3
  agentic; bracci MTP6-only vs DUALE(policy). Gate R3.1/R3.2 (prosa ≥ −3%,
  agentic ≥ +10%).
- **T5 spot-check numerica**: greedy, DUALE-MTP vs ckpt7 e DUALE-DFlash vs DF7-oggi →
  divergenza entro il caveat batched-verify noto.

## 8. Rollout produzione

1. Branch `drafter-routing` da `dflash2` (base = main 0a59add + 0008/0009 + dflash2
   3 commit). Patch duratura `patches/drafter-routing/` una-patch-per-feature dopo
   T1-T3.
2. Immagine `docker-llm-service:vulkan-fork-dflash2-route` (~7 min).
3. T4/T5 su :8090 GPU dedicata.
4. Switch llm-service (SOLO approvazione utente): config come §3, backup
   `.bak-20260819-ckpt7`, rollback = immagine precedente.
5. Osservazione 24h su `spec-route:` + contatori.

**Fuori scope/follow-up**: PR a charlie (porting DFlash2 + routing), push branch fork
GitHub, affinità slot, adattivo, phase-aware.

## 9. Vincoli di lavoro (per l'implementatore)

- **Leggere AGENTS.md del repo ROCmFPX prima di toccare codice** (regola repo).
- llm-service = produzione gestita dall'utente: FERMO da 19/08 12:39 su richiesta
  utente per questo lavoro; restart solo a fine lavoro o su indicazione. Health check
  via `docker ps` o IP container (NON host :1234).
- Patch durature in `patches/drafter-routing/` (format-patch --stdout con nome
  esplicito); commit locali sul branch per il lavoro intermedio.
- Ogni esperimento GPU con piano .md; marker TREATMENT nei log; p_min sempre esplicito.
- Modelli in `~/llmodels/` mai toccati. Build SOLO da Dockerfile.vulkan-rocmfpx.
- Clone di lavoro: `<lab-repo>/ROCmFPX` (remotes: origin=charlie,
  fork=pugant GitHub; push solo con approvazione).

## 10. Riferimenti rapidi (anchor codice, branch dflash2 — precisione ±2 righe)

common.h:312-315 (draft.mparams singolo), :349 (types vettore) · arg.cpp:3647-3656
(--spec-type lista) · server-context.cpp:731-732 (singleton), 862/1032-1037 (check
n_rs_seq), 872-961 (load draft), 915-924 (mutua esclusione), 1070-1072 (assegnazione
slot.spec/ctx), 1281-1330 (get_available_slot), 131-186 (prompt_save/load), 2020-2061
(checkpoint), 2113-2125 (cache idle), 2456-2557 (update_slots loop), 2535-2551 (dp),
2583 (trim), 2816-2907 (trailing rollback) · server-task.cpp:313-336 (spec params
body), 495-524 (pattern 0008), 2447-2455/3016-3105 (disk cache) · speculative.cpp:48-56
(effective n_max), 73-79 (config copy), 975+ (draft_dflash), 990-1006 (stato DFlash),
1082-1088 (clamp block_size), 1121-1128 (flag ctx DFlash), 1456-1495 (stato MTP),
1557-1558 (flag pre_norm), 1716-1758 (resync MTP), 2941-3010 (init), 3029/3033 (impl
ctor), 3131-3143 (process all), 3173-3250 (draft loop), 3211-3225 (impl_last),
3284-3296 (get_state first-wins).
