# Reasoning Pressure (steering soft del budget reasoning) — Design Spec

**Data:** 2026-08-17 · **Filone:** ThinkingCap alla radice (redirect utente: anti-overthinking, NON budget floor)
**Contesto produzione:** fork ROCmFPX ckpt6 (patch 0001-0009), Qwen3.8-27B STRIX_LEAN, MTP n6, llm-service
**Report letteratura di riferimento:** `docs/research/2026-08-17-thinkingcap-root-literature.md` (non incluso nel repo)

## 1. Problema e obiettivo

Qwen3.8-27B non ha un segnale di stop forte nei pesi: sui task densi il reasoning
esplode e finisce TRONCATO dall'hard cap (dato A/B/B': 70-90% exhausted a level low).
La riga dichiarativa (L0) è NO-GO (àncora all'uso pieno: 70→80→90%).

**Obiettivo:** far sì che il modello chiuda il reasoning *da solo e bene* entro il
budget, tramite steering inference-time soft nel sampler — la famiglia L3 (guided
decoding) semplificata e training-free, senza predictor.

**Metrica di successo:** % exhausted (tagli) da 70-90% → **<30-40%** a parità di
qualità percepita (gate A/B). Il budget per-level (1024/2048/4096/8192 da pi) resta
la reference.

**NON obiettivi (espliciti):** budget floor / force-continue "Wait" (il modello non
under-thinka — decisione utente 17/08); interventi sui pesi (L4); predictor hidden-states
(Fase C paper-grade); ridurre i token dei task brevi (che oggi chiudono bene).

## 2. Design scelto — Approccio 1: estensione del sampler `reasoning-budget`

Tutta la logica vive nel sampler esistente (`common/reasoning-budget.cpp`, patch 0010
su branch `spec-cache-soft-wrap` = codice ckpt6). Nessun engine work: il sampler è già
nella chain del target → interazione MTP/cache ereditata dal forced-end in produzione.

### 2.1 State machine estesa

```
IDLE → (start_tag) → COUNTING → (end_tag naturale) → DONE   [invariato]
                         │ used ≥ pressure_start×budget (e UTF-8 complete):
                         │   FORCING_NOTA (nota deterministica token-per-token,
                         │                stessa logica accept/apply di FORCING,
                         │                a fine sequenza ↩ COUNTING, non DONE)
                         │ COUNTING + used ≥ squeeze_from (condizione DERIVATA,
                         │ NON uno stato: vedi §2.2)
                         ▼
                      [squeeze attivo in apply] → chiusura naturale o argmax → DONE
                         │ budget=0 (rete finale INVARIATA):
                         ▼
                      FORCING (forced-end 0005 + soft-wrap 0008/0009 di oggi)
```

- **SQUEEZE non è uno stato**: è una condizione derivata valutata in `apply()` quando
  `state == COUNTING` e `used ≥ squeeze_from`. Tutte le transizioni di COUNTING
  (end_matcher → DONE, remaining--, WAITING_UTF8/FORCING a 0, re-arm) restano le
  stesse — nessuna duplicazione.
- `FORCING_NOTA` è un nuovo stato che riusa la logica accept/apply di FORCING
  (sequenza = nota_tokens, masking -inf) ma a fine sequenza torna a COUNTING.
- La nota si inietta UNA volta per blocco think: flag `note_injected` azzerato dal
  re-arm su nuovo start_tag (insieme al ricalcolo delle soglie).
- **Guard UTF-8 al trigger della nota**: WAITING_UTF8 esistente copre SOLO
  l'exhaustion; il trigger della nota (75%) ha lo stesso bisogno → se l'ultimo token
  accettato è un pezzo UTF-8 incompleto, l'iniezione è differita al primo token
  completo (flag `note_pending`, stesso pattern del WAITING_UTF8 esistente).
- **Contabilità token della nota**: i token forzati della nota NON decrementano
  `remaining` (come i token del forced-end oggi) → `used = budget - remaining` misura
  solo i token pensati dal modello.

### 2.2 Formula squeeze (condizione in `apply`, stato COUNTING)

```cpp
// soglie in double, calcolate a init (e al re-arm):
squeeze_from = min(pressure_start * budget + grace, budget - 1);
// in apply, se state==COUNTING && used >= squeeze_from:
double x = (double(used) - squeeze_from) / (budget - squeeze_from);
x = std::min(std::max(x, 0.0), 1.0);        // clamp [0,1] (finestre degeneri)
const double boost = x * x * max_boost;     // ramp quadratico
// boost applicato al token end ATTESO DAL MATCHER:
const llama_token tok = end_tokens[end_matcher.pos];  // di solito == end_tokens[0]
// scan di cur_p per l'id `tok` (come fa il masking loop) e:
cur_p[i].logit += boost;                    // SOLO quel token
```

- `end_tokens` è la SEQUENZA tokenizzata dell'end tag (il matcher non assume un
  token singolo): il boost va al prossimo atteso (`end_tokens[end_matcher.pos]`);
  per Qwen3.8 `</think>` è tipicamente 1 token speciale, ma il codice non lo assume.
- Clamp doppio (`squeeze_from` e `x`) copre i budget piccoli: con budget=64 → nota a
  48, squeeze da 63 (finestra 1 token: di fatto nota → cap; comportamento degraduto
  accettato e documentato).
- Quadratico: a metà finestra solo +Δ/4; con `max_boost` ~9 l'end diventa argmax
  negli ultimi ~5-8% del budget. Se non chiude → FORCING a budget 0 (rete finale).
- **Attivazione solo con budget esplicito**: `reasoning_budget_tokens >= 0` PRIMA del
  filler `INT_MAX` (sampling.cpp:305): con budget illimitato + grammar_lazy lo
  steering è inerte (niente soglie a 1.6e9 né overflow int32 — soglie in double).

### 2.3 Nota contestuale

Sequenza forzata, tokenizzata UNA volta a init, deterministica:

```
\n\n[Budget notice: wrap up the reasoning and give the final answer]\n\n
```

- Visibile nel `reasoning_content` estratto (trasparenza; breve, inglese, parentesi
  quadre stile sistema). NON filtrata: il resend del client deve ricomporla identica
  per l'exact cache match (il filtro romperebbe il round-trip).
- Personalizzabile per-request (`reasoning_pressure_notice`).

## 3. Parametri e cablaggio

Per-request (pattern 0006/0008), fallback sui default server:

| Chiave body OAI | Default | Significato |
|---|---|---|
| `reasoning_pressure_start` | `0.75` | frazione del budget a cui scatta la nota |
| `reasoning_pressure_grace` | `200` | token di grazia post-nota prima dello squeeze |
| `reasoning_pressure_boost` | `9.0` | max_boost del ramp quadratico |
| `reasoning_pressure_notice` | testo §2.3 | nota iniettata |

- Attivazione: `reasoning_pressure_start > 0` E budget ESPLICITO
  (`reasoning_budget_tokens >= 0` prima del filler INT_MAX di sampling.cpp:305 — il
  check vive dove il sampler rbudget viene costruito, in modo che INT_MAX non attivi
  mai soglie). Con `start=0` → bit-per-bit il comportamento di oggi.
- **Rollout esplicito**: col default 0.75, lo steering è ATTIVO per ogni richiesta
  con budget esplicito una volta applicata 0010. Per l'A/B è il braccio ON; il
  default di produzione si decide col GO (switch immagine).
- CLI/env default server: `--reasoning-pressure-start/-grace/-boost/-notice` +
  `LLM_REASONING_PRESSURE_START/GRACE/BOOST/NOTICE` (pattern `LLM_REASONING_BUDGET`).
- Touch point: `common/reasoning-budget.{cpp,h}` (state machine, init estesa con i
  nuovi parametri, clone/reset completi), `common/common.h` (campi params sampling),
  `tools/server/server-common.cpp` (body OAI per-request, righe ~1178-1195),
  `tools/server/server-task.cpp` (tokenize nota + passaggio, zona ~496-525),
  `tools/server/server-context.cpp` (chat_params se necessario per il path slot),
  `common/common.cpp`/`common/arg.cpp` (CLI/env default).
- Patch duratura: `patches/reasoning-pressure/0010-reasoning-pressure.patch` —
  **0010 CONTINUA la numerazione della serie del branch `spec-cache-soft-wrap`**
  (0001-0009 in `patches/spec-cache-trailing-rollback/`), dir separata per la regola
  una-patch-per-feature; ricostruire ckpt6+0010 = serie 0001-0010.
- La patch 0010 NON modifica il comportamento di 0005-0009 (forced-end, soft-wrap,
  alias): con steering disattivato l'output è identico (T3 lo dimostra).

## 4. Interazioni da preservare (e verificare nei test)

- **MTP + clone/rollback (punto critico)**: nota e boost agiscono in `apply` a valle
  del verify exact; il path MTP clona il sampler e lo RIPRISTINA al reject parziale
  (server-context.cpp:3570 save / :3631 restore). Il `clone()` esistente NON copia
  nemmeno `force_pos` (gap latente, innocuo col forced-end a budget 0 dove i rollback
  sono rari, ma ESPOSTO dalla nota che vive a 75% del budget): 0010 DEVE estendere
  `clone()` a TUTTI i campi di stato (force_pos esistente + note_injected/note_pending/
  note_pos + soglie + `remaining` (base di `used`, oggi resettato a `budget` dal clone)
  + `start/end_matcher.pos` (il secondo è il TARGET del boost; oggi entrambi azzerati
  dalla clone)) e `reset()` ad azzerarli. L'enumerazione è normativa, non esaustiva:
  vale "tutti i campi" anche per campi futuri. Senza questo, un rollback mid-nota
  re-inietta/corrompe la sequenza e rompe il round-trip.
- **Ordine chain**: l'apply di rbudget gira PRIMA di grammar e sampler chain
  (sampling.cpp:631): masking e boost vedono il candidate set completo — è questo che
  rende il meccanismo funzionante; da affermare con commento inline nel codice.
- **Round-trip cache**: la nota è una sequenza deterministica nel reasoning → il
  resend del client la contiene → exact match. Verifica T1 obbligatoria (pattern t1
  del piano 2026-08-15, `scripts/diff-prompt-cache.py` — script non incluso nel repo).
- **Soft-wrap**: se il modello chiude dopo nota/squeeze → DONE naturale, zero
  exhausted; il messaggio wrap 0008 resta solo per il caso forced-end.
- **Re-arm multi-blocco**: nuovo `<think>` azzera nota/squeeze per quel blocco.
- **UTF-8**: guard dedicata al trigger nota (§2.1, `note_pending`) — NON ereditata
  da WAITING_UTF8 che copre solo l'exhaustion.

## 5. Esperimento e gate GO/NO-GO

### 5.1 Sanity tecnici (container test :1235, GPU dedicata pre-autorizzata)
- **T1 round-trip**: task denso con budget che fa scattare nota+chiusura → resend
  verbatim content+reasoning → NESSUN cold fallback (exact hit). Include il caso
  rollback-MTP mid-nota (log -lv 5: cercare restore/checkpoint attorno all'iniezione).
  Strumento: pattern t1/t3 esistenti.
- **T2 efficacia locale**: 3-4 task densi con budget 2048 E con budget 1024 (la
  finestra più stretta di produzione: 56 token di squeeze), -lv 5 → verificare nei
  log: nota emessa al 75%, chiusura naturale entro grace o in squeeze, zero frasi
  mozzate, round MTP corti confinati alla finestra squeeze.
- **T3 non-regressione**: `reasoning_pressure_start=0` → output bit-identico a ckpt6
  (stesso seed/prompt, confronto contenuti).

### 5.2 A/B su task reali (gate qualità)
- Bracci: OFF (ckpt6) vs ON (ckpt6+0010), stessi task/level del manifest A/B/B'
  (27 task riusabili), estrazione con `scripts/ab-bprime-extract.py` adattato (script non incluso nel repo).
- Metriche primarie: % exhausted per level, token reasoning medi/sd, wall per turno.
- Gate qualità: confronto blind su campione 10-15 task a level low (massima pressione
  sui tagli): ON ≥ OFF; zero artefatti da chiusura (frasi mozzate, loop noti).
- **GO produzione**: exhausted <30-40% + gate qualità + T1-T3 verdi → switch immagine
  llm-service con consenso utente. **NO-GO**: documentare (curva troppo debole o
  qualità degradata) — restano hard cap + soft-wrap attuali.

## 6. Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Il modello ignora la nota (chiusura non anticipata) | T2 misura l'efficacia prima dell'A/B; boost/slope parametrizzati |
| Qualità degradata da chiusura forzata anticipata | gate blind A/B; ramp quadratico tardivo |
| Cache round-trip rotto dalla nota | T1 obbligatorio pre-A/B; nota deterministica |
| Interazione MTP inattesa nella finestra squeeze | sanity verify nel T2 (round corti attesi solo lì); eredità forced-end |
| Plateau/degrado stile "troppi Wait" (ICLR26) | UNA sola nota per blocco, mai ripetuta |
| Effetto àncora (come L0) | nessun numero dichiarato al modello: la nota menziona il comportamento, non il budget residuo |

## 7. Out of scope

Floor/force-continue Wait; predictor hidden-states (Fase C); interventi pesi;
steering sui task brevi; modifica del forced-end/soft-wrap esistenti.
