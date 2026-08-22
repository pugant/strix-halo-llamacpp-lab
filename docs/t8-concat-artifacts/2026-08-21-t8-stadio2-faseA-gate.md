# Piano — T8 Stadio 2, Fase A gate: condizionamento DFlash-su-prefix-MTP (Task 8)

Data: 2026-08-21. Repo `ROCmFPX/` branch `t8-concat` tip `ae6bad9cf`; immagine
`docker-llm-service:vulkan-fork-t8-concat` (`e13264592171`, doppio-tag
`vulkan-fork-dflash2-route`, bit-identica a `cfb43d5c9ea9`). GPU libera
(container llm-service assente, verificato 2026-08-21).

## 1. Ipotesi e domanda

Il round concat (k1=1) condiziona il blocco DFlash su un prefix di 1 token head
MTP *non ancora confermato dal target*. Se il condizionamento degrada la
qualita' del blocco (perche' l'head e' sbagliato, o perche' il drafter DFlash
non e' allenato su context con token MTP), l'acceptance del segmento DFlash
scende e il guadagno di colonna si perde. Domanda del gate: **di quanto?**
Rischio dichiarato spec §9: ±12-20%% mai misurato.

## 2. Definizione metrica (spec §7 VERBATIM, non riformulata)

> sia `acc_concat` la frazione media di token accettati per posizione sul
> segmento DFlash effettivo dei round concat (posizioni k1+1..fine del round,
> normalizzata sul segmento presente in ogni round per l'early-stop p_min), e
> `acc_df7` la stessa metrica sul segmento equivalente di una run DF7 pura
> (concat OFF). Il gate e' `acc_concat >= 0.90 x acc_df7` (-10% RELATIVO).

Operazionalizzazione (dai formati reali degli strumenti, verificati nel codice):

- Round = riga `R,<draft_len>,<n_acc>` del SPEC_VERIFY_LOG (sampling.cpp:792):
  `draft_len` = token effettivamente draftati nel round (con eventuale
  troncamento di budget a fine request), `n_acc` = accettati contigui da
  posizione 0 (`n_acc <= draft_len` sempre, verify exact greedy).
- **Segmento DFlash effettivo**: braccio A (round concat, k1=1): posizioni
  0-indicizzate `1..draft_len-1` del round (la 0 e' l'head MTP); braccio B:
  `0..draft_len-1` (tutto il draft). Lunghezza presente `L = draft_len - k1`.
- **Token accettati sul segmento** per round: `a = clamp(min(n_acc, draft_len) - k1, 0, L)`
  (braccio B: `a = n_acc`). Un round concat con head rifiutata (`n_acc = 0`)
  contribuisce `0 / L`: e' l'effetto-condizionamento che il gate misura per
  definizione (il blocco era condizionato su quell'head).
- **Normalizzazione early-stop**: denominatore = somma delle `L` dei round
  (ogni round contribuisce solo le posizioni presenti; per DFlash2 la larghezza
  e' piena tranne le code di budget della request, identiche nei due bracci).
- `acc = (somma a) / (somma L)` sui round del braccio; per-posizione relativa
  al segmento `j = 1..7`: den_j = #round con `L >= j`, num_j = #round con
  `n_acc >= k1 + j`.

## 3. Bracci

Stesso server, stessa immagine, cambia SOLO il flag concat (fairness T7):

- **A CONCAT k1=1**: boot duale T7 + `--spec-concat-k1 1`.
- **B DF7-puro**: boot duale T7 SENZA flag (default k1=0; T4d: k1=0 == assente).

Boot comune (protocollo dupbuild/T7 verbatim): `-m Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf
-ngl 999 -fa on --jinja -c 16384 --spec-type draft-mtp,draft-dflash
--spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-draft-ngl all
--spec-draft-n-max 7 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10`,
porta 1237. Strumentazione identica nei due bracci: `-lv 4` (TRACE, vedi §5),
`-e LLAMA_TRACE=1`, `-e SPEC_VERIFY_LOG=/tmp/spec-verify-fasea.log`.
Niente `--metrics` (nessun gate Prometheus qui).

## 4. Richieste (VERBATIM, 3 identiche nei due bracci)

Corpi JSON formato `bench-routing-vs-mono.sh::bench_tok` (righe 63-65):
`{"model":"q","tools":<TOOLS>,"max_tokens":<MT>,"temperature":0,"messages":[{"role":"user","content":"<PROMPT>"}]}`.

- `TOOLS` = payload get_time minimal, verbatim da `bench-routing-vs-mono.sh:24`.
- **D1** (det, max_tokens 600) = `bench-routing-vs-mono.sh:29` / `bench-dupbuild.sh:17`:
  `Conta da 1 a 200, un numero per riga, solo i numeri.`
- **AG1** (agentic, max_tokens 700) = `bench-routing-vs-mono.sh:32`:
  `Scrivi dieci funzioni Python molto brevi, una per riga: somma, sottrazione, moltiplicazione, divisione, modulo, potenza, minimo, massimo, valore assoluto, arrotondamento. Ogni funzione su una riga con def e return.`
- **AG2** (agentic, max_tokens 700) = `bench-routing-vs-mono.sh:33`:
  `Genera un array JSON di trenta oggetti utente: id progressivo da 1, nome user_N, email user_N at example.com, attivo true, punteggio N virgola 5. Solo JSON valido, senza commenti.`

(AG1+AG2 = prime due agentic standard T7 in ordine canonico; AG3 escluso per
tempo, stessa classe.) Seed: nessun campo seed nel body (protocollo T7
identico); temp 0 = greedy, il seed e' inerte per il target e per il selettore
DFlash2 (usato solo a temp>0) — dichiarato, non variabile nascosta.

Warm-up scartato (baseline strumenti): `Rispondi solo OK.` max_tokens 50
temp 0 senza tools (route mtp), come dupbuild.

## 5. Strumentazione e tagging dei round concat

- **SPEC_VERIFY_LOG** (path nel container): righe `R` (round: draft_len,
  n_acc) + `P,k,dft,amx,smp,p_dft,H` (una per riga del batch verify, anche
  OLTRE lo stop esatto — controfattuale p_dft). Solo overload senza-dists
  (temp 0): confermato da T4/T6.
- **`-lv 4` (TRACE, non 5)**: abilita l'UNICA riga TRC per-round-composto
  `spec ...: concat round composed, k1=1 (head=N tokens, seq S)`
  (speculative.cpp:3789; la prima e' INF) SENZA il rumore DBG (candidati per
  token, backend). Volume TRC in speculative.cpp = 1 call-site.
- **`LLAMA_TRACE=1`**: righe server per-round `accepted X/Y draft tokens`
  (+ variante `(restore checkpoint)`), server-context.cpp:4157/4253 — un
  anchor per ogni round verificato = per ogni riga R.
- **Tagging composed** (walk del log per finestra request): flag=false;
  riga `concat round composed` -> flag=true; riga `accepted ... draft tokens`
  -> il round corrente e' composed sse flag; flag=false. Ordine garantito
  dal flusso single-slot: head-pubblicata (draft) precede la verify dello
  stesso round.
- **Cross-check**: (1) #anchor == #R per finestra (hard); (2) #round composed
  == `rounds` nel marker per-request `statistics concat: k1 = 1, rounds = N,
  mtp_accepted = A/T` (braccio A); (3) braccio B: zero marker concat e zero
  tag composed; (4) delta cumulativo `statistics draft-dflash: #gen drafts`
  == #R per finestra; (5) righe `restore checkpoint` contato (atteso 0, RS
  ring fix T5; se >0 la run viene segnalata in report); (6) sanity P: argmax
  == sampled dove sampled>=0 (greedy intatto, >0.99).

## 6. Protocollo run

**Step 0 (probe strumenti, ~4 min, BOOT dedicato poi rimosso)**: `bench-t8-fasea.sh
probe` — boot braccio A (flag k1=1) + warm-up + 1 request det corta CON tools
(`Che ore sono? Usa lo strumento, poi elenca tre attivita da fare oggi.`,
max_tokens 80, temp 0). Verifiche: (a) speclog con righe R; (b) #righe
`accepted X/Y draft tokens` (LLAMA_TRACE) == #R; (c) almeno un marker TRC
`concat round composed` (-lv 4) e marker `statistics concat`; (d) volume log
accettabile; (e) routing signal=tools -> dflash. Se una qualsiasi fallisce ->
BLOCKED prima del protocollo (strumento rotto, non meccanismo). Il probe NON
entra nelle metriche (container `fasea-probe`, artifacts in `fasea/probe/`).

3 run per braccio, **fresh boot per run** (run = warm-up + D1 + AG1 + AG2 in
sequenza), bracci interleaved per run (ordine A,B fisso) per bilanciare drift
termico. Motivo del fresh boot: a temp 0 le repliche same-boot sono
pseudo-repliche (greedy identico); la variabilita' dominante documentata e'
boot-to-boot (floor near-tie: D1 bimodale 59/76 token, memoria T4) e il gate
vuole mean±stddev su quella. Attesa: ~6 boot x (60-120s load + 4 request
brevi) ≈ 20-25 min totali.

Artifacts in `logs/test-t8-concat/fasea/{armA,armB}-run{1,2,3}/`:
`resp-{warm,D1,AG1,AG2}.json`, snapshot cumulativi `speclog-cum-<n>-<id>.log`
e `srv-cum-<n>-<id>.log` (finestre ricavate a analisi per conteggio righe),
`srv-full.log` finale. Marker `TREATMENT=fasea-<arm>-run<r>` in ogni riga
TSV/console e negli header dei TSV. Container `fasea-*` rimossi da trap.

TSV di riepilogo `fasea-summary.tsv`: per request tok/s
(`timings.predicted_per_second`), prompt/completion tokens, routing marker.

## 7. Analisi e gate

1. Parse finestre per request x run x braccio -> round tagged.
2. `acc` per braccio: pooled su tutte le run/request (primario) + per request
   (mean±stddev sulle 3 run) + per run.
3. Per-posizione j=1..7 (relativa al segmento): acc_j, profilo ratio
   A/B — lettura del condizionamento (prime posizioni dopo l'head vs coda).
4. Decomposizione (readout, NON gate): head-acceptance (share round composed
   con n_acc>=1; confronto con `mtp_accepted` del marker) e acc del segmento
   condizionata a head accettata; profilo controfattuale `p_dft` medio per
   posizione (non troncato dallo stop esatto — isola il condizionamento dalla
   troncatura).
5. **Gate (proposta, decisione UTENTE)**: `acc_concat >= 0.90 x acc_df7`.
6. tok/s concat vs puro: informativo SOLO (strumentazione -lv 4+trace attiva
   in entrambi i bracci, confronto interno valido).

## 8. Fermate e criteri di blocco

- Server fail / health KO dopo 240s -> run segnato, ripetuto una volta.
- Mismatch #anchor vs #R o composed vs marker -> la run e' ANOMALA: analisi si
  ferma, si indaga (possibile bug meccanismo -> BLOCKED, niente patch).
- Divergenza qualitativa inattesa (es. round R>8 in braccio B) -> BLOCKED.
- Niente modifiche di codice, niente commit: solo docs+scripts.

## 9. Output

- Report: `docs/research/2026-08-21-t8-fasea-conditioning.md` — tabella
  per-posizione, acc mean±stddev per request e aggregato, verdetto gate
  CALCOLATO ma marcato "proposta — decisione utente", lettura condizionamento,
  tok/s informativo.
- Esito all'utente: PASS -> Fase B (Chunk 2, sessione separata); FAIL ->
  Stadio 2 si ferma (report causa + memoria).

## 10. Emenda 22/08 (decisione utente opzione c): verifica AG content-controlled

**Motivazione.** Il run principale ha isolato un confound: a temp 0 le
repliche same-arm sono bit-identiche ma AG1/AG2 divergono TRA bracci (near-tie
f32 capovolti dalla struttura di batch 9 vs 8 colonne; prima divergenza nel
reasoning a char 410/1034). Il pooled §7 (0.9815, PASS) e' quindi portato
dalle request a contenuto divergente, mentre l'unico confronto pulito (D1,
output identico) da' 0.6536 (FAIL, -35%). L'utente chiede la verifica
content-controlled sulle agentic PRIMA di decidere.

**Disegno.** Stessi bracci/stesso protocollo §3/§6 (fresh boot per run, 3 run
per braccio interleaved, warm-up scartato, strumenti identici). Request
VERBATIM dai bench T7, tutte con tools (route dflash):
- AG1 (700) e AG2 (700) come da run principale;
- AG3 (600) = `bench-routing-vs-mono.sh:35` / `bench-dflash-df3-agentic.sh:49`
  (unica agentic standard T7 rimasta: log formattati, pattern data/ora/ID);
- D2 (600) = `bench-routing-vs-mono.sh:30` (det T7 standard: alfabeto in
  avanti e indietro — classe pattern-lettere: serve a testare se il crollo
  pos-3 e' specifico dei pattern numerici o generale).
Artifacts in `logs/test-t8-concat/fasea/verify-ag/`, marker
`TREATMENT=fasea-ag-<arm>-run<r>`, container `fasea-ag-*` (trap cleanup).

**Content-control a posteriori.** Per ogni (braccio, run, request):
1. ricostruisce lo stream di token di output dallo speclog (round in ordine:
   token dft accettati + token campionato di chiusura round; validazione:
   lunghezza == `usage.completion_tokens` a meno di 1);
2. allinea token-per-token gli stream A vs B (run1; le repliche 2-3 sono
   bit-identiche, verificate) -> prefisso comune effettivo (LCP);
3. round content-controlled = round la cui intera finestra di verify
   (posizioni p..p+draft_len) cade nel LCP: stesso contesto e stessa
   continuazione target nei due bracci;
4. metrica §7 (identica: frazione accettata per posizione sul segmento
   DFlash effettivo, normalizzazione early-stop) calcolata SOLO su quei
   round; braccio A: solo round composti (k1=1), plain-A informativo;
   braccio B: tutti.

**Criteri di validita'.** Copertura riportata SEMPRE (round
content-controlled / round totali, posizioni coperte / totali, per request e
aggregato). Se i round content-controlled AG1+AG2+AG3+D2 sono < 30 complessivi
-> copertura insufficiente, dichiarata esplicitamente, e FALLBACK: multi-turn
con la risposta del braccio B pre-generata nel contesto condiviso (turn-2
breve, max_tokens 150) per aumentare i tratti identici — eseguito solo in
quel caso, con emenda al report.

**Formula di decisione (nessuna decisione automatica).** Verdetto combinato
D1+AG content-controlled: ratio pooled (round D1 interamente identici +
finestre AG identiche) vs soglia 0.90, piu' per-request e profilo
per-posizione AG (domanda: pos-3 crolla anche su pattern non numerici o e'
specifica di D1?). Output: sezione "Verifica AG content-controlled" nel
report con numeri, copertura, profilo pos-3, verdetto combinato e
raccomandazione MOTIVATA — il gate resta dell'utente.

### 10.1 Fallback eseguito (esito run-ag del 22/08: copertura insufficiente)

Esito run-ag (6/6 boot OK, 24/24 request 200): AG1 allineata per 262 token
(23 round composti A / 59 round B content-controlled, ratio 1.133);
AG2/AG3/D2 divergono al TOKEN 0 (preflip sistematica per-arm: i 3 boot di
ciascun braccio sono repliche identiche ma i bracci partono su modalita'
diverse — stessa classe di flip documentata in T4 tra boot k1=0). Copertura
23 round < 30 -> **fallback attivato** come da criterio.

**Disegno fallback (multi-turn, contesto condiviso forzato).** Turn-1 =
prompt T7 VERBATIM; assistant turn = PRIMI ELEMENTI DEL PATTERN scritti a
mano (non generati dal modello: identico per costruzione nei due bracci);
turn-2 = "Continua esattamente da dove ti sei fermato." La continuazione e'
contenuto-pattern ad alta confidenza (basso rischio di flip al token 0; le
flip restano possibili e sono escluse onestamente dall'LCP). Quattro finestre:
- F1 (alfabeto, classe pattern-lettere = D2): assistant "A".."N" uno per
  riga; continuazione attesa O, P, Q... poi alfabeto inverso.
- F2 (numeri, classe D1): assistant "1".."25" uno per riga; continuazione
  26, 27... — la regione del muro na=3 di D1.
- F3 (log, classe AG3): assistant = prime 10 righe di log standard
  (data/ora crescente, INFO, api, richiesta N, stato 200).
- F4 (JSON, classe AG2): assistant = array JSON con primi 5 oggetti
  user_1..user_5 (punteggio N.5).
Tutte con tools (route dflash), temp 0, max_tokens 300, stessi bracci/boot
del protocollo (3 run per braccio, fresh boot, interleaved). Artifacts
`verify-ag/fallback/`, marker `TREATMENT=fasea-fb-<arm>-run<r>`, container
`fasea-fb-*`.
Analisi: stessa macchina LCP/token-per-token; metrica §7 SOLO su round
content-controlled; copertura riportata; verdetto combinato finale =
D1 (run principale) + finestre AG1-cc (run-ag) + finestre F1-F4 (fallback);
profilo per-posizione su F1-F4 (la domanda pos-3 resta aperta anche qui);
raccomandazione motivata, decisione dell'utente.
