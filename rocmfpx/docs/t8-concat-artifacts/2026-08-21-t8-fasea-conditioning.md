# T8 Stadio 2 — Fase A: condizionamento DFlash-su-prefix-MTP (gate §7)

Data: 2026-08-21. Piano: `docs/superpowers/plans/2026-08-21-t8-stadio2-faseA-gate.md`
(scritto PRIMA del run). Script: `scripts/bench-t8-fasea.sh` (run + analyze).
Immagine `vulkan-fork-t8-concat` (`e13264592171`), repo `t8-concat` @ `ae6bad9cf`.
GPU dedicata (nessun container llm-service). Marker TREATMENT in tutti gli
artifact. **Il verdetto del gate e' una PROPOSTA: la decisione e' dell'utente.**

## 1. Protocollo eseguito

- Braccio **A** (CONCAT): boot duale T7 + `--spec-concat-k1 1`.
- Braccio **B** (DF7-puro): identico SENZA flag (k1=0).
- 3 run per braccio, fresh boot per run, ordine interleaved A/B per indice di
  run; warm-up scartato; 3 request identiche per run: D1 (det, 600) + AG1/AG2
  (agentic T7, 700), tutte con tools minimal (route `signal=tools ->
  drafter=dflash` verificata per finestra), temp 0, `-c 16384`, p_min 0.75,
  p_split 0.10, n-max 7.
- Strumentazione identica nei bracci: `SPEC_VERIFY_LOG` (righe R/P),
  `LLAMA_TRACE=1` (anchor `accepted X/Y` per round), `-lv 4` (marker TRC
  `concat round composed` per round composto). 6/6 boot OK, 18/18 request
  HTTP 200, container rimossi.
- Coerenza strumenti (tutte le finestre): #anchor == #R, valori anchor == R,
  #round composti == marker `statistics concat`, delta `#gen drafts` == #R,
  zero righe W, zero `restore checkpoint`, routing atteso. Nessuna anomalia.

Metrica (spec §7, verbatim nel piano §2): `acc =` token accettati / posizioni
presenti sul segmento DFlash effettivo (A: posizioni k1+1..fine dei round
composti; B: tutto il draft), normalizzata per early-stop/budget.

## 2. Metrica gate come da definizione (pooled 3 request x 3 run)

| metrica | valore |
|---|---|
| `acc_concat` (round composti A) | 3039/4215 = **0.7210** |
| `acc_df7` (round tutti B) | 4209/5730 = **0.7346** |
| ratio | **0.9815** |
| soglia | 0.90 |
| verdetto calcolato (letterale §7) | **PASS (proposta — decisione utente)** |

Round: A composti 606, A plain 267, B 822. Head-acceptance 594/606 = 0.980.

## 3. CONFOUND critico: divergenza di contenuto tra bracci

A temp 0 le 3 repliche dello stesso braccio sono BIT-IDENTICHE (hash risposta
uguali, stddev acc = 0.0000: il floor boot-to-boot documentato NON e' scattato
in questa sessione). Tra bracci, pero':

| req | hash A | hash B | esito |
|---|---|---|---|
| D1  | 9c5b6624a95e | 9c5b6624a95e | **output IDENTICO** (600 tok, finish=length) |
| AG1 | 499c8cc2eb67 | 0c61d71c98af | diverge a char 410 (reasoning: "Could use builtins?" vs "Could be:") |
| AG2 | 0c8cc597eee8 | b5b5665374d0 | diverge a char 1034 (reasoning: "Need ensure valid." vs "Check count.") |

Le divergenze AG1/AG2 sono near-tie del target capovolti dalla struttura di
batch diversa (9 vs 8 colonne -> numeriche f32), lo stesso floor documentato in
T4 (k1=0 vs no-flag, gap lp 0.000/0.000). Conseguenza: l'acceptance di AG1/AG2
misura "il contenuto che ciascun braccio ha generato", non il condizionamento.
Il braccio B e' finito su continuazioni piu' dure per il proprio drafter
(AG2: acc 0.6099 vs D1 0.9217), cosa che GONFIA il ratio pooled.

**L'unico confronto content-controlled e' D1** (stesso token stream nei due
bracci):

| subset | acc_concat | acc_df7 | ratio | esito vs 0.90 |
|---|---|---|---|---|
| D1+AG1+AG2 (pooled, come §7) | 0.7210 | 0.7346 | 0.9815 | PASS |
| D1 (content-controlled) | 0.6024 | 0.9217 | **0.6536** | **FAIL (-35%)** |
| D1+AG1 | 0.6572 | 0.8523 | 0.7710 | FAIL |
| AG1+AG2 (solo diverse) | 0.8242 | 0.6565 | 1.2554 | (non interpretabile) |

Il PASS pooled e' quindi interamente portato dalle request a contenuto
divergente; sul dato pulito il gate fallisce con ampio margine.

## 4. Dove cade l'acceptance (D1, contenuto identico)

Profilo per-posizione relativa al segmento (pooled 3 run; `p_dft` =
probabilita' target del token draftato, controfattuale non troncato dallo
stop esatto):

| pos | acc A | acc B | ratio | p_dft A | p_dft B |
|---|---|---|---|---|---|
| 1 | 0.989 | 0.975 | 1.01 | 0.984 | 0.977 |
| 2 | 0.936 | 0.951 | 0.99 | 0.928 | 0.957 |
| **3** | **0.479** | **0.925** | **0.52** | **0.494** | **0.941** |
| 4 | 0.473 | 0.912 | 0.52 | 0.956 | 0.959 |
| 5 | 0.462 | 0.912 | 0.51 | 0.983 | 0.985 |
| 6 | 0.452 | 0.887 | 0.51 | 0.964 | 0.930 |
| 7 | 0.419 | 0.887 | 0.47 | 0.880 | 0.918 |

Lettura:

- Il danno NON e' sulle prime posizioni dopo il prefix MTP (pos 1-2 a
  parita'/meglio del puro) e NON e' in coda: il controfattuale p_dft a pos 4+
  e' sano (0.95-0.98) — la loro acceptance bassa e' solo la conseguenza
  contigua del rifiuto a pos 3.
- Il crollo e' concentrato alla **posizione 3 del segmento** (3a colonna del
  blocco DFlash): acc 0.479 vs 0.925, p_dft 0.494 vs 0.941.
- Distribuzione na sui round composti D1 (run1, 94 round): bimodale
  `{8: 39, 3: 43, altri: 12}` — o il round va pieno (8/8) o muore a na=3.
  Sequenza periodica (stretch di 32 round pieni, poi cicli 7x na=3 + na=2 +
  na=8), in fase con il contenuto (conteggio numerico ripetitivo).
- Meccanismo del rifiuto a pos 3 (detokenizzazione, run1 round 45-49): il
  blocco condizionato sull'head propone la cifra del NUMERO PRECEDENTE
  (es. target "1" di "101", draft "0" di "100"; target "2" di "102", draft
  "1" di "101"...) — un clone/out-by-one del pattern, con p_dft ~ 0 (mediana
  0.000 sui 55 stop di D1): il target lo rifiuta con certezza, non e' un
  near-tie. Lo stesso tratto di testo e' draftato correttamente dallo stesso
  drafter nel braccio B (71/81 round pieni).
- Head quasi sempre accettata (0.980): il problema non e' la qualita' del
  token MTP, e' il blocco DFlash condizionato su di esso. Acc del segmento
  condizionata a head accettata: 0.7357 (vs 0.7210 non condizionata).
- [informativo] I round PLAIN dello stesso boot A (post-rejection, selezione
  distorta) accettano 0.37-0.46 — non confrontabili con B (mix diverso).

## 5. tok/s (informativo, NON gate; strumentazione identica nei bracci)

| req | A (concat) | B (puro) | delta |
|---|---|---|---|
| D1  | 34.2 ±0.4 | 47.3 ±0.2 | **-27.7%** |
| AG1 | 34.5 ±0.2 | 40.4 ±0.4 | -14.5% |
| AG2 | 35.7 ±0.3 | 33.6 ±0.4 | +6.4% |
| media | 34.8 | 40.4 | -13.9% |

Su D1 (contenuto identico) il costo in tok/s del crollo di acceptance e'
diretto e grande. AG1/AG2 mescolano l'effetto acceptance con la divergenza di
contenuto (lunghezze/formattazioni diverse).

## 6. Sintesi per la decisione (run principale 21/08 — aggiornata in §9/§10)

> NOTA 22/08: dopo questo run l'utente ha chiesto la verifica
> content-controlled (opzione c). Le sezioni §9-§10 la riportano e la
> §10 sostituisce questa sintesi per la decisione.

- **Verdetto calcolato sulla definizione letterale §7 (pooled 3 request):
  ratio 0.9815 >= 0.90 -> PASS.** Ma e' dominato da AG1/AG2, le cui
  continuazioni divergono tra bracci (near-tie f32): non misurano
  condizionamento.
- **Sul solo confronto a contenuto controllato (D1, output identico): ratio
  0.6536, molto sotto 0.90 -> il condizionamento costa ~35% di acceptance
  relativa**, concentrato alla posizione 3 del segmento (clone del pattern
  numerico), con -27.7% tok/s sulla stessa request.
- L'ipotesi di rischio della spec ("condizionamento mai misurato, ±12-20%") e'
  confermata nella direzione pessimistica SUI CONTENUTI PATTERN-NUMERICI; su
  AG1/AG2 (contenuto diverso) il meccanismo non mostra danni — anzi il braccio
  B e' finito su continuazioni piu' dure.
- Cosa NON e': non e' un difetto del verify (output greedy identico a parita'
  di contenuto), non e' la qualita' dell'head (98% accettata), non sono
  anomalie di strumento (tutti i cross-check coerenti, zero restore).
- Opzioni per l'utente: (a) FAIL — fermare lo Stadio 2 qui (il dato pulito
  viola la soglia; il k1=6 la aggraverebbe: piu' posizioni condizionate);
  (b) PASS formale sul pooled e procedere in Fase B con il rischio
  documentato; (c) richiesta di dati aggiuntivi (es. D1-like aggiuntive a
  contenuto controllato, o AG1/AG2 ri-forzate sullo stesso testo via
  prefix-match) prima di decidere.

## 7. Limiti

- 3 request, di cui 2 con contenuto divergente tra bracci: la stima pulita
  poggia su D1 (600 token, 98+81 round per run, 3 repliche identiche).
- Repliche deterministiche (stddev 0.0000): quantificano il rumore di
  replicazione (nullo), non la variabilita' boot-to-boot (non scattata).
- `-lv 4` + trace attivi in ENTRAMBI i bracci: il confronto tok/s interno e'
  valido, i valori assoluti sono leggermente depressi vs un boot silenzioso.
- La metrica e' implementation-faithful (early-stop/budget normalizzati per
  round sulle posizioni presenti); per DFlash2 la larghezza e' piena a parte
  le code di budget, identiche nei bracci.

## 8. Artifact

`logs/test-t8-concat/fasea/`: `arm{A,B}-run{1,2,3}/` (meta.txt, resp-*.json,
speclog-cum-*, srv-cum-*, srv-full.log), `fasea-runlog.tsv` (live),
`fasea-metrics.json` (metriche), `run-console.log`, `probe/` (Step 0,
escluso dalle metriche).

## 9. Emenda 22/08 (opzione c utente): verifica AG content-controlled

Piano: §10 del file piano (emendato PRIMA del run). Script:
`bench-t8-fasea.sh run-ag` / `analyze-ag` + fallback `run-fb` / `analyze-fb`.
Metodo: ricostruzione token-per-token degli stream di output dallo speclog
(token dft accettati + token di chiusura round; validati vs
`usage.completion_tokens`), allineamento A-vs-B, prefisso comune effettivo
(LCP), round content-controlled = finestra di verify interamente dentro LCP.
Metrica §7 invariata, calcolata SOLO su quei round.

### 9.1 run-ag (AG1+AG2+AG3+D2, 3 run per braccio, 6/6 boot OK)

- AG1: LCP 262 token su 541/443 -> **23 round composti A / 59 round B cc**;
  acc 0.5404 vs 0.4770, **ratio 1.133** (sul reasoning iniziale il concat
  non danneggia, anzi: profilo pos 2-7 tutto >= 1.0).
- AG2/AG3/D2: **divergenza al TOKEN 0** tra bracci (i 3 boot per braccio
  restano repliche identiche: la flip e' sistematica per-config, non rumore
  per-boot — stessa classe documentata in T4). LCP=0, zero finestre.
- Copertura 23 round < 30 -> **fallback attivato** come da criterio del piano.
- Nota strumento: AG3 braccio B stream 598 vs completion 600 (guard che
  droppa l'ultimo round troncato dal budget; benigno, coerente su 3 run).

### 9.2 Fallback §10.1 (multi-turn: turn-1 T7 verbatim + pattern scritto a
mano identico nei bracci + "Continua esattamente da dove ti sei fermato.")

| finestra | classe | LCP | cc A(comp) | cc B | acc A | acc B | ratio |
|---|---|---|---|---|---|---|---|
| F1 alfabeto | pattern-lettere | 182 | 20/34 | 49/75 | 0.1857 | 0.3703 | **0.502** |
| F2 numeri 26+ | pattern-numerici (D1) | 0 | 0 | 0 | — | — | flip token-0 |
| F3 log | pattern-log | 18 | 4/30 | 4/52 | 0.1786 | 0.3214 | 0.556 (n=4) |
| F4 JSON | pattern-JSON | 25 | 2/17 | 5/68 | 0.5714 | 0.4571 | 1.250 (n=2) |
| pooled FB | | | 26 | 58 | 0.2143 | 0.3744 | **0.572** |

- F1 e' il dato nuovo decisivo: **il crollo NON e' specifico dei numeri** —
  sull'alfabeto il segmento composto accetta la meta' del puro, e la
  penalita' parte gia' dalla posizione 1 (0.40 vs 0.82) invece che dalla 3:
  su contenuto pattern a confidenza piu' bassa il condizionamento degrada
  tutto il blocco, non solo il punto di incremento (D1: pos 1-2 a parita',
  muro netto a pos 3; F1: ratio ~0.5 uniforme su tutte le posizioni).
- In entrambi i casi il rifiuto e' categorico (p_dft al primo mismatch:
  mediana 0.000 su 43 stop D1+F1) — errori da clone del pattern, non near-tie.
- F4 (JSON) in 2 round cc va bene (aneddoticamente la classe JSON
  strutturato regge; copertura troppo sottile per concludere).
- tok/s F1: 18.8 vs 25.5 (**-26%**, coerente col costo acceptance).

### 9.3 Verdetto combinato content-controlled (la lettura pulita del gate)

| parte | acc_concat | acc_df7 | ratio | round cc |
|---|---|---|---|---|
| D1 (run principale) | 394/654 = 0.6024 | 518/562 = 0.9217 | **0.654** | 94/81 (run1) |
| AG1-cc (reasoning) | 87/161 = 0.5404 | 197/413 = 0.4770 | **1.133** | 23/59 |
| F1-F4 fallback (pattern) | 39/182 = 0.2143 | 152/406 = 0.3744 | **0.572** | 26/58 |
| **TOTALE combinato** | 520/997 = 0.5216 | 867/1381 = 0.6278 | **0.831** | 143/198 |

Ratio per finestra content-controlled: 0.654 (D1), 1.133 (AG1), 0.502 (F1),
0.556 (F3, n=4), 1.250 (F4, n=2) -> mean 0.819, spread (stddev) 0.33.
Soglia 0.90: **il combinato content-controlled e' SOTTO soglia (0.831)**,
come gia' D1-only; il PASS del pooled formale (0.9815, §2) e' confermato
artefatto della divergenza di contenuto (le repliche same-arm sono
bit-identiche: non c'e' rumore che mediare, solo contenuti diversi).

## 10. Sintesi aggiornata per la decisione (emenda 22/08)

- Il quadro content-controlled ora copre tre classi: pattern numerico (D1,
  0.654), pattern lettere (F1, 0.502), reasoning libero (AG1, 1.133). Il
  danno e' concentrato e grave sul contenuto PATTERN (ripetitivo), assente
  sul reasoning. Il pool combinato 0.831 sta sotto la soglia 0.90 solo
  perche' le finestre pattern dominano il conteggio posizioni.
- Meccanismo uniforme nelle finestre colpite: il blocco DFlash condizionato
  sull'head MTP propone token clonati/out-by-one del pattern (p_dft ~ 0 al
  punto di stop); l'head stessa e' accettata al 98%. Non e' un difetto del
  verify (output identico a parita' di contenuto) ne' dello strumento
  (tutti i cross-check coerenti su 12 boot).
- tok/s (contenuto identico): D1 -27.7%, F1 -26%: il costo e' reale e
  immediato sulle classi colpite.
- **Raccomandazione motivata (non decisione): FAIL del gate fase A.** La
  definizione §7 valutata sui soli dati leggibili (content-controlled) da'
  0.831 < 0.90 (e 0.654/0.502 sulle classi pattern, che sono esattamente la
  classe target di DFlash). Il PASS formale pooled poggia su confronti tra
  contenuti diversi. In Fase B (k1=6) il numero di posizioni condizionate
  cresce da 1 a 6: nelle finestre colpite la penalita' e' strutturale e
  crescerebbe. Se l'utente volesse proseguire nonostante il dato, la via
  ragionevole e' una policy che escluda dal concat il contenuto
  pattern-ripetitivo (classes §4) — ma e' un percoso non validato dalla
  spec attuale e con guadagno residuo da dimostrare.
- Decisione: **FAIL -> lo Stadio 2 si ferma (niente k1=6/Fase B)** | PASS
  formale col rischio documentato | altri dati. Il gate resta dell'utente.

## 11. Artifact aggiornato (emenda 22/08)

`logs/test-t8-concat/fasea/verify-ag/`: `arm{A,B}-run{1,2,3}/` (run-ag),
`verify-ag-runlog.tsv`, `verify-ag-metrics.json`, `fallback/` (run-fb:
`arm{A,B}-run{1,2,3}/`, `body-F{1..4}.json`, `fallback-runlog.tsv`,
`fallback-metrics.json`, `run-console.log` x2).
