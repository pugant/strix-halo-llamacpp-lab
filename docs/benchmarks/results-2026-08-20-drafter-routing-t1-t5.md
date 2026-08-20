# Drafter routing (MTP ↔ DFlash2 per-request) — Esiti T1-T5

**Data:** 19-20/08/2026 · **Branch:** `drafter-routing` (14 commit oltre `dflash2`, patch duratura
`patches/drafter-routing/0001-drafter-routing-mtp-dflash-per-request.patch`) ·
**Immagine:** `docker-llm-service:vulkan-fork-dflash2-route` ·
**Spec:** `docs/design/2026-08-19-t7f2-drafter-routing-design.md` (emendata 19/08 sera: kind
`dflash-prefix-miss`) · **Piano:** piani di implementazione/esperimento interni `2026-08-19-t7f2-drafter-routing-impl.md` +
`2026-08-20-rs-rollback-dflash-experiment.md` (non inclusi nel repo).

**Config duale:** `--spec-type draft-mtp,draft-dflash --spec-draft-model
<drafter> --spec-draft-ngl all --spec-draft-n-max 7 --spec-draft-p-min 0.75
--spec-draft-p-split 0.10` (MTP→6 clamp in duale, DFlash→7). Modello
Qwen3.8-27B STRIX_LEAN, Vulkan RADV, temp 0, c 16384.

## Sintesi gate

| Test | Gate | Esito |
|---|---|---|
| T1 smoke dual-load | 14 check (boot/policy/override/400/fallback/cache-switch/metrics) | **PASS 14/14** |
| T2 cache round-trip | R3.3 (4 gate × config simple+prod, + post-RS re-cert) | **PASS 4/4** per config |
| T3 percorsi sacri 0005-0009 | S1-S4 + G-final (simple+prod) | **PASS 7/7** (post-RS) |
| T4 A/B routing vs mono | **R3.1** prosa ≥ −3% | **PASS** (+1,0% / −0,2%, 2 run) |
| | **R3.2** agentic ≥ +10% (media 3) | **PASS** (+19,6% / +18,9%, 2 run) |
| T5 spot-check numerica | (a)≡(b), (c)≡(d) char-identical | **PASS** (4/4 bracci identici, ct=305) |

## T4 — tabella (run rs2; rs1 coerente)

| prompt | classe | MONO-MTP6 | DUALE | Δ% | routing |
|---|---|---|---|---|---|
| P1 | prosa | 19,8 | 19,7 | −0,5% | mtp |
| P2 | prosa | 20,4 | 20,4 | +0,0% | mtp |
| D1 | det | 45,9 | 54,7 | +19,2% | dflash |
| D2 | det | 33,7 | 40,7 | +20,8% | dflash |
| A1 | agentic | 38,6 | 46,0 | +19,2% | dflash |
| A2 | agentic | 42,0 | 51,1 | +21,7% | dflash |
| A3 | agentic | 39,9 | 46,2 | +15,8% | dflash |

Invarianti verificati: prose routate a mtp con **dflash gen-drafts delta = 0**
(nessun lavoro parassita); tools → dflash con mtp delta = 0; `replay stalled` 0.

## Il ciclo RS (decisione utente 20/08)

Prima del flip RS, T4 falliva R3.1 (prosa −7,1/−7,5/−15,0%): la condizione
blanket `[TAG_RS_STATE_ROLLBACK_SUPPORT]` (common/common.cpp, motivata a monte
dai SOLI metodi ngram) azzerava `n_rs_seq` con qualunque non-MTP → target su
seq_rm-FULL → ogni round con reject parziale pagava snapshot+restore+replay
(~150 MB; `replay stalled` 3×/run). Flip `38483cc64` (RS mantenuto con
draft-dflash): tassa eliminata, **S3 trailing-rollback tornato raggiungibile
nel duale** (gate originale ripristinato), S1-a risolto. Tutti i gate
G1-G6 dell'esperimento PASS; T5 conferma che il flip NON altera la numerica.

## Bug pre-esistenti trovati e corretti dai test (tutti reviewati)

| Commit | Bug |
|---|---|
| `4a2ec491c` | crash fatale su task dflash-routed nel path reject-parziale/replay (ctx MTP non trimmato → assert) |
| `db2d81c5d` | reasoning-budget mai esaurito su ogni config dflash (sampler clone ricaricava `remaining`) — rilevante anche per mono-dflash in produzione |
| `a303242e2` + `c7ad19bff` | coda `\n\n` mancante nella sequenza forced-end (caveat 15/08; server+cli) |
| `bad0be701` | MTP n_max=7 in duale (clamp chain_heads inerte su nextn mono-head) |
| `8b35a795f`, `9d0ed2380` | osservabilità (pre-registrazione contatori; warning resync MTP demoto a DBG per seq routate altrove — 107/run, tutte benigne) |

## Limiti documentati (non regressioni del routing)

1. **Template + tools:** aggiungere `tools` al body re-renderizza il system
   block del template → riuso prefix ~40 token (lcp=40). Proprietà del template
   Qwen3.8, identica in mono su ckpt7. Le conversazioni miste reale (prosa →
   agentic con tools) non riusano il KV del system prompt.
2. **Dopo switch drafter con cache hit:** il drafter appena-routato parte con
   prefix draft mancante (`dflash-prefix-miss (<P> tok)`) → primi draft
   degradati, output corretto, KV target pienamente riusato (T2: riuso
   ~99,4-100% a ogni switch).
3. **Boot fallback:** file drafter assente → mono MTP-nextn con WARNING;
   override `spec_drafter:"dflash"` → 400 esplicito (`not loaded`).

## Artefatti

- Test: `logs/test-drafter-routing/` (t1-t5, run console + log server + json),
  `logs/test-drafter-routing/t3-rs/`, `t2-rs/`, script in `scripts/` + snapshot.
- Bench: `logs/bench-routing-vs-mono/` (run pre-RS e rs1/rs2, TSV + response).
- Build: `rebuild-image-{1..7}.log` (stessa immagine, tag unico).

**Prossimo passo:** Task 12 rollout produzione — 🚫 GATED su approvazione
esplicita utente (switch immagine+flag llm-service, backup config, osservazione
24h su `spec-route:` + contatori).
