# t8-exclusion — Branch-2: pattern-window exclusion del round concat

Serie patch del ramo `t8-exclusion` (T8 Branch-2, piano
`docs/superpowers/plans/2026-08-22-t8-branch2-exclusion.md`).
Base: `t8-concat` @ `ed3bdd661` (stadio 2 archiviato, serie in
`../t8-concat/` 0001-0011). Applicazione in ordine numerico.

| # | File | Commit | Contenuto |
|---|------|--------|-----------|
| 0001 | `0001-concat-exclusion-detector-selftest.patch` | `531854c7e` | Rilevatore a due segnali (AC theta=0.60 + classe-costante m=4, w=16) header-only `common/spec-concat-exclusion.h`; flag `--spec-concat-exclusion` (default ON, efficace solo a k1>0) e `--spec-concat-selftest FNAME`; selftest pre-load su 200 fixture di parità (gate G1: 200/200 + mutazione 199/200). NESSUN wiring di round. |
| 0002 | `0002-wire-exclusion-gate-round-concat.patch` | `6d55f6c7d` | Gate nel round concat (PRIMA dell'iniezione head): finestra pattern-like → round plain. Osservabilità: boot marker `spec-route: concat exclusion active (theta_ac=0.60 m=4 w=16)` (nested in concat_ready, POST-load), riga per-round `spec-concat: gated p=N segnale=<ac|classe|ac+classe> (seq S)` a SPC_INF, statistics `concat-exclusion: #excl rounds / #gated rounds` (gated su k1>0 && flag ON). A k1=0 il gate NON gira (guard unica via concat_rounds_ready). |

Nessun 0003: la certificazione runtime (Task 7) non ha richiesto fix.

## Applicazione della serie

Su un clone del fork portato alla base `t8-concat` @ `ed3bdd661` (=
serie `../t8-concat/` 0001-0011 applicata con `git am` + commit docs-archive
`ed3bdd661`; `git branch --contains ed3bdd661` → `t8-concat`, `t8-exclusion`):

```sh
git am patches/t8-exclusion/0001-concat-exclusion-detector-selftest.patch \
        patches/t8-exclusion/0002-wire-exclusion-gate-round-concat.patch
# riproduce: 531854c7e -> 6d55f6c7d (ramo t8-exclusion)
```

## Certificazione runtime (Task 7, 2026-08-22) — G0 + G2 VERDI

Immagine certificata: `docker-llm-service:vulkan-fork-t8-exclusion` =
`ba57d9d50aa5` (2026-08-22 20:10:54, commit `6d55f6c7d`).
Log: `logs/test-t8-concat/b2/cert/` (workspace).

- **G0 inerzia a k1=0 (bit-identico, PASS 10/10)**: metodologia T4d/T4s5
  della suite t8-concat riusata cross-image — boot duale T7 senza flag
  concat, prompt D2 fisso (prima request su boot fresco, temp 0), 2 boot
  per braccio. Le 4 run (baseline certificata `vulkan-fork-t8-concat`
  `e13264592171` x2 + exclusion `ba57d9d50aa5` x2) sono char-identical:
  `sha256 = 40c1f3a3fe1351637cb240439e06474a80f1a37b1471a225c0fc66a60a1ccc5e`.
  Zero marker concat/exclusion nei log a k1=0 (inerzia strutturale
  confermata a runtime). NESSUNA divergenza (nemmeno near-tie).
- **G2 suite certificate (90/90)**: `t8-concat-t1` 60/60, `t8-concat-t2`
  20/20, `t8-ring-window-causal` 10/10 sull'immagine exclusion — suite
  VERBATIM (copie `scripts/t8-b2-cert-t{1,2}.sh`: unica deviazione il tag
  immagine + dir log, diff in `logs/test-t8-concat/b2/cert/deviazione-suitediff.txt`;
  la causal prende il tag da argomento). Evidenze chiave riprodotte
  identiche alla recert t7 (es. T5a composti=27 R,7=49; causal delta
  6..16 distinti=7).
- **Prima osservazione runtime del gate** (k1>0, flag default ON):
  - boot marker dopo `concat mode k1=1`, assente a k1=0 e in mono-dflash;
  - righe gated selettive: D1 t1 `gated p=2 segnale=classe` (1 round su 29
    valutati); conversazione prosa t2: 0 gated su 55; boot concorrente
    mixw: 8 gated su 71 (`segnale=ac p=5` e `p=8`) — il gate valuta OGNI
    round concat e taglia solo le finestre pattern-like, i round composti
    restano sostanziali (suite tutte verdi, metrica concat 10/10 coerente);
  - statistics `concat-exclusion: #excl rounds = N, #gated rounds = M`
    presenti a k1>0, assenti a k1=0.

### Invocation G2 letterali (workspace)

```sh
bash scripts/t8-b2-cert-t1.sh                                              # 60 check
bash scripts/t8-b2-cert-t2.sh simple                                       # 20 check
bash scripts/t8-ring-window-causal.sh docker-llm-service:vulkan-fork-t8-exclusion 8094 b2excl   # 10 check
```

### Nuance di riproduzione

- **ID immagine baseline**: G0 ha usato il tag `vulkan-fork-t8-concat` =
  `e13264592171`, mentre i log `t7-recert-*` citano `cfb43d5c9ea9` — stesso
  contenuto, ID diverso per re-wrapping del manifest (verificato 22/08:
  `docker image inspect` di entrambi → `Created` identico
  2026-08-21T21:02:45.921202595+02:00, `Config` e `RootFS` identici).
- **Exit code nei log depositati**: la run G0 depositata stampa i contatori
  (`PASS=10 FAIL=0`) ma non una riga `exit=` esplicita — l'exit 0 e' letto
  dal codice di uscita del processo. Lo script (`g0-bitidentico.sh`,
  hardening post-review) ora termina con `echo "G0_EXIT=$RC"` esplicito per
  le recert future.
