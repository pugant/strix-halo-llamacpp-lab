# Patch upstream llama.cpp — ARCHIVIATA (non inviata)

**Stato: topic CHIUSO 18/08/2026 su decisione utente.** La PR a ggml-org/llama.cpp
non è stata inviata: AGENTS.md del repo vieta commit message / PR description /
push / PR-create da parte dell'agente (ban risk per il contributor), e l'utente
ha scelto di non redigere personalmente i testi richiesti.

## Cosa contiene

`0001-server-reasoning-budget-forced-newline.patch` — fix del round-trip della
sequenza forzata a fine reasoning budget su **upstream master 3dc7285** (18/08):
forced sequence `'\n' + message + ('\n' se message non vuota) + end_tag`,
tokenizzata come stringa unica, nel handler `reasoning_budget_message` di
`tools/server/server-schema.cpp` (+ helper `first_reasoning_end_tag`). Port della
nostra patch fork 0005+0009 su upstream.

## Verifica effettuata (prima dell'archiviazione)

- Bug confermato nel codice: concatenazione token senza newline, riga ~421.
- e2e A/B (Qwen3.8-2B-Q4_K_M + template 27B via `--chat-template-file`):
  main = 116/170 token ri-valutati al resend (DIVERGED), patch = 17/170
  (EXACT-HIT), baseline naturale 20/173; idem con wrap-up message (S2).
- Dettagli: `docs/superpowers/plans/2026-08-18-upstream-roundtrip-verify.md`;
  factsheet per PR: `docs/upstream-pr/2026-08-18-reasoning-budget-roundtrip-factsheet.md`;
  script: `scripts/upstream-roundtrip-test.py`. (Tutti artefatti del workspace
  originale, non inclusi nel repo.)

## Se si riapre

Rifare il clone (shallow), applicare questa patch, buildare, rilanciare lo
script di verifica; poi commit/push/PR **a cura del contributor umano** con AI
disclosure nel template PR. Nota: upstream evolve — ricontrollare che il
handler sia ancora in `server-schema.cpp` e che il bug non sia stato fixato.
