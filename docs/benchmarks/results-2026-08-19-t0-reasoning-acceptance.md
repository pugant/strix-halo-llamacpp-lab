# Risultati — T0: acceptance DFlash2 vs MTP sul thinking (gating policy routing T7-f2)

**Data:** 2026-08-19 · **Piano:** piano di test interno `2026-08-19-t0-reasoning-acceptance.md` (non incluso nel repo)
**Setup:** identico T7 — LEAN 13.8 GB, Vulkan RADV, GPU dedicata (llm-service fermo
12:18→12:47, restart + health OK), `-c 16384 -fa on --jinja`, temp 0, p_min 0.75,
p_split 0.10, warm-up scartato, marker TREATMENT. Bracci: MTP6 (ckpt7) / DF7 (dflash2).
Script `scripts/bench-t0-thinking-acceptance.sh` (non incluso nel repo), log `logs/bench-t0-thinking/`.

## Dati

**Set A — thinking-dominant** (reasoning lungo, risposta ~1 token; max_tokens 3000):

| Prompt | MTP6 tok/s | DF7 tok/s | Δ | acc mean MTP | acc mean DF7 | per-pos DF7 |
|---|---|---|---|---|---|---|
| A1 treno | 27.4 | 26.8 | −2% | 3.85 | 4.85 | (0.90, 0.73, 0.61, 0.55, 0.47, 0.35, 0.25) |
| A2 radice | 26.5 | **30.1** | **+14%** | 3.63 | **5.27** | (0.87, 0.77, 0.64, 0.59, 0.53, 0.46, 0.42) |
| A3 scatole | 24.0 | 22.9 | −5% | 3.47 | 4.06 | (0.83, 0.64, 0.49, 0.39, 0.30, 0.24, 0.18) |

Reasoning chars: A1 2547/3682, A2 821/1327, A3 4131/10180 (MTP/DF7 — la divergenza
numerica batched-verify, nota preesistente, cambia la traiettoria di reasoning).

**Anchor B (controllo riproducibilità vs T7):**

| Prompt | MTP6 | DF7 | T7 riferimento |
|---|---|---|---|
| B4 det (conta 1-200) | 44.7 | 56.8 | 45.2 / 57.4 ✓ ±1% |
| B5 prosa (storia Roma) | 21.0 | 15.3 | 19.6 / 14.2 (+7-8% run-to-run, Δ relativo −27% confermato) |

Anchor riprodotti → run valido.

## Lettura

1. **Il thinking NON è classe-prosa.** Acceptance DF7 su thinking 4.06-5.27 (vs prose
   2.39, vs det 7.59): sta tra le classi, vicino al det. Per-pos resta ≥0.42 fino a
   pos 7 su A1/A2 (la prosa crolla a 0.03). Anche MTP sull'thinking sale (3.47-3.85
   vs 2.57 prose).
2. **tok/s su thinking = quasi-parità**: −2% / +14% / −5% per prompt (**media +2.4%**).
   La penalità −26% della prosa libera NON si applica al reasoning.
3. Aggregazione token-weighted: −3.7% — confondono nota: su A3 la divergenza numerica
   fa ragionare DF7 2.5× più a lungo (10180 vs 4131 char), quindi il peso dell'unico
   prompt in calo è sovrastimato nell'aggregato; non è un effetto velocità del drafter.

## Verdetto (regola congelata nel piano)

- Acceptance mean DF7 set A ≥ 3.5: **PASS** (4.06/4.85/5.27).
- tok/s DF7 ≥ MTP6 −3%: **PASS su media per-prompt** (+2.4%; token-weighted −3.7% al
  confine, con confound descritto).

**→ Thinking = DFlash-tollerante → politica a REQUEST INTERA** (switch drafter solo a
confine task). La fase-aware (MTP per thinking, switch a end_tag) comprerebbe ~0-4%
sulle fasi di thinking al costo di switch mid-generation + re-encode prefix + tagging
checkpoint: **scartata su dati (YAGNI quantificato)**. Resta fallback documentato se
la telemetria di produzione mostrasse sotto-performance su request thinking-heavy
(il campo per-seq `drafter` del design la supporterebbe senza rework).

## Implicazione per il design T7-f2

- Architettura Sezione 1 **senza relaxation**: switch SOLO a confine task (nessun
  cambio mid-generation, checkpoint/salvage nel regime per cui sono stati scritti).
- Politica: segnale request-level (`tools`/`tool_choice` → DFlash, default MTP) +
  override body opzionale. Il costo del misrouting prosa→DF7 resta −26%: la policy
  resta conservativa (default MTP), ma il timore "DFlash danneggia il reasoning
  agentic" è confutato dai dati (il thinking agentic è proprio il caso tolerance).

## Caveat

- Prompt corti/ctx fresco (produzione: ctx lunghi + budget 2048): il floor banda
  abbassa entrambi i drafter in modo simile; la comparazione relativa regge.
- 3 prompt per classe: sufficienti per la decisione di policy (gap ampio), non per
  claim fini sul +14% di A2.
