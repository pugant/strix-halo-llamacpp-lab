# Spec — T7-f2: Drafter routing per workload (MTP ↔ DFlash2) in the ROCmFPX fork

**Date:** 2026-08-19 · **Status:** approved in co-design with the user (brainstorming session)
**Context:** T7 DFlash2 (report `docs/benchmarks/results-2026-08-19-dflash2-vs-mtp.md` +
Addendum 1) concluded: DFlash2 wins on deterministic/agentic workloads (+23-39%),
loses on free prose (−26%). This spec defines the routing: **one server, one
target model, two active drafters, per-request choice**.

## 1. Goal and requirements (locked with the user)

Run production (llm-service) with the right drafter per workload:
- prose/chat → **MTP n6** (19.6-21.0 tok/s; DFlash penalty −26%)
- agentic/det → **DFlash2 n7** (33.8-57.4 tok/s, +23-39%)

**Binding requirements:**
- R1 — **Client-agnostic**: any OAI client (curl, webui, pi) works without custom
  fields. Accepted fallback but not required: pi-only flag. Optional body field
  `spec_drafter` as an explicit override for power users.
- R2 — **A single target model in execution** (no double instance/two weights).
- R3 — Production success criteria: (1) prose ≥ −3% vs MTP6; (2) agentic ≥ +10%;
  (3) **zero cache regressions** on fixes 0005-0009 (cold-fallback/rollback/checkpoint);
  (4) curl without custom fields = same behavior; (5) rollback = image switch,
  no code retraction.
- R4 — Destination: llm-service production (switch with final user approval).
- R5 — Internal harness purpose: NOT material to share with the community.

## 2. Experimental evidence (supporting the decisions)

**T0 (gating policy, run 19/08)** — report
`docs/benchmarks/results-2026-08-19-t0-reasoning-acceptance.md`, logs
`logs/bench-t0-thinking/`:
- **Thinking is NOT a prose-class task**: DF7 acceptance on reasoning 4.06/4.85/5.27
  (prose 2.39, det 7.59); tok/s near-parity (−2%/+14%/−5%, average +2.4%).
- → **Whole-request policy**; phase-aware (switch at end_tag) DISCARDED on the data
  (it would buy ~0-4% at the cost of mid-generation switching + re-encode + checkpoint
  tagging). Documented fallback, not implemented.

**T7 A/B** — DF7 agentic: coding +23%, logs +28%, json +3%; det record 57.4.
**Community** — NO ready-made multi-drafter per-request: verified in
llama.cpp master code (a single `mparams`/`ctx_dft`; `common_speculative_init_result` =
RAII factory for ONE draft model), vLLM/SGLang (one drafter per deployment), SpecForge
(arXiv 2603.18567 = EAGLE-3 training framework, no routing).

## 3. Architecture — dual-drafter in a single `common_speculative`

The fork's framework impl already contains: a vector of impls for `common_speculative`,
per-seq state (`dparams` + `[n_seq]` vectors), priority chain with `drafting` flag.
The surgery adds the plumbing for two **draft-model** impls (MTP + DFlash) and the
per-seq choice. **No new drafter**: the MTP and DFlash impls are the existing, tested ones.

**CLI activation** (no need to vectorize mparams — our case is exactly
"an external draft model + MTP-nextn in the target GGUF"):
```
--spec-type draft-mtp,draft-dflash --spec-draft-model /llmodels/QWEN3.8/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
--spec-draft-n-max 7 --spec-draft-p-min 0.75 --spec-draft-p-split 0.10 --spec-draft-ngl all
```
Rule: the `draft-mtp` type does NOT consume a file (it uses the target's nextn); the first
type with an external draft-model consumes `--spec-draft-model`. A single `--spec-type` = mono
mode → identity policy, **zero behavior changes** (R3.4, R3 rollback).

### 3.1 The six surgical touches (~250-400 LOC total)

**T1. Context — `tools/server/server-context.cpp` `load_model` (~80 LOC).**
Today the mutually exclusive if/else-if chain (915-961) creates a single ctx_dft (singleton
members 731-732). In dual mode: two cparams and two ctxs:
- `ctx_dft_mtp`: on the **target model**, `ctx_type=LLAMA_CONTEXT_TYPE_MTP`,
  `ctx_other=ctx_tgt` (existing branch 939-961 for nextn);
- `ctx_dft_dflash`: on the **DFlash model** (loaded from file, existing branch 872-933),
  `ctx_other=ctx_tgt`, causal-off + layer extractions (DFlash ctor flags,
  speculative.cpp:1121-1128).
The extractions on the target (MTP pre_norm at 1557-1558, DFlash layer_inp) are **additive**
and coexist. Additional members: `model_dft_dflash`, `ctx_dft_mtp`, `ctx_dft_dflash`.

**T2. Per-impl config — `common/speculative.cpp` `common_speculative_init` (2941-3010,
~40 LOC).** The config copied per impl (73-79) today carries the single `ctx_dft`; it becomes:
MTP config → `ctx_dft_mtp`, DFlash config → `ctx_dft_dflash` (new field or
`ctx_dft_by_type`). Impls instantiated as today (3029, 3033). **Per-type guard
(mandatory, trap verified at 2951)**: the DFlash impl is constructed ONLY if
`ctx_dft_dflash != nullptr` (today `has_draft_dflash = enabled && ctx_dft != nullptr`
would construct a DFlash impl on the target's MTP ctx if the model path is missing);
symmetrically, the MTP impl only if its own ctx exists. A type whose model/ctx
is not available is **dropped at boot with a WARNING** and the server starts mono
(see §5 line 1: this fallback is NEW code).

**T3. Per-seq gating — `common/speculative.h:32-69` + `speculative.cpp` (~60 LOC).**
New field `common_speculative_type drafter` in `common_speculative_draft_params`.
Asymmetric gating, motivated by how the cache works (see §4):
- **`draft()` gated for both** (loop 3173-3250): only the impl active for that seq;
- **DFlash `process()` gated** (3131-3143): the encoder + KV injection is the real cost
  and pollutes the DFlash ctx — never for MTP seqs;
- **MTP `process()` UNGATED for all seqs**: the boundary row capture
  (`pending_h`) is cheap and is exactly what makes the state required by the
  prompt cache save available (server-context.cpp:149-152; `state_required` is global and
  MTP is the only stateful one). Gating it for DFlash-routed seqs → state never captured →
  `prompt_save` skips the whole entry → agentic conversations lose the cache
  (R3.3 violation, found in review).
`impl_last[seq_id]` (3211-3225) attributes `accept()` to the active impl.
**Switch ONLY at task boundary** (never mid-generation): the value is set at task
start and does not change. Known and accepted side effect (from review iter.2): with
ungated MTP process, the batches of DFlash-routed seqs are also mirrored into the KV of
`ctx_dft_mtp` — neutral-to-positive (the MTP ctx stays warm for future MTP tasks on the
slot; stale rows are covered by the T6 trim).

**T4. Sizing at max (~10 LOC).** `n_rs_seq` and `n_batch`/`n_ubatch` computed with
`n_draft = max` = 7 (the upstream comment in the code anticipates exactly this).
Per impl the effective n-max values stay auto-clamped (MTP→6 via chain_heads 1566-1567;
DFlash→7 via block_size 1082-1088). The `n_rs_seq` checks (server-context 862, 1032-1037)
read the global one → they pass.

**T5. Policy — `tools/server/server-task.cpp` (~30 LOC).** At body parse
(next to the existing `speculative.n_min/n_max/p_min` at 313-322; `#if 0` embryo at
324-336): non-empty `tools`/`tool_choice` → `drafter=DFLASH`, otherwise `MTP`
(conservative default: misrouting prose→DFlash costs −26%, the inverse zero).
Optional explicit override `spec_drafter` ∈ {`mtp`,`dflash`,`auto`} where
**`auto` ≡ absent** (= the policy decides). Unknown value → clear 400.

**T6. Trim/checkpoint both ctxs (~10 LOC).** The point that trims `ctx_dft`
(server-context.cpp:2583, `common_context_seq_rm` beyond `ckpt.pos_max`) also trims the
other ctx: stale rows beyond the accepted boundary must not survive a
future switch (position collisions).

### 3.2 Data flow

```
POST /v1/chat/completions (standard OAI body)
  → server_task_params_from_json: drafter = override | policy(tools)
  → get_available_slot: UNCHANGED (LCP/LRU on the target KV — drafter-agnostic)
  → update_slots: per-slot dp {n_max, n_min, p_min, drafter} (zone 2535-2551)
  → common_speculative_draft: ONLY the impl active for that seq
  → common_speculative_process: MTP always (boundary for the cache), DFlash only if active
  → verify batch: unchanged — the numerical invariance argument is PER-SEQ
    (each seq is verified against its own single drafter's tokens)
  → checkpoint/trim: both ctx_dfts; spec_state = MTP blob as today
```

**n_parallel**: production runs `--parallel 4 --kv-unified` (start-llama-server.sh,
Qwen3.8-27B branch) → slots on different drafters in the same round is a live scenario. The
per-seq mechanism of T3 covers it. The strict-qwen check requires np=1 and in production
it is NOT used: do not enable it.

### 3.3 Deliberate YAGNI (out of scope, documented)

- Generalization to N external draft models; porting of `common_speculative_init_result`
  from master; slot-affinity per drafter in `get_available_slot` (1281-1330; only if
  telemetry shows a relevant switch-cost); adaptive routing on acceptance; phase-aware
  (discarded on T0 data).

## 4. Cache and checkpoint (criterion R3.3)

Principle: **the target KV is drafter-independent** and is the part worth the 91% of
saved prefill; the draft-side state already has a graded degradation path.

1. **Entry tag** (`prompt_save`, server-context.cpp:131-155): a `drafter`
   field in the RAM prompt cache entry. The "dft" part of the entry = KV state of the
   ctx of the **active drafter** (the call
   `prompt_cache.save(prompt, ctx_tgt, ctx_dft, …)`
   at 154 today passes the singleton: it passes the active drafter's ctx). **Note (from
   review, do NOT "clean this up")**: dflash-tagged entries will still carry the MTP spec
   blob (dead weight, ~KB) because the save gate requires the stateful impl's state —
   MTP process is ungated precisely for this (§3.1 T3).
2. **Load** (`prompt_load`, 157-186): target KV restored **always**; dft part +
   spec blob only on matching tag.
3. **Surgical semantic change** (the ONLY modification to the semantics of fixes
   0005-0009, in the LESS punitive direction): today "entry without spec on an
   MTP-required server → `res=false` → cold regeneration" (163-169). With routing the
   full-cold narrows to the **target** mismatch alone; drafter mismatch → target ok,
   draft rebuilt: MTP = resync with empty boundary (one discarded draft,
   existing path 1716-1758); DFlash = no re-encode of the prefix exists in the
   impl (verified in implementation on 19/08: `begin()` only has the
   pos_max<N-1 warning, the ctx only receives deltas) → missing prefix = degraded
   drafts, correct output (verify on the target). [Amendment 19/08 evening: the honest
   log kind is `dflash-prefix-miss`, see §6.]
4. **Checkpoint-salvage 0007/0009**: MTP remains the only stateful impl (DFlash does not
   override `get_state` — default 190-192) → `common_speculative_get_state`
   first-wins (3284-3296) remains correct. The checkpoint (`data_spec`, 2020-2061;
   snapshot 2520-2529) is labeled with the active drafter (with switching only at task
   boundary, one task = one drafter, no mid-flight case). Trailing-rollback (2816-2907):
   trim tgt + both ctxs (2833-2838) + rollback_state as today.
5. **Idle cache (2113-2125) and disk cache (server-task.cpp:2447-2455, 3016-3105)**:
   same rule as the tag (disk cache not enabled in production: covered for
   correctness, light tests).
6. **UNCHANGED**: LCP/LRU, context-shift (2419-2424), forced-end/soft-wrap 0008/0009.

## 5. Error handling (every failure degrades to the status quo, never a crash)

| Failure | Behavior |
|---|---|
| External drafter missing/corrupt at load | **NEW boot fallback**: drop the dflash type + WARNING + mono MTP-nextn start. NB: today the behavior is `SRV_ERR` + server does NOT start (server-context.cpp:886-890) — this fallback is voluntary additional code, with the per-type guard of §3.1 T2 |
| Mono mode (a single `--spec-type`) | Identity policy, zero new paths |
| `spec_drafter` with a value not in the enum | 400 with a clear enum |
| `spec_drafter=dflash` on a mono-MTP server (dflash dropped at boot) | Explicit 400 "drafter dflash not loaded; active: mtp" — NEVER a silent fallback |
| Tag mismatch at cache load | Target restored, draft rebuilt, counted INFO log |
| DFlash loaded, 0 requests use it | ~1.1 GB RAM, zero per-round cost (gating skips process/draft) |
| Misroute tools→prose | −26% on that single response; target cache intact |
| DFlash impl failure mid-task | Like a broken drafter today: task abort, slot reset |

**Invariants (assertions + tests T1-T3)**: a single active impl per seq per round;
`impl_last[seq]` = active impl; switch only between tasks; trim of both ctxs at every
checkpoint.

## 6. Observability (explicit user request for troubleshooting)

Grep-able `spec-route:` markers at INFO level (3):
```
spec-route: dual mode active: draft-mtp (nextn, n_max=6) + draft-dflash (<file>, n_max=7)
spec-route: task <id> seq <n>: signal=tools|none|override:<val> → drafter=<mtp|dflash>
spec-route: cache tag mismatch (entry=<mtp>, active=<dflash>) seq <n>: target restored, draft rebuild=dflash-prefix-miss (<P> tok)
```
[Amendment 19/08 evening: kind `dflash-prefix-miss` in place of `dflash-reencode` —
no re-encode of the prefix exists in the impl; P = KV rows missing in the active ctx.
`mtp-resync` unchanged.]
Free with dual-load: `statistics draft-mtp:` and `statistics draft-dflash:` per-impl
(separate acceptance/calls) + per-task `slot print_timing`. Prometheus counters on
`/metrics`: `spec_route_requests_total{drafter}`, `spec_route_override_total`,
`spec_route_cache_rebuild_total{kind}` (cumulative: delta between reads). **In mono**
(including boot with fallback): the counters stay registered with the `drafter`
label fixed to the only active type — the policy is identity and every request counts
there. No new logs beyond level 3; body stays at 5 in short windows (known rule).

## 7. Testing (gates in sequence)

- **T1 dual-load smoke** (container :8090): both impls loaded; request without
  tools → `draft-mtp` acceptance marker; with tools → `draft-dflash`; conversation with
  alternating classes → no abort, prefill delta ~0.
- **T2 cache round-trip with alternating drafter**: 4 turns with a class switch at every
  turn; gate: zero cold-fallback of the target, expected draft rebuilds, saved prefill ≥
  threshold. **Certifies R3.3.** To be run both on the plain :8090 container AND with the
  full production config (`--parallel 4 --kv-unified --cache-ram 65535`,
  start-llama-server.sh Qwen3.8-27B branch) — parallelism is a live scenario, not a
  theoretical one.
- **T3 regression on the sacred paths**: scenarios 0006-0009 (budget-forced end, altered
  resend, trailing rollback, checkpoint restore) in dual mode with both classes;
  outcomes = mono. Same dual configuration as T2.
- **T4 A/B routing vs mono** (dedicated GPU, dedicated .md plan): 6 T7 prompts + 3
  agentic; arms MTP6-only vs DUAL(policy). Gate R3.1/R3.2 (prose ≥ −3%,
  agentic ≥ +10%).
- **T5 numerical spot-check**: greedy, DUAL-MTP vs ckpt7 and DUAL-DFlash vs DF7-today →
  divergence within the known batched-verify caveat.

## 8. Production rollout

1. Branch `drafter-routing` from `dflash2` (base = main 0a59add + 0008/0009 + dflash2
   3 commits). Durable patch `patches/drafter-routing/` one-patch-per-feature after
   T1-T3.
2. Image `docker-llm-service:vulkan-fork-dflash2-route` (~7 min).
3. T4/T5 on :8090 dedicated GPU.
4. llm-service switch (ONLY with user approval): config as §3, backup
   `.bak-20260819-ckpt7`, rollback = previous image.
5. 24h observation on `spec-route:` + counters.

**Out of scope/follow-up**: PR to charlie (DFlash2 porting + routing), fork branch push
to GitHub, slot affinity, adaptive, phase-aware.

## 9. Working constraints (for the implementer)

- **Read the ROCmFPX repo's AGENTS.md before touching code** (repo rule).
- llm-service = production managed by the user: STOPPED since 19/08 12:39 on user
  request for this work; restart only at the end of the work or on instruction. Health
  check via `docker ps` or the container IP (NOT host :1234).
- Durable patches in `patches/drafter-routing/` (format-patch --stdout with an explicit
  name); local commits on the branch for intermediate work.
- Every GPU experiment with an .md plan; TREATMENT markers in the logs; p_min always
  explicit.
- Models in `~/llmodels/` never touched. Build ONLY from Dockerfile.vulkan-rocmfpx.
- Working clone: `<lab-repo>/ROCmFPX` (remotes: origin=charlie,
  fork=pugant GitHub; push only with approval).

## 10. Quick references (code anchors, branch dflash2 — precision ±2 lines)

common.h:312-315 (single draft.mparams), :349 (types vector) · arg.cpp:3647-3656
(--spec-type list) · server-context.cpp:731-732 (singleton), 862/1032-1037
(n_rs_seq checks), 872-961 (draft load), 915-924 (mutual exclusion), 1070-1072 (slot
spec/ctx assignment), 1281-1330 (get_available_slot), 131-186 (prompt_save/load),
2020-2061 (checkpoint), 2113-2125 (idle cache), 2456-2557 (update_slots loop),
2535-2551 (dp), 2583 (trim), 2816-2907 (trailing rollback) · server-task.cpp:313-336
(body spec params), 495-524 (0008 pattern), 2447-2455/3016-3105 (disk cache) ·
speculative.cpp:48-56 (effective n_max), 73-79 (config copy), 975+ (draft_dflash),
990-1006 (DFlash state), 1082-1088 (block_size clamp), 1121-1128 (DFlash ctx flags),
1456-1495 (MTP state), 1557-1558 (pre_norm flag), 1716-1758 (MTP resync),
2941-3010 (init), 3029/3033 (impl ctor), 3131-3143 (process all), 3173-3250 (draft
loop), 3211-3225 (impl_last), 3284-3296 (get_state first-wins).
