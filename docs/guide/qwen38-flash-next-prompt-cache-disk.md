# Persistent prompt cache — guide (any model on this fork, shown on `qwen4exp`)

How a server restart stops costing a full re-prefill of every client session: the on-SSD prompt-cache library outlives the process, and a 107k-token context comes back in **1.57 s** instead of the **920 s** a cold re-prefill of the same tokens measured on this machine.

Deployed on a real always-on multimodal agent server since **2026-09-01**; rolling back is removing the flags (see [Rollback](#rollback)).

## What it is

The fork's server already keeps an SSD-backed prompt cache (`--cache-disk`) for mid-session reuse — but its metadata lives in RAM and is wiped at boot, so every restart starts from an empty cache. `--cache-disk-persist` adds a **second, durable tier**: entries are promoted into a library under `<cache-dir>/.llama-prompt-cache-v1/persist/entry-<id>/` — a saved state payload for the target (plus the drafter's, when speculative decoding is on) and an 88-byte little-endian sidecar with the configuration fingerprint, token ids, hit count, timestamps and CRC-32s.

The design is inspired by **antirez's [`ds4_kvstore`](https://github.com/antirez/ds4)** (the same lineage as the `--ple-disk` offload's dwarstar):

- **hit-decay eviction** — entry score decays with a 6 h half-life; an entry no session revisits ages itself out of the budget;
- **boot-time adoption** — the next boot adopts valid entries, discards corrupted ones (sidecar magic/CRC), keeps other-configuration entries aside and garbage-collects them oldest-first;
- **crash-safe commit by rename** — an entry exists iff its sidecar does, and the sidecar is written last (payloads → fsync → temp sidecar → fsync → rename → fsync dir); a crash mid-save leaves only nameless temporaries that the next boot sweeps;
- **one writer** — a second server on the same cache-dir finds the library locked, disables the persistent tier, and keeps its own per-run cache.

## Read this first: the verbatim requirement

With the MTP drafter active (the configuration this was built for), a restore is only valid at an **exact token boundary**: the drafter's state is consistent only where the longest common prefix equals the cached token count, and the trailing-rollback salvage that absorbs small deltas mid-session does not apply across a restart.

- **What restores**: a **verbatim replay** — resend exactly the tokens already served (raw `/completion`), then the new turn as a delta. Measured: `restored=107283, prefilled=31`.
- **What does not**: a **chat-template history replay**. Re-rendering the assistant turns is not token-identical to what the model generated (an unfinished generation is unclosed reasoning), the junctions shift, the boundary check fails — and the request silently falls back to a full prefill. No error is raised; watch `restored` in the logs to know which path you are on.

Whether your client can speak verbatim is a property of the client. Also note a bare resend of a stored prompt is its own strict prefix and exercises a different path — the intended use is **extending** a restored context.

## How to enable it

```bash
llama-server -m Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  -ngl 999 --jinja -c 16384 --no-mmap --metrics \
  --cache-disk /path/to/prompt-cache --cache-disk-persist
```

| Flag | Meaning |
|---|---|
| `--cache-disk PATH` | the SSD cache directory (**required** for the persistent tier) |
| `--cache-disk-persist` | turn the persistent library on (`--no-cache-disk-persist`, default off) |
| `--cache-disk-persist-mib N` | library budget in MiB (**default 16384, minimum 1024**) |
| `--cache-disk-persist-min-tokens N` | do not save prompts shorter than N tokens (**default 1024**) |

Boot fails fast if `--cache-disk-persist` is set without a `--cache-disk` path or with the per-run limit (`--cache-disk-limit`, default 8192 MiB) disabled. The flags compose with `--ple-disk` and `-md <drafter>` — that combination is the production configuration. Pair the directory with real disk (NVMe): a ~107k-token entry is ~2.2 GiB on disk.

## What to expect

- **Restore**: the state load for a 107,283-token entry measured **1.36–1.57 s** (the end-to-end first token at 14.3 s vs 920 s cold: **64×**). Each restore additionally pays one full CRC read of the payload, which the kernel absorbs once.
- **Save**: ~0.4 s per save in production (a 1,449-token multimodal prompt, 133 MB); saves happen after a completed request, off the hot path.
- **RAM: unchanged.** This is a latency feature; the entry lives on SSD inside its own budget. (The RAM relief on this model comes from `--ple-disk`, which composes with this.)
- **Determinism**: the same library rebuilt in an independent boot re-served the same request char-identical (111/111 characters) — the restore is deterministic across restarts.

## Monitoring

Twelve `llamacpp:persist_*` counters and gauges on `/metrics` (the endpoint requires `--metrics`, else the server returns 501):

| Metric | Meaning |
|---|---|
| `llamacpp:persist_saves_total` / `loads_total` / `hits_total` | library writes, restores, hits on an already-restored prefix |
| `llamacpp:persist_evictions_total` / `gc_orphans_total` | budget evictions / other-config entries collected |
| `llamacpp:persist_bytes_written_total` / `bytes_read_total` | library byte traffic |
| `llamacpp:persist_last_restore_ms` / `last_tokens_restored` / `last_tokens_prefilled` | the last restore, decomposed |

The quick health check is the boot line in the logs — `persist boot adopt: entries=N orphans=N gc=N bytes=N budget_bytes=N` — and, after a request you expect to restore, `persist load: entry=N restored=R prefilled=P ms=M`: **R is the number that tells you the restore fired** (P alone can be a silent full prefill).

```bash
curl -s http://127.0.0.1:8080/metrics | grep '^llamacpp:persist_'
```

## FAQ

**Does it change the output?** No: the restored state is a saved state of this engine, and the cross-restart determinism gate re-served the identical response (111/111 characters).

**Does it work with the external MTP drafter?** Yes — that is the production configuration and the reason for the verbatim requirement above.

**Does it work with vision/multimodal prompts?** Yes, in production since 2026-09-01 (the first deploy iteration had three save-path bugs on multimodal slots, all fixed at the root: the save path now keys on a prompt's actual media cells; a multimodal prompt saves its text-token view).

**What happens if the server crashes mid-save?** Nothing is half-committed: without a renamed sidecar the entry does not exist, and the next boot discards the nameless temporaries.

**Can two servers share one cache directory?** Not the persistent library: the first server holds an exclusive lock; the second disables its persistent tier (a log line says so) and keeps its per-run cache.

## Rollback

Remove `--cache-disk-persist` (and `--cache-disk` if it was added for it) and restart: the library on disk is left alone — never read, never written — and behaves exactly as before the feature existed. No data migration in either direction; deleting the `persist/` subtree under the cache dir reclaims the space.

For the measured gates (restore vs cold re-prefill, determinism, eviction/GC, boot-discard) see the thread note ([`../experiments/qwen38-flash-next-runtime.md`](../experiments/qwen38-flash-next-runtime.md) §8) and the implementation series [`patches/t23-kv-disk-persist/`](../../patches/t23-kv-disk-persist/) (12 patches), already part of the [`rocmfpx/`](../../rocmfpx/) snapshot.
