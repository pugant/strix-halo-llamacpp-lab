# Qwen3.8-Flash-Next PLE disk-offload — guide (the `qwen4exp` arch)

How the 98.5 GiB Qwen3.8-Flash-Next quant ([`pugant/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-GGUF`](https://huggingface.co/pugant/Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN-GGUF)) runs on one 128 GB Strix Halo machine with ~36 GB of RAM to spare: the biggest tensor in the model is never loaded.

Deployed on a real always-on agent server since **2026-08-31**; rolling back is one flag and one restart (see [Rollback](#rollback)).

## What it is

The qwen4exp architecture carries a **PLE n-gram table** (`per_layer_token_embd`, 51B parameters) that is **35.76 GiB of the 98.5 GiB GGUF** — more than a third of the file, consulted as a read-only gather target during decode. With `--ple-disk` the table is **not loaded at model load**: its rows are read **on demand from the GGUF file itself** via `pread`, through a fixed-size LRU cache of whole 128-row blocks kept in the on-disk (compressed) format.

- **No sidecar files** — nothing is extracted or preprocessed; the published GGUF is used verbatim.
- **Zero preprocessing** — no conversion step, no extra disk usage; the flag changes where the table lives, not what the model computes.
- **Bit-exact by construction** — the store serves dequantized rows identical to the in-memory path (unit-tested bit-exact against the CPU `get_rows` path, with a ULP-level parity fallback for the Vulkan gather).

The design and the A/B scripts live in the patch series [`patches/t25-ple-disk/`](../../patches/t25-ple-disk/) (15 patches: 12 base + 3 v2; the char-identical check and the tg/pp A/B bench are patches 0011/0012), already part of the [`rocmfpx/`](../../rocmfpx/) snapshot — no patching needed, just build it. Arriving from the HF card and want to go start-to-finish? The [Build section of the root README](../../README.md#build) builds this engine from [`docker/Dockerfile.vulkan-rocmfpx-local`](../../docker/Dockerfile.vulkan-rocmfpx-local).

## The memory math

| Piece | Where it lives with `--ple-disk` |
|---|---|
| PLE n-gram table, 35.76 GiB | **on disk** (GGUF itself), RAM only for cached blocks |
| Everything else (~62.7 GiB of weights) | RAM/GPU as usual |
| Block cache | `--ple-cache-mib`, default 4096 MiB |
| KV cache | sized by `-c`/cache-RAM — grows with context |
| External MTP drafter | ~3.85 GiB at Q8_0 |
| GDN/RS rollback state ring | ~7.2 GiB of buffers |
| Vulkan/runtime/OS overhead | a few GiB |

On the 128 GB machine that arithmetic leaves ~36 GB free in practice, in our production instance — headroom for the KV cache at long agent contexts, instead of a model that barely fits with none.

## How to enable it

```bash
llama-server -m Qwen3.8-Flash-Next-Q4_0_ROCMFP4_STRIX_LEAN.gguf \
  -ngl 999 --jinja -c 16384 --no-mmap --ple-disk --metrics
```

- `--ple-disk` — turn the offload on (`--no-ple-disk` to disable; default off).
- `--ple-cache-mib N` — block-cache budget in MiB, **default 4096**; the server rejects values **below 102**.
- `--no-mmap` — the offload **pairs with `--no-mmap`** in our production combo: with mmap the kernel may fault the table pages in wholesale and the offload loses its meaning (on Vulkan, mmap also collapses prompt processing — see [Performance & hardware](../experiments/README.md#performance--hardware)). With `--no-mmap`, budget the rest of RAM for the non-PLE weights, KV and the block cache.

The same two flags pass through `llama-bench` (`--ple-disk`, `--ple-cache-mib`), which is how the A/B numbers were taken.

## What to expect

- **Warm page cache → ~3% tg (token generation) cost.** With the kernel page cache already holding the touched blocks, measured tg was 30.45 vs ~31.5 tok/s with the table fully in RAM — the price of the on-demand gather path.
- **Cold first touch is slower** — every block pays its first physical read from disk (the first real encounter with those bytes). This is a warm-up cost, not a steady-state one.
- **Fully cold cache (bench from `drop_caches`): pp (prompt processing) 512 −58% / pp2048 −52%** — the one regime where the offload genuinely hurts is prompt processing on a cold cache.
- **The kernel page cache is the real L2.** Once a block has been touched it stays in RAM until evicted, so repeated workloads (an always-on agent server) run at the warm figures above. The v2 patches in the series (per-block dedup + `POSIX_FADV_WILLNEED` readahead submission) close most of the residual gap in the kernel-warm regime.

The three regimes, in one table:

| Regime | What you pay | Typical figure |
|---|---|---|
| Block cache warm (steady server) | gather-path overhead only | tg ~3% (30.45 vs ~31.5 tok/s) |
| Kernel page cache warm, block cache cold | dequant + copy, no disk I/O | small; v2 closes most of the residual gap in this regime |
| Everything cold (fresh boot / `drop_caches`) | first physical read of every touched block | pp512 −58% / pp2048 −52% |

Practical reading: for a long-lived server this is close to free; for one-shot cold bench runs, expect the cold numbers.

## FAQ

**Does it change the output?** No. The A/B ran the same greedy workload with and without the flag and the outputs were **char-identical**; the store is unit-tested bit-exact against the in-memory gather path.

**Does it work together with the external MTP drafter?** Yes — that is the production configuration; the published A/B benches were run with the MTP drafter active (`tg`/`pp`, 5 reps, median).

**Does it work on both backends?** The row gather is host-side, so the store is backend-neutral; what we measured — and what the series parity-checks — is the **Vulkan/RADV** build (the series carries a VK-vs-CPU `get_rows` parity utility with a ULP fallback).

**Is the GGUF modified, or is anything written next to it?** No and no. The model file is read-only input; there are no sidecar files and no preprocessing step.

## Monitoring

Three Prometheus counters on `/metrics` — the endpoint requires `--metrics` on the command line (without it the server returns **501**):

| Counter | Meaning |
|---|---|
| `llamacpp:ple_hits_total` | row lookups served from the RAM block cache |
| `llamacpp:ple_misses_total` | row lookups that required a disk read |
| `llamacpp:ple_blocks_read_total` | 128-row blocks read from disk |

The hit/miss ratio tells you whether your `--ple-cache-mib` budget covers the working set: a climbing miss rate on a steady workload means the cache is thrashing — raise the budget, or accept the disk reads (which the kernel page cache absorbs after first touch anyway).

Sizing note: the cache holds blocks in their on-disk compressed format, so 4096 MiB of cache covers substantially more than 4 GiB of the table's logical rows; there is rarely a reason to go below the default.

Quick check of the counters:

```bash
curl -s http://127.0.0.1:8080/metrics | grep '^llamacpp:ple_'
```

## Rollback

Remove `--ple-disk` from the server command line and restart: the model loads with the PLE table in RAM exactly as before the feature existed. There is no data migration, no extra file to clean up, and the GGUF is byte-identical either way — rollback cost is one model load.

For the measured A/B (char-identical output check, tg/pp cost, the v2 gap reduction) see the A/B scripts in the patch series ([`patches/t25-ple-disk/`](../../patches/t25-ple-disk/), patches 0011/0012) and the series index in [`PATCHES.md`](../../PATCHES.md).
