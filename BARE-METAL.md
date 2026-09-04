# Bare-metal configuration

Every number this lab publishes was measured on one machine. This page is its
full, dated configuration — the context for every benchmark note and model card.

## Current configuration

As of 2026-09-03.

| Component | Value |
|---|---|
| SoC | AMD Ryzen AI MAX+ 395 with Radeon 8060S iGPU — 16 cores / 32 threads |
| RAM | 128 GB unified LPDDR5X (~256 GB/s UMA bandwidth; some notes round to ~270) — ~122 GiB visible to the OS (source: README hardware line, `vulkan-nommap-backend.md`) |
| Storage | Kingston OM8TAP42048K1-A00 1.9 TB NVMe (PCIe 4, QLC, DRAM-less) |
| OS / kernel | Ubuntu 26.04.1 LTS / 7.0.0-30-generic |
| Graphics driver | Mesa RADV 26.0.8-1ubuntu0.3 (Vulkan, primary backend) |
| ROCm | 7.2.4 (HIP backend) |
| Docker | server 29.7.2 |
| Power profile | balanced (never forced to performance during measurements) |
| UMA / VRAM | VRAM partition 512 MB; HIP inference requires `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` (source: nemotron note); GPU-addressable unified memory via kernel parameters — see [below](#unified-memory-kernel-parameters) |
| Clock behavior | sustained ~2220 MHz plateau under decode load (source: `docs/experiments/vulkan-nommap-backend.md`) |

### Unified memory kernel parameters

Out of the box the amdgpu driver caps GPU-addressable memory well below what this
machine physically has. Every number measured since 2026-08-30 assumes the
following kernel parameters, set in `/etc/default/grub`
(`GRUB_CMDLINE_LINUX_DEFAULT`) and active since:

| Parameter | Value | Effect |
|---|---|---|
| `amdgpu.gttsize` | `126976` | GTT size in MB — 124 GiB of unified memory GPU-addressable |
| `ttm.pages_limit` | `32505856` | TTM page limit (× 4 KiB ≈ 124 GiB) |
| `ttm.page_pool_size` | `32505856` | TTM page pool (× 4 KiB ≈ 124 GiB) |
| `amd_iommu` | `off` | IOMMU off — part of the toolboxes' recommended baseline for these workloads |
| `amdgpu.lockup_timeout` | `60000` | GPU lockup timeout raised to 60 s |
| `amdgpu.timeout_fatal_disable` | `1` | GPU timeouts do not escalate to fatal resets |

To apply: edit `/etc/default/grub`, run `sudo update-grub`, reboot. Verify at
runtime:

```console
$ cat /sys/class/drm/card*/device/mem_info_gtt_total
133143986176   ← 124.0 GiB, measured 2026-09-04
```

Settings of this kind are documented in:
- Donato Capitella's Strix Halo configuration video — <https://www.youtube.com/watch?v=sMacAPpgXhQ>
- Strix Halo toolboxes — <https://strix-halo-toolboxes.com/>
- Frame.work community guide (Fedora 42) — <https://community.frame.work/t/amd-strix-halo-llama-cpp-installation-guide-for-fedora-42/75856>
- kyuz0/amd-strix-halo-llm-finetuning — <https://github.com/kyuz0/amd-strix-halo-llm-finetuning>

## How every number in this lab is measured

- Every measurement runs in a container of the tagged image named in the experiment's note (primary source: the note; the README engine section links the notes).
- Vulkan measurements use `--no-mmap` (mmap changes pp ~3× — see `docs/experiments/vulkan-nommap-backend.md`).
- Comparison benchmarks are greedy, same-mode, identical flags across arms (`docs/experiments/results-2026-08-17-rocm-vs-vulkan-tg.md`).
- 5 repetitions where the note declares them; the published number is the note's, never a re-average.
- The power profile is never touched during measurements.
- Perplexity corpora and protocol: `docs/experiments/results-2026-08-17-ppl-strategy-phaseA.md`.

## Configuration changelog

| Date | Change | Source |
|---|---|---|
| 2026-08-10 | ROCm 7.2.4 already in use (install date unknown) | `docs/experiments/results-2026-08-10.md` |
| 2026-08-12 | Power profile `balanced` observed (not forced) | `docs/experiments/results-2026-08-12-publish.md` |
| 2026-08-22 | Kernel 7.0.0-30-generic installed via apt | system apt history (`/var/log/apt/history.log.1.gz`) |
| 2026-08-29 | Production switched to Vulkan + `--no-mmap` + KV q8_0 | `docs/experiments/vulkan-nommap-backend.md` |
| 2026-08-30 | Unified-memory kernel parameters applied (`amdgpu.gttsize=126976`, TTM limits) — GTT 124 GiB verified at runtime | `/etc/default/grub`; `mem_info_gtt_total` |

OS/point-release history predates the lab's notes; the kernel row reflects the package
install dates, not first boot.

## What we deliberately don't publish

LAN topology and internal addresses, hostname, serial numbers, machine-specific local paths. We
publish the machine class and the software stack, not the identity of the
installation.
