# Risultati — Validazione in-house Vulkan vs ROCm (Qwen3.6-35B-A3B)

**Data:** 2026-08-14 · GPU esclusiva (llm-service fermato durante i run) · llama-bench `-ngl 999 -fa 1 -p 512 -n 128`
Immagini: `docker-llm-service:latest` (ROCm, build 00d5452) · `docker-llm-service:vulkan-backup` (Vulkan, build 3f7c79d7b)

## Tabella riassuntiva — Qwen3.6-35B-A3B, stessa macchina, stessa sessione

| Backend | Quant | Size | pp512 (tok/s) | tg128 (tok/s) |
|---|---|---|---|---|
| **ROCm** | **Q4_0_ROCMFP4_STRIX_LEAN** (produzione) | 17.73 GiB | **1420.72 ± 14** | **71.23 ± 0.5** |
| ROCm | UD-Q5_K_M_MTP | 25.22 GiB | 1359.39 ± 12 | 50.76 ± 0.2 |
| **Vulkan RADV** (mesa) | UD-Q5_K_M_MTP | 25.22 GiB | 1008.09 ± 14 | **57.93 ± 0.7** |
| Vulkan AMDVLK | UD-Q5_K_M_MTP | 25.22 GiB | 663.02 ± 6 | 55.84 ± 0.7 |

## Lettura dei dati

1. **Controllo backend-only (stesso identico GGUF UD-Q5_K_M_MTP):** RADV batte ROCm su tg128 di
   **+14.1%** (57.93 vs 50.76) → il claim community "~+18% tg per Vulkan RADV" **replica in-house**
   entro la tolleranza metodologica. Su pp512 ROCm vince **+34.8%** (1359 vs 1008) → anche questo
   conferma la community (ROCm/AMDVLK vincono pp).
2. **Confronto stack-vs-stack (ciò che conta in produzione):** il nostro stack ROCmFP4 vince su
   entrambi i fronti: tg128 **+22.9%** (71.23 vs 57.93) e pp512 **+40.9%** (1420 vs 1008). Il motivo
   è il formato: ROCmFP4 pesa 17.7 GiB vs 25.2 GiB del Q5_K_M → meno banda LPDDR5X consumata a
   uguale qualità percettiva.
3. **RADV >> AMDVLK**, come da community: qui la differenza è enorme su pp512 (+52%, 1008 vs 663)
   e lieve su tg (+3.7%). AMDVLK va considerato obsoleto.

## Note operative (per riprodurre)

- ICD RADV nel container vulkan-backup: `-e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`
  (⚠️ il nome file esatto ha il suffisso `.x86_64`; con il nome sbagliato ggml-vulkan **cade
  silenziosamente su backend CPU** senza alcun errore — verificare sempre la riga `ggml_vulkan: 0 = ... (radv)`
  nell'output).
- ROCm richiede `--device /dev/kfd --device /dev/dri --group-add video --group-add render` +
  `HSA_OVERRIDE_GFX_VERSION=11.5.1`; Vulkan basta `/dev/dri` + `render`.
- File grezzi: `bench-qwen36-35b-rocmfp4.txt`, `bench-qwen36-35b-udq5km-rocm.txt`,
  `bench-qwen36-35b-udq5km-vulkan-radv.txt`, `bench-qwen36-35b-udq5km-vulkan-amdlvk.txt`.
- Baseline storica produzione "63 tok/s plain" (server-timing, design grug 11/08) coeva e coerente
  con i 71.2 llama-bench di oggi (server-timing < llama-bench).

## Conclusione

Il quadro community è **confermato a parità di quant** (RADV +14% tg), ma la scelta di stack resta
**ROCmFP4**: vince tg e pp grazie al formato fp4 più leggero, oltre a MTP e capacity. Vulkan RADV
resta il piano B (driver userspace, nessun bisogno del fork ROCmFPX per quants standard).
