# Costo checkpoint/ring e boost MTP vs contesto (3.2)

**Data:** 2026-08-17 16:55-17:05 · GPU dedicata · stessa immagine/modello/greedy del 3.1 ·
stesso prompt per tutti i bracci (seed fisso; NON confrontabile 1:1 col 3.1, corpus
variato). Warm-up scartato, mean±sd su 2 rep.

| Braccio | 1k | 24k | Lettura |
|---|---|---|---|
| T1 base (MTP n6, default) | 28.89 ± 0.01 | **23.47 ± 0.03** | baseline |
| T2 `-cpent -1` (no ckpt prefill) | — | 24.34 ± 0.27 | **+3.7%** (~3σ, piccolo) |
| T3 `-ctxcp 4` (ring 4) | — | 22.83 ± 1.0 | −2.7% NON significativo (sd ±1.0) |
| T4 senza MTP | 9.33 ± 0.3 | 7.66 ± 0.06 | floor puro |

## Conclusioni

1. **Checkpoint/ring: leva morta.** Creare checkpoint in prefill costa ≤4%; il ring 32 vs
   4 è indistinguibile. I default del fork sono già ottimali lato tg. (E `-cpent -1` in
   produzione romperebbe il checkpoint-salvage 0007 per un guadagno ~nullo.) Sub-filone
   3.2 CHIUSO.
2. **Boost MTP COSTANTE col contesto**: boost(1k) = 28.89/9.33 = **3.09×**, boost(24k) =
   23.47/7.66 = **3.06×**. L'acceptance NON degrada a ctx lungo: l'MTP resta la leva
   migliore e robusta. (Dato di base anche per il filone 2: la prosa penalizza
   l'acceptance ASSOLUTA, non quella relativa al contesto.)
3. **Il calo 1k→24k (−19%) è TUTTO nel floor** (9.33→7.66 senza MTP; con MTP 28.89→23.47
   = −19% uguale: il boost moltiplicativo amplifica il floor, non lo causa). E il 3.1 ha
   già escluso la via "KV q8_0" (kernel dequant Vulkan costa più della banda risparmiata).
   → Il margine floor rimasto è in **overhead per-token** (kernel launch/dispatch) e
   attenzione con KV grande: rimanda ai sub-filoni **3.3 (HIP/Vulkan graphs)** e **3.4
   (fused verifier/kernels)**.

Raw: /tmp/bench-kv-T{1..4}*.json, log /tmp/bench-ckpt-orchestrator.log (effimeri).
