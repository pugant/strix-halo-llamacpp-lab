#!/usr/bin/env python3
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# check-imatrix-coverage.py — GATE copertura imatrix prima della quantizzazione
# Formato imatrix del fork: entries "<tensor>.in_sum2" + "<tensor>.counts" (coppie).
# Modello BF16 multi-shard: le tensor info sono distribuite sui 2 shard → si leggono entrambi.
# MTP (blk.N ultimo, tensori nextn.*): NON copribile via imatrix (il layer non è eseguito in
# forward) — atteso mancante, path unweighted in quantize (idem imatrix unsloth Qwen3.6).
# Uso (container convert): python3 check-imatrix-coverage.py <shard1.gguf> <shard2.gguf> <imatrix.gguf>
import re
import sys

import numpy as np
from gguf import GGUFReader

shard1, shard2, imatrix_path = sys.argv[1], sys.argv[2], sys.argv[3]

model_tensors = {}
for path in (shard1, shard2):
    for t in GGUFReader(path).tensors:
        if len(t.shape) == 2:
            model_tensors[t.name] = tuple(t.shape)

ri = GGUFReader(imatrix_path)
imatrix = {}
for t in ri.tensors:
    base = re.sub(r"\.(in_sum2|counts)$", "", t.name)
    if t.name.endswith(".in_sum2"):
        imatrix[base] = np.ndarray(t.shape, dtype=np.float32, buffer=t.data)

print(f"MODELLO: {len(model_tensors)} tensori 2D (da 2 shard)")
print(f"IMATRIX: {len(imatrix)} tensori coperti (entries in_sum2)")

matched = set(imatrix) & set(model_tensors)
missing = set(model_tensors) - set(imatrix)
extra = set(imatrix) - set(model_tensors)
print(f"match={len(matched)}  extra={len(extra)}  mancanti={len(missing)}")

blk_m = sorted({int(m.group(1)) for n in matched for m in [re.match(r"blk\.(\d+)\.", n)] if m})
blk_t = sorted({int(m.group(1)) for n in model_tensors for m in [re.match(r"blk\.(\d+)\.", n)] if m})
print(f"layer coperti: {len(blk_m)}/{len(blk_t)} (range {blk_m[0]}..{blk_m[-1]}, modello {blk_t[0]}..{blk_t[-1]})")

mtp_layer = blk_t[-1]
# attesi senza entry per design degli observer (hook solo su matmul; MTP non eseguito in forward;
# stesso set dell'imatrix unsloth Qwen3.6/3.8: entries_count=496)
EXPECTED_MISSING = lambda n: (n.startswith(f"blk.{mtp_layer}.")
                              or n == "token_embd.weight"
                              or n == "output.weight"          # lm_head: no observer per design
                              or n.endswith(".ssm_conv1d.weight"))
mtp_missing = [n for n in missing if n.startswith(f"blk.{mtp_layer}.")]
other_missing = [n for n in missing if not EXPECTED_MISSING(n)]
print(f"MTP blk.{mtp_layer}: {len(mtp_missing)} tensori senza entry (atteso: non eseguito in forward)")
print(f"conv1d+emb senza entry (atteso, no observer): {len(missing)-len(other_missing)-len(mtp_missing)}")
print(f"NON-ATTESI mancanti: {len(other_missing)}", other_missing[:15])

from collections import Counter
def cat(n):
    if ".ssm_" in n or ".attn_gate" in n: return "ssm/deltanet"
    if ".nextn." in n: return "nextn (MTP)"
    if ".attn_" in n: return "attention"
    if ".ffn_" in n: return "ffn"
    if "token_embd" in n or "output" in n: return "emb/out"
    return "altro"
mcat, tcat = Counter(cat(n) for n in matched), Counter(cat(n) for n in model_tensors)
for c in tcat:
    print(f"  {c:14s}: {mcat.get(c,0)}/{tcat[c]}")

nan_tot = inf_tot = shape_bad = 0
for n in matched:
    a = imatrix[n]
    if not np.isfinite(a).all():
        nan_tot += int(np.isnan(a).sum()); inf_tot += int(np.isinf(a).sum())
    if a.shape[0] != model_tensors[n][0]:  # ne0 = n_per_row
        shape_bad += 1
print(f"sanità: NaN={nan_tot} Inf={inf_tot} shape-mismatch={shape_bad}")

# gate: tutti i layer non-MTP coperti (mancanti solo tensori MTP o casi limite), sanità ok
expected_total = len(model_tensors) - len([n for n in missing if EXPECTED_MISSING(n)]) - len(other_missing)
ok = (len(other_missing) == 0 and len(matched) == expected_total
      and blk_m and blk_m[-1] >= blk_t[-2]   # tutto tranne l'ultimo layer (MTP)
      and nan_tot == 0 and inf_tot == 0 and shape_bad == 0)
print(f"\nGATE COVERAGE: {'PASS ✓' if ok else 'FAIL — rivedere corpus/chunks'}")
sys.exit(0 if ok else 1)
