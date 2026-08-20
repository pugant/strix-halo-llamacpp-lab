#!/usr/bin/env python3
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# Corpus sources (vars below): AGENTIC = $LLMODELS_DIR/calibration/grug-calibration.txt
#   (agentic-coding EN corpus produced by the grug pipeline, see docs/design/
#   2026-08-11-grug-35b-v2-strix-lean-design.md); DOCS_GLOB/CODE_GLOBS = this
#   repo's docs/*.md (Italian technical prose) and scripts/*.sh|*.py (real code).
#   Override LAB_DIR to point at a different checkout of this lab.
"""Prepara il corpus di calibrazione per l'imatrix di Qwen3.8-27B.

Fonti interlacciate (round-robin su segmenti, così i primi ~131k token
usati da llama-imatrix --chunks 256 sono un mix bilanciato):
  1. grug-calibration.txt  — agentic coding EN (tool-call, codice, SWE-bench)
  2. docs/*.md repo         — prosa tecnica italiana
  3. sorgenti .py/.sh       — codice reale locale (copertura tokenizzazione codice)

Output: $LLMODELS_DIR/calibration/qwen38-calibration.txt
"""
import glob
import os
import random
import re

random.seed(38)

LLMODELS = os.environ.get("LLMODELS_DIR") or os.path.expanduser("~/llmodels")
LAB_DIR = os.environ.get("LAB_DIR") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
CAL_DIR = os.path.join(LLMODELS, "calibration")
AGENTIC = os.path.join(CAL_DIR, "grug-calibration.txt")
DOCS_GLOB = os.path.join(LAB_DIR, "docs", "**", "*.md")
CODE_GLOBS = [
    os.path.join(LAB_DIR, "scripts", "*.sh"),
    os.path.join(LAB_DIR, "scripts", "*.py"),
]
OUT = os.path.join(CAL_DIR, "qwen38-calibration.txt")
TARGET_MB = 2.5          # ordine di grandezza; l'interlacciamento garantisce il mix nei primi 500 KB
SEGMENT_CHARS = 3000     # segmenti ~750 token ciascuno


def load_segments(paths, max_total_chars):
    """Splitta i file in segmenti e campiona fino a max_total_chars."""
    segs = []
    for p in paths:
        try:
            text = open(p, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        # saltare file minuscoli e righe palesemente non-testo (binari)
        if len(text) < 500:
            continue
        for i in range(0, len(text), SEGMENT_CHARS):
            s = text[i:i + SEGMENT_CHARS].strip()
            if len(s) > 200:
                segs.append(s)
    random.shuffle(segs)
    out, tot = [], 0
    for s in segs:
        out.append(s)
        tot += len(s)
        if tot >= max_total_chars:
            break
    return out


def main():
    agentic = load_segments([AGENTIC], int(1.4 * 1024 * 1024))          # ~1.4 MB (55%)
    italian = load_segments(sorted(glob.glob(DOCS_GLOB, recursive=True)), int(0.7 * 1024 * 1024))  # ~0.7 MB (28%)
    code = load_segments([p for g in CODE_GLOBS for p in sorted(glob.glob(g))], int(0.4 * 1024 * 1024))  # ~0.4 MB (17%)

    # interlacciamento: 3 agentic : 2 italiano : 1 codice → mix stabile su ogni finestra
    inter = []
    ia, ii, ic = iter(agentic), iter(italian), iter(code)
    while True:
        added = False
        for it, n in ((ia, 3), (ii, 2), (ic, 1)):
            for _ in range(n):
                try:
                    inter.append(next(it))
                    added = True
                except StopIteration:
                    pass
        if not added:
            break

    text = "\n\n".join(inter)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    n_chars = len(text)
    print(f"OK {OUT}: {n_chars/1024:.0f} KB — segmenti: agentic={len(agentic)} italiano={len(italian)} codice={len(code)}")
    # sanity: i primi 100 KB devono contenere tutti e 3 i componenti
    head = text[:100_000]
    print(f"mix nei primi 100KB: tool_call={'<tool_call>' in head}, docs-IT={'## ' in head}, codice={'def ' in head or 'python3' in head}")


if __name__ == "__main__":
    main()
