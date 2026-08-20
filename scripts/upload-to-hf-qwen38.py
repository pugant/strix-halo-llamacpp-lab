#!/usr/bin/env python3
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# upload-to-hf-qwen38.py — strategia private → verify → (flip public manuale, previa conferma utente)
import os, sys, time
from huggingface_hub import HfApi

api = HfApi()
WS = os.environ.get("LAB_DIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(os.environ.get("LLMODELS_DIR") or os.path.expanduser("~/llmodels"), "models")

REPO_ID = "pugant/Qwen3.8-27B-MTP-Q4_0_ROCMFP4_STRIX_LEAN"
FILES = [
    (f"{WS}/publish/qwen38-27b-q4lean/.gitattributes", ".gitattributes"),
    (f"{WS}/publish/qwen38-27b-q4lean/LICENSE",        "LICENSE"),
    (f"{MODELS}/QWEN3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf", "Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf"),
    # README per ultimo (contiene i bench definitivi)
    (f"{WS}/publish/qwen38-27b-q4lean/README.md",      "README.md"),
]

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True)
    log(f"[✓] Private repo ready: {REPO_ID}")
except Exception as e:
    log(f"[✗] create_repo failed: {e}"); sys.exit(1)

for local, remote in FILES:
    if not os.path.exists(local):
        log(f"[✗] MISSING: {local}"); sys.exit(1)
    size_mb = os.path.getsize(local) / 1024 / 1024
    log(f"[i] → {remote}  ({size_mb:.1f} MB)")
    t0 = time.time()
    try:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO_ID, repo_type="model")
        log(f"[✓] done {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"[✗] upload failed: {e}"); sys.exit(1)

log("[✓✓] UPLOAD COMPLETATO — pronto per verifica + flip public (conferma utente)")
