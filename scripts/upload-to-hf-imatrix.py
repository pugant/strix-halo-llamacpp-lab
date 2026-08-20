#!/usr/bin/env python3
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# upload-to-hf-imatrix.py — pubblica l'imatrix Qwen3.8-27B (private → verify → public)
# Autorizzazione utente: "per chiunque voglia utilizzarla" (sessione 19/08 sera).
import os, sys, time
from huggingface_hub import HfApi

api = HfApi()
WS = os.environ.get("LAB_DIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(os.environ.get("LLMODELS_DIR") or os.path.expanduser("~/llmodels"), "models")

REPO_ID = "pugant/Qwen3.8-27B-imatrix"
FILES = [
    (f"{WS}/publish/imatrix-qwen38/.gitattributes", ".gitattributes"),
    (f"{WS}/publish/imatrix-qwen38/LICENSE",        "LICENSE"),
    (f"{MODELS}/QWEN3.8/imatrix-qwen38.gguf",       "imatrix-qwen38.gguf"),
    # README per ultimo (card)
    (f"{WS}/publish/imatrix-qwen38/README.md",      "README.md"),
]

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

try:
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=True, exist_ok=True)
    log(f"[OK] Private repo ready: {REPO_ID}")
except Exception as e:
    log(f"[FAIL] create_repo: {e}"); sys.exit(1)

for local, remote in FILES:
    if not os.path.exists(local):
        log(f"[FAIL] MISSING: {local}"); sys.exit(1)
    size_mb = os.path.getsize(local) / 1024 / 1024
    log(f"[i] -> {remote}  ({size_mb:.1f} MB)")
    t0 = time.time()
    try:
        api.upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO_ID, repo_type="model")
        log(f"[OK] done {time.time()-t0:.1f}s")
    except Exception as e:
        log(f"[FAIL] upload {remote}: {e}"); sys.exit(1)

# verifica tree
files = [f.rfilename for f in api.list_repo_tree(REPO_ID, repo_type="model", recursive=True)]
log(f"[VERIFY] tree: {files}")
expected = {r for _, r in FILES}
if not expected.issubset(set(files)):
    log("[FAIL] file mancanti nel tree"); sys.exit(1)

# flip public (autorizzato)
try:
    api.update_repo_settings(repo_id=REPO_ID, repo_type="model", private=False)
    log("[OK] repo PUBLIC")
except Exception as e:
    log(f"[FAIL] flip public: {e}"); sys.exit(1)

log(f"[DONE] https://huggingface.co/{REPO_ID}")
