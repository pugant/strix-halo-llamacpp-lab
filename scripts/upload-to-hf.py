#!/usr/bin/env python3
# Part of strix-halo-llamacpp-lab — see README.md.
# Replication scripts for the Qwen3.8-27B ROCmFP4-STRIX_LEAN pipeline.
# Env: LLMODELS_DIR (default $HOME/llmodels), HF_TOKEN (downloads/uploads).
# upload-to-hf.py
#
# Strategy "private → verify → public":
#   1. Create 2 PRIVATE repos
#   2. Upload all files (huggingface_hub automatic resume)
#   3. (Flip to public is manual, after user verification)
#
# Resume: huggingface_hub LFS handles automatic resume. Safe to re-run.
import os, sys, time
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
WS = os.environ.get("LAB_DIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(os.environ.get("LLMODELS_DIR") or os.path.expanduser("~/llmodels"), "models")

REPOS = [
    # grug/ornith already published on HF (2026-08-12), skipped here
    # nemotron already published (2026-08-13), skipped here
    {
        "name": "qwen36-35b-q6",
        "repo_id": "pugant/Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX",
        "files": [
            (f"{WS}/publish/qwen36-35b-q6/.gitattributes", ".gitattributes"),
            (f"{WS}/publish/qwen36-35b-q6/LICENSE",        "LICENSE"),
            (f"{MODELS}/QWEN3.6/Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX.gguf", "Qwen3.6-35B-A3B-MTP-Q6_0_ROCMFPX.gguf"),
            # README last (contains the final benches)
            (f"{WS}/publish/qwen36-35b-q6/README.md",      "README.md"),
        ],
    },
]

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

for repo in REPOS:
    log("=" * 60)
    log(f"=== {repo['name'].upper()} — {repo['repo_id']}")
    log("=" * 60)

    # Create private repo (exist_ok for re-run after partial failure)
    try:
        api.create_repo(repo_id=repo["repo_id"], repo_type="model",
                        private=True, exist_ok=True)
        log(f"[✓] Private repo ready: {repo['repo_id']}")
    except Exception as e:
        log(f"[✗] create_repo failed: {e}")
        sys.exit(1)

    # Upload file by file (small ones first, large ones after)
    for local, remote in repo["files"]:
        if not os.path.exists(local):
            log(f"[✗] MISSING: {local}")
            sys.exit(1)
        size_mb = os.path.getsize(local) / 1024 / 1024
        log(f"[i] → {remote}  ({size_mb:.1f} MB)  src={os.path.basename(local)}")
        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=remote,
                repo_id=repo["repo_id"],
                repo_type="model",
            )
            elapsed = time.time() - t0
            speed = size_mb / elapsed if elapsed > 0 else 0
            log(f"[✓] done {elapsed:.1f}s ({speed:.1f} MB/s)")
        except Exception as e:
            log(f"[✗] upload failed: {e}")
            sys.exit(1)

    log(f"[✓] REPO {repo['name'].upper()} COMPLETE")

log("=" * 60)
log("[✓✓] ALL REPOS COMPLETE — ready for verification + flip to public")
log("=" * 60)
