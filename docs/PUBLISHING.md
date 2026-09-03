# Publishing checklist

Reusable checklist for every feature release of this lab. Three blocks, in order.

## 1. Lab repo

- [ ] Patch series in `patches/<series>/` — one patch per commit, `git am` verified
      on a clean tree.
- [ ] `rocmfpx/` snapshot updated to the series result (`git rm -r` + `git archive`
      re-import).
- [ ] `PATCHES.md`: series row + apply-note chain (fork base → series → result).
- [ ] EN write-up (dated note in `docs/experiments/` or a new thread) and/or a
      `docs/guide/` replication guide; link it from `docs/experiments/README.md`.
- [ ] README: credits and capability lines updated when the feature ships.
- [ ] New experiment notes carry the bare-metal footer:
      `*Measured on the lab's [bare-metal Strix Halo](../../BARE-METAL.md), configuration as of the note's date.*`
      (render-check the footer link). Machine config changes → update `BARE-METAL.md`
      (Current + changelog row) in the same push.
- [ ] Push GitHub (`git push origin main`) AND the Gitea mirror
      (`git -c credential.helper=store push gitea main`) in the same step.

## 2. Model cards (Hugging Face)

- [ ] Draft card edits in `docs/.drafts/` (gitignored — never committed). Start
      from a snapshot: `curl -sf https://huggingface.co/<repo>/raw/main/README.md
      -o docs/.drafts/<key>.orig.md` (that snapshot is also the rollback).
- [ ] Only numbers already published in this lab; only features the model can
      actually exercise. Use the T26 standard blocks (`## Runtime`, or the rich
      `## Run it with our engine` block for the flagship).
- [ ] Upload ONE repo at a time, with the write token already in
      `~/.cache/huggingface/token`, `HF_HUB_DISABLE_XET=1` in the environment:

      ```bash
      SP=$(python3 -m site --user-site)   # where pip --user keeps huggingface_hub
      # (adjust if your tools live under another interpreter's user site)
      PYTHONPATH=$SP HF_HUB_DISABLE_XET=1 python3 -c "
      from huggingface_hub import HfApi
      HfApi().upload_file(path_or_fileobj='docs/.drafts/<key>.md',
                          path_in_repo='README.md',
                          repo_id='pugant/<repo>', repo_type='model')"
      ```

- [ ] After each upload: fetch the raw README back and `diff` it against the
      draft; check the rendered card page.

## 3. Cross-checks (final)

- [ ] Sensitive-pattern gate on the tree and on every card diff (grep must return nothing):
      `grep -rInE 'bosgame|192\.168\.|hf_[A-Za-z0-9]{20}|ghp_|github_pat_|:8193|:8195|:8081' .`
      (known benign exceptions: upstream `rocmfpx/` vendored docs, `localhost:1234` examples).
- [ ] Bidirectional links: card → lab `main`; lab engine table → cards; guides.
- [ ] Sensitive scan of every diff: no home paths, internal ports, private names.
- [ ] Two-stage review (numbers-vs-sources, then EN pass).
- [ ] User GO before every push/upload. Link check (`curl -sI`) on every new link.
