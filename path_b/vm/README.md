# path_b/vm/ — set up a new GPU instance, save and restore outputs

A GPU instance can be deleted whenever it is idle (on Verda, a shut-down instance keeps billing; a
deleted one does not). Code lives in git, results on Hugging Face or the Mac, and `setup.sh`
rebuilds the rest.

## One command each way (from the Mac, repo root)

| When | Command | What it does |
|---|---|---|
| Before deleting an instance | `path_b/vm/save_instance.sh` | Checks the instance's code is clean and known to the Mac; copies every output file Hugging Face does not hold (`hf_outputs.py pending`) to `path_b/data/outputs/<run>/` and the logs to `~/V2D/vm_logs/<instance>_<date>/` with a `SAVED.md`; ends with "safe to delete" or what is missing |
| Right before deleting it | `path_b/vm/save_instance.sh --remove-deploy-key` | The same, then deletes the instance's deploy key from GitHub (it needs the instance reachable, and the key is useless once the instance is gone) |
| A new instance is up | `path_b/vm/new_instance.sh <ip>` | Points the `l40s` alias at `<ip>`; puts `setup.sh`, the MANO files and `path_b/.env` on it; gives it a Hugging Face token and a read-only deploy key (added with `gh`); runs `setup.sh` in tmux and follows it; pushes this checkout's branch through the bare repo and runs the unit tests. Options: `--restore RUN` (default `4c2eac31`), `--no-restore`, `--no-frames`, `--branch B` |

`new_instance.sh` takes about an hour on a fresh instance, mostly the stereo image and engine and
the restore; rerunning it on a ready instance takes a few minutes and changes nothing. Its only
manual step: the Hugging Face token comes from `path_b/.env`. With `HF_WRITE_TOKEN=` there, the
instance can upload; with only the read-only `HF_TOKEN`, run `ssh -t l40s /root/.local/bin/hf auth
login` with a write token before an upload. On the instance, `path_b/.env` is sourced only for
`v2d_submit.py` (its `HF_TOKEN` would override the login).

The sections below are what the two scripts do, step by step.

## Before deleting an instance

1. Push code that is only on the Mac to GitHub (or it lives only on the Mac).
2. Upload the outputs you want to keep (needs `hf auth login` with a **Write** token):
   ```bash
   cd /root/v2d/NVIDIA_V2D_Egocentric
   uv run --project tools python path_b/vm/hf_outputs.py upload --run <run> [--frames]
   uv run --project tools python path_b/vm/hf_outputs.py list
   ```
   `<run>` is a folder of `path_b/data/outputs/`. Results (stage reports, loss reports, scales,
   bundles, stages 2–3 outputs, waterfall, pages) are small; `--frames` adds the per-frame
   rectified images and depth as one `frames.tar` per episode (~0.9 GB each), which saves rerunning
   P0–P2 (~6 GPU-minutes per episode). Restoring all 20 public episodes with frames (17.6 GB) took
   30 min on the L40S instance (2026-10-19, run `4c2eac31`), and `run.py` then kept every stage. They go to `path_b/runs/<run>/` in
   `rose-tea-cafe/v2d-t3-recon`; `tools/track3/bundles.py` reads only its top-level `episode_*`.

## On a new instance (Ubuntu with NVIDIA driver, Docker and the NVIDIA Container Toolkit)

```bash
scp path_b/vm/setup.sh <host>:
ssh -t <host> bash setup.sh
```

It stops where it needs you, prints what to do, and continues from there when rerun:

| Step it waits for | What to do |
|---|---|
| GitHub deploy key (once per instance) | Repo → Settings → Deploy keys → Add the printed key, read-only |
| Hugging Face login | `ssh -t <host> /root/.local/bin/hf auth login` (answer **n** to the git-credential question) |
| MANO files | `scp ~/V2D/mano_v1_2/models/MANO_{RIGHT,LEFT}.pkl <host>:/root/v2d/NVIDIA_V2D_Egocentric/baseline/reconstruction/data/weights/hand/models/` |

Then it installs git-lfs, ffmpeg, uv and Python 3.11; clones the repo (`BRANCH`, default
`main_buffer`) and its LFS files; downloads the Track 3 dataset and the team's bundles; builds the
four environments (tools, path_b, baseline host, stages 2–3 as in README §5.12); builds the stereo
image, its ONNX model and a TensorRT engine for this GPU; restores outputs if asked; and runs the
smoke tests. About an hour on a fresh instance, mostly the 15 GB TensorRT base image and the engine
build; seconds on a set-up one. Logs: `/root/v2d/logs/setup_<step>.log`.

| Setting | Effect |
|---|---|
| `BRANCH=zjifan121` | Branch for the first clone |
| `RESTORE_RUN=<run>` | Restore that run's outputs from Hugging Face into `path_b/data/outputs/<run>/` |
| `RESTORE_FRAMES=1` | With `RESTORE_RUN`: also its per-frame intermediates |
| `SELF_TEST=1` | Also run the local scorer's self-test (~17 min, CPU) |

Example: `RESTORE_RUN=4c2eac31 RESTORE_FRAMES=1 bash setup.sh`.

Code not yet on GitHub reaches the instance from the Mac with
`git push --no-verify <host>:/root/v2d/repo.git <branch>`, then
`git -C /root/v2d/NVIDIA_V2D_Egocentric pull --ff-only /root/v2d/repo.git <branch>`.

