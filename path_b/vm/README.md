# path_b/vm/ — set up a new GPU instance, save and restore outputs

A GPU instance can be deleted whenever it is idle (on Verda, a shut-down instance keeps billing; a
deleted one does not). Code lives in git, results on Hugging Face, and `setup.sh` rebuilds the rest.

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
   P0–P2 (~6 GPU-minutes per episode). They go to `path_b/runs/<run>/` in
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
