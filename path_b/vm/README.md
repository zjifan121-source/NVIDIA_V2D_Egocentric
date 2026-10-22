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

## SPIDER for S4 (`s4_scene`, once per instance)

SPIDER (facebookresearch/spider, CC BY-NC 4.0) is installed beside the repo, never copied into it,
in its own Python 3.12 environment: its pins (mujoco 3.7.0, mujoco-warp 3.7.0.1, warp-lang 1.12.1)
differ from FlashCHORD's (mujoco 3.8.1, warp-lang 1.14.0). The commit below is the one Path B's
adapter was written against.

```bash
git clone https://github.com/facebookresearch/spider /root/v2d/spider
git -C /root/v2d/spider checkout 44717007de41cbef7565dff7ff9f4453557a2d3d
cd /root/v2d/spider && uv sync --python 3.12 && uv pip install coacd trimesh   # decompose.py needs both
export SPIDER_DIR=/root/v2d/spider SPIDER_PY=/root/v2d/spider/.venv/bin/python
export SPIDER_ROBOT=xhand            # ships with SPIDER; Sharpa needs SPIDER_ROBOT_ASSETS=<dir with bimanual.xml>
```

Without `SPIDER_DIR` and `SPIDER_PY`, `s4_scene` is skipped with that reason.
