# NVIDIA V2D Egocentric — Track 3 Team Repository

Private team repository for the **NVIDIA Isaac Video-to-Data (V2D) Challenge, Track 3: Egocentric Video → Policy**.

It is built on NVIDIA's [`video_to_data`](https://github.com/nvidia-isaac/video_to_data) at `release/0.2.0`. The team works on two independent approaches, kept on separate branches:

- **Path A: build on the baseline.** Add to and fix NVIDIA's pipeline inside `baseline/`, on the `path-a` branch.
- **Path B: a completely new approach.** Lives in `path_b/` on the `path-b` branch. It shares only the inputs and the evaluator with the rest of the repo.

Both paths are scored by the same shared evaluator, so their results are directly comparable.

> **`baseline/` is NVIDIA's repository exactly as released**, including NVIDIA's own [README](baseline/README.md), docs and agent skills. Their paths are relative to `baseline/`, so run NVIDIA's commands from there, and start an agent inside `baseline/` to use NVIDIA's skills.

---

## Contents

1. [Quick facts](#1-quick-facts)
2. [Repository structure](#2-repository-structure)
3. [Branches and workflow](#3-branches-and-workflow)
4. [First-time setup](#4-first-time-setup)
5. [Running the baseline (reconstruction)](#5-running-the-baseline-reconstruction)
6. [Track 3 data essentials](#6-track-3-data-essentials)
7. [Modifying the pipeline](#7-modifying-the-pipeline)
8. [What never goes into git](#8-what-never-goes-into-git)
9. [Licensing](#9-licensing)
10. [Troubleshooting](#10-troubleshooting)
11. [Links](#11-links)

---

## 1. Quick facts

| | |
|---|---|
| Baseline | `nvidia-isaac/video_to_data` `release/0.2.0` |
| Baseline tag in this repo | `baseline-0.2.0` (NVIDIA's exact commit, unmodified) |
| Newer NVIDIA releases | v0.3.0 (end-to-end GR00T workflow), v0.4.0 (FlashCHORD). Brought in only when needed ([§3](#3-branches-and-workflow)). |
| Remotes | `origin` = this repo, `upstream` = NVIDIA |
| Dataset | [`nvidia/video_to_data_challenge`](https://huggingface.co/datasets/nvidia/video_to_data_challenge) → `track_3/` (CC-BY-4.0, ~1.5 GB) |
| Validated GPUs | RTX A6000 (SM 86), L40S (SM 89) |
| Unsupported GPUs | Blackwell `sm_120` (e.g. RTX PRO 6000 Blackwell). Others (H100, V100) are not validated. |
| Runtime | Linux + Docker + NVIDIA Container Toolkit. **Macs and Colab cannot run the pipeline.** |
| Team compute | CMU / PSC Bridges-2 (see [§5.10](#510-running-on-psc-bridges-2) for status) |

---

## 2. Repository structure

Legend: **[N]** NVIDIA, unmodified · **[S]** shared, on `main` · **[A]** `path-a` branch only · **[B]** `path-b` branch only · **[L]** local only, never committed

```
NVIDIA_V2D_Egocentric/
├── README.md  NOTICE  LICENSE  .gitignore   [S] team files (LICENSE is a copy of NVIDIA's)
│
├── baseline/                       [N] NVIDIA video_to_data release/0.2.0, exactly as released
│   ├── README.md  LICENSE  SECURITY.md  .gitignore  .gitattributes
│   ├── .github/ .claude/ .codex/   NVIDIA's CI and agent skills (CI does not run from here)
│   ├── docs/                       NVIDIA's docs site
│   ├── reconstruction/             video → depth, masks, mesh, 6-DoF object pose, MANO hands
│   │   ├── scripts/                install / build / download-weights scripts
│   │   ├── docs/ego_e2e_setup.md   NVIDIA's ego pipeline guide (source for §5)
│   │   ├── assets/airplane.mp4     NVIDIA's sample video (Git LFS)
│   │   ├── modules/
│   │   │   ├── v2d_pipelines/      entrypoints: run_ego_reconstruction.py, run_ego_wilor.py, ...
│   │   │   ├── v2d_common/         typed data contracts, result bundle writer
│   │   │   ├── v2d_docker/         host-side launcher that runs each module in Docker
│   │   │   ├── v2d_moge/ v2d_droid_slam/ v2d_wilor/ v2d_grounding_dino/ v2d_sam2/
│   │   │   ├── v2d_sam3d/ v2d_foundation_pose/ v2d_hamer/ v2d_gsplat_refinement/ v2d_geocalib/ ...
│   │   │   └── v2d_t3_<name>/      [A] new Path A modules (prefix t3_ = ours)
│   │   └── data/                   [L] runtime cache (paths are relative to baseline/reconstruction/)
│   │       ├── weights/            model weights, incl. hand/models/MANO_RIGHT.pkl
│   │       └── outputs/<branch>/<episode>/   one folder per run
│   ├── robotic_grounding/          retargeting + Isaac Lab PPO (not covered by this README yet)
│   └── video_ingestion_agent/      not used for Track 3
│
├── assets/                         [S] shared inputs (see assets/README.md)
│   └── data/                       [L] Track 3 dataset
│
├── shared/                         [S] shared code: evaluator, result format (not created yet)
└── path_b/                         [B] Path B's pipeline
```

Rules that keep this layout working:

- **Everything about the baseline sits in `baseline/`.** On `main` it is exactly NVIDIA's `release/0.2.0`; this prints nothing:
  `git diff --stat baseline-0.2.0 main:baseline`
- **Path A** adds to and edits `baseline/`, and only on the `path-a` branch. There is no separate Path A folder.
- **Path B** lives entirely in `path_b/`, only on the `path-b` branch. It uses only the shared inputs (`assets/`) and the shared evaluator (`shared/`), and never touches `baseline/`.
- **Shared things** (inputs, evaluator, this README) change only on `main`.

---

## 3. Branches and workflow

Three branches:

```
upstream (NVIDIA)   release/0.2.0 ──► tag baseline-0.2.0
                                         │
origin (this repo)                      main      shared: baseline/ (unmodified), assets/, shared/, README
                         ┌───────────────┴───────────────┐
                       path-a                         path-b
               main + changes inside baseline/   main + path_b/
```

| Branch | Contains | Merges into |
|---|---|---|
| `main` | Unmodified baseline + shared inputs, evaluator and docs | — |
| `path-a` | `main` + Path A's additions/edits inside `baseline/` | nothing |
| `path-b` | `main` + `path_b/` | nothing |

**The rules:**

1. Changes to `main` go through a pull request from a short-lived branch, deleted after merging.
2. After anything merges into `main`, merge `main` into `path-a` and `path-b`.
3. **Never merge `path-a` into `path-b` or the reverse, and never merge a path branch into `main`.**
4. Use **Create a merge commit** on GitHub, not squash or rebase.
5. Tag every leaderboard submission: `submit/a-YYYY-MM-DD-N` or `submit/b-YYYY-MM-DD-N`. There are only 5 submissions per week, so each score must be traceable to exact code.

GitHub does not enforce branch protection on this private repo's plan, so these rules hold only by convention.

**Typical cycle on a path branch:**

```bash
git switch path-a && git pull
# ...edit inside baseline/, run, score...
git add -p
git commit -m "Seed SAM3D in HaMeR mode"
git push
```

**Bring shared changes from `main` into your path branch:**

```bash
git switch path-a && git pull
git merge origin/main
git push
```

**Work on both paths at once** with worktrees, so each folder is a separate checkout:

```bash
git worktree add ../v2d-path-a path-a
git worktree add ../v2d-path-b path-b
```

**See everything Path A changed vs. NVIDIA:**

```bash
git diff --stat baseline-0.2.0 origin/path-a:baseline
```

**Bring in a newer NVIDIA release** (e.g. for FlashCHORD). Because `baseline/` on `main` is unmodified, it is replaced wholesale on a short-lived branch, then merged into the path branches as usual:

```bash
git fetch upstream --tags
git switch -c sync-upstream main
git rm -r -q baseline
git read-tree --prefix=baseline/ -u upstream/<release-branch>
git commit -m "Update baseline/ to NVIDIA <release>"
# re-run the smoke test (§5.4), tag the new baseline, open a PR into main
```

---

## 4. First-time setup

### 4.1 Repository access

1. Accept the GitHub collaborator invitation (check your email).
2. Install the GitHub CLI and Git LFS. On macOS: `brew install gh git-lfs`. On Linux, use your package manager.
3. Authenticate. GitHub no longer accepts passwords for git, so use:
   ```bash
   gh auth login        # GitHub.com → HTTPS → Yes → Login with a web browser
   ```
4. Clone and wire up NVIDIA as `upstream`:
   ```bash
   git clone https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric.git
   cd NVIDIA_V2D_Egocentric
   git remote add upstream https://github.com/nvidia-isaac/video_to_data.git
   git lfs install
   git lfs pull upstream          # LFS files (sample video, assets) come from NVIDIA's public repo
   ```
   Check with `ls -lh baseline/reconstruction/assets/airplane.mp4`. It should be several MB, not a ~130-byte pointer file.

### 4.2 Personal accounts and licenses (each person, once)

| What | Why | How |
|---|---|---|
| Hugging Face account + token | SAM3D weights are gated; dataset download | Create a token at huggingface.co → Settings → Access Tokens. Request access on the SAM 3D model page. Then `huggingface-cli login` or `export HF_TOKEN=...` |
| MANO license | Hand model used by every hand step; it cannot be redistributed | Register at https://mano.is.tue.mpg.de/ and download `MANO_RIGHT.pkl` yourself |

Never share `MANO_RIGHT.pkl` or tokens through this repo or chat.

### 4.3 Where code runs

| Machine | Use for |
|---|---|
| Your laptop (incl. Mac) | Editing, git, reviewing outputs (overlay videos, Three.js viewer) |
| Linux GPU machine with Docker (A6000 / L40S) | Running the pipeline, works as documented |
| PSC Bridges-2 L40S nodes | Target team compute; **needs a launcher port first** ([§5.10](#510-running-on-psc-bridges-2)) |

---

## 5. Running the baseline (reconstruction)

This section covers **video → reconstruction result bundle**. Robotic grounding (retargeting + PPO policy) is the next stage and will be documented separately. NVIDIA's guide is `baseline/reconstruction/docs/ego_e2e_setup.md`.

**All commands below run from `baseline/reconstruction/`.** The Track 3 dataset lives at `../../assets/data/track_3/` from there.

### 5.1 Prerequisites on the GPU machine

- Linux, an RTX A6000 or L40S (≥ 24 GB VRAM), and a working NVIDIA driver
- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Check with:
  `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`
- Your user can run Docker without `sudo` (member of the `docker` group)
- Python 3.10+, `ffmpeg` on `PATH`, Git LFS
- Plenty of disk space: Docker images and weights are large

### 5.2 Install, build, download weights (once per machine)

```bash
cd baseline/reconstruction
./scripts/install_ego_reconstruction_packages.sh      # host-side Python wrappers (no ML deps)
./scripts/build_ego_reconstruction_packages.sh        # builds the per-module Docker images (slow)
export HF_TOKEN=<your token>                          # or: huggingface-cli login
./scripts/download_ego_reconstruction_weights.sh --mode hamer_prompt
```

Weight modes:

| Mode | Use when |
|---|---|
| `hamer_prompt` | Recommended. HaMeR hands + SAM3D-generated object mesh |
| `hamer_mesh` | HaMeR hands + a provided object mesh (e.g. Track 3 scans) |
| `dynhamr_prompt` | Legacy ViPE + Dyn-HaMR hands |
| *(no flag)* | Everything |

### 5.3 Place the MANO model

```
baseline/reconstruction/data/weights/hand/
├── models/
│   └── MANO_RIGHT.pkl        ← your own download
└── BMC/
    └── *.npy                 ← only needed for the DynHaMR mode
```

Put it only here. The HaMeR pipeline copies it where its container needs it, so don't place it anywhere else.

### 5.4 Smoke test on NVIDIA's sample video

Run this first, unmodified, to prove the machine works:

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
    --video assets/airplane.mp4 \
    --object_prompt "A toy airplane" \
    --output_dir data/outputs/smoke/airplane_hamer \
    --reference_frame 0 \
    --undistort \
    --hand_tracking hamer \
    --run_droid_slam \
    --run_gravity_alignment \
    --run_gsplat_refinement \
    --export_threejs_result \
    --dev
```

Success means a `result*/result.npz` file exists and the overlay videos look sensible.

### 5.5 Download the Track 3 data

```bash
huggingface-cli download nvidia/video_to_data_challenge --repo-type dataset \
    --include "track_3/*" --local-dir ../../assets/data/
ls ../../assets/data/track_3/public   # data/ videos/ mesh/ urdf/ meta/
```

`assets/data/track_3/evaluation/` holds videos only; its ground truth is hidden. Only `public/` (20 episodes) can be scored locally. Full layout and file naming: [`assets/README.md`](assets/README.md).

Clip path pattern: `../../assets/data/track_3/public/videos/chunk-000/observation.images.ego_cam_a/episode_XXXXXX.mp4`

### 5.6 Prepare a Track 3 clip

1. **Use the head colour camera `ego_cam_a`** (2028×1520). `ego_cam_b`/`ego_cam_c` are the mono stereo pair.
2. **Check that frames are upright.** The calibration file lists a 180° rotation for this camera, and the hand and object detectors fail on upside-down images.
   ```bash
   ffmpeg -y -i <clip>.mp4 -frames:v 1 /tmp/frame0.png     # open and look
   # if upside down, rotate 180° into a new file:
   ffmpeg -i <clip>.mp4 -vf "hflip,vflip" -c:v libx264 -crf 17 <clip>_upright.mp4
   ```
3. **Choose the object prompt.** The episode's objects are in its parquet schema metadata (`objects`), e.g. `white_pot` → prompt `"white pot"`. The baseline tracks **one** object: the top Grounding DINO box. For two-object episodes, prompt the manipulated one.
4. **Choose `--reference_frame`.** Pick a frame where the object and both hands are fully visible. Frame 0 is not always a good choice, and if no hand or no object is detected on it, the run stops with an error.

### 5.7 Run on a Track 3 episode

Standard baseline (HaMeR hands, SAM3D mesh):

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
    --video <clip>.mp4 \
    --object_prompt "white pot" \
    --output_dir data/outputs/main/<episode> \
    --reference_frame <N> \
    --undistort \
    --hand_tracking hamer \
    --run_droid_slam \
    --run_gravity_alignment \
    --run_gsplat_refinement \
    --export_threejs_result \
    --dev
```

Variant using the **provided scanned mesh** instead of SAM3D (download weights with `--mode hamer_mesh`):

```bash
python modules/v2d_pipelines/run_ego_reconstruction.py \
    --video <clip>.mp4 \
    --object_prompt "white pot" \
    --object_mesh ../../assets/data/track_3/public/mesh/white_pot/white_pot.obj \
    --skip_object_scale_estimation \
    --output_dir data/outputs/main/<episode>_scanmesh \
    --reference_frame <N> --undistort --hand_tracking hamer \
    --run_droid_slam --run_gravity_alignment --run_gsplat_refinement \
    --export_threejs_result --dev
```

Scans ship as `.glb`. NVIDIA's example uses `.obj`, and `.glb` input is untested. To convert:
`python -c "import trimesh; trimesh.load('white_pot.glb', force='mesh').export('white_pot.obj')"`.
`mini_sweeper`, `wooden_piece_1` and `wooden_piece_2` ship `<name>_visual.glb` + `<name>_collision.glb` instead of `<name>.glb`; convert the `_visual.glb`.
The prompt is still required, because it drives detection, masks and pose initialisation.

**Output folder convention:** `data/outputs/<branch>/<episode>[_variant]`. Never reuse another branch's output folder (see §5.9).

### 5.8 Outputs and how to inspect them

```
<output_dir>/
├── result/                         base bundle
├── result_slam/                    with DROID-SLAM camera
├── result_slam_gravity_aligned/    final bundle when SLAM + gravity alignment ran
│   ├── result.npz                  per-frame camera, object and both-hand poses
│   ├── mesh.obj (+ .mtl, texture)  object mesh used for tracking
│   ├── manifest.json               which sources/settings produced it
│   └── threejs_scene/index.html    3D viewer (open in a browser)
├── masks_overlay.mp4, hamer_aligned_overlay.mp4, refined_simple_overlay.mp4, ...
└── per-step intermediate folders (frames/, depth/, masks/, poses/, hamer*/ ...)
```

The run prints the final bundle as `final_result_bundle:`. When debugging, **watch the overlay videos first**: most failures (wrong object, lost mask, flipped object) are visible there.

`result.npz` key facts:
- Poses are 4×4 `A_to_B` matrices.
- Camera frame is OpenCV (+X right, +Y down, +Z forward, metres).
- After gravity alignment, world is +Z up.
- Hand "wrist" transforms are MANO parameters (`[R(global_orient) | cam_t]`), not the wrist joint pose. Run MANO forward to get joints.

### 5.9 Caching and re-running

- Every step is skipped if its main output already exists (it prints `[skip]`). Re-running the same command resumes.
- **To redo a step, delete that step's output folder** (and anything downstream that depends on it), or use a fresh `--output_dir`.
- Changing code does **not** invalidate the cache. After editing a module, delete the affected outputs or you'll be looking at old results.
- `--dev` mounts your local module source into the containers, so Python edits take effect without rebuilding images. Rebuild (§5.2) only after changing a Dockerfile or dependencies.

### 5.10 Running on PSC Bridges-2

**Status: not working yet. Treat this as an open shared task.** The launcher lives in `baseline/`, so the port is a Path A-style change unless the team agrees to carry it on `main`.

- Bridges-2 has **L40S nodes** (8× L40S 48 GB per node), a validated GPU.
- Bridges-2 provides **Apptainer/Singularity, not Docker**. The baseline's host launcher (`baseline/reconstruction/modules/v2d_docker/`) calls `docker run` for every step, so the pipeline cannot run there unchanged.

Plan for the port:
1. Build the module Docker images on any Linux x86-64 machine with Docker (§5.2) and push them to a private registry, e.g. GitHub Container Registry.
2. On Bridges-2, convert each one: `apptainer pull <name>.sif docker://ghcr.io/<owner>/<image>:<tag>`. Store `.sif` files, weights and data on Ocean (`$PROJECT`), not in your home directory.
3. Add an Apptainer backend to the launcher: `apptainer exec --nv --bind <output_dir>,<weights>,<repo> <name>.sif ...` in place of `docker run --gpus all -v ...`. Do this behind a flag, so Docker machines keep working.
4. Run inside a Slurm job on an L40S node (interactive for debugging, batch for many episodes). Check PSC's Bridges-2 user guide for the current partition and GPU-type names.

Until this lands, run the baseline on a Linux GPU machine with Docker.

---

## 6. Track 3 data essentials

Read `assets/data/track_3/README.md` once. The parts that bite:

| Fact | Consequence |
|---|---|
| 20 public episodes, 10 tasks, 10 objects, ~262 frames each at 20.25 fps | Small. Use all of it for local validation; don't overfit to one clip. |
| Episode indices are not contiguous (public: 0, 2, 11, …, 45) | Iterate over files or `meta/info.json` → `episode_indices`, not `range(20)` |
| Video frame *i* = parquet row *i* in every clip | No time sync needed |
| `pose` = `[x, y, z, qw, qx, qy, qz]`, **w-first** quaternion, metres | SciPy wants x, y, z, w. Reorder before `Rotation.from_quat`. |
| Poses are `world_T_object` **in the scanned mesh's frame** | A SAM3D mesh's frame is arbitrary (reference-camera axes). Compare against scans, or solve the rotation. |
| x, y centred per episode; z absolute (table ≈ 0.61 m) | Episodes don't share an origin |
| `visible == False` → pose is all zeros | **Filter on `visible`** before using any pose |
| No camera-to-mocap extrinsic is provided | Reconstruction and ground truth live in different worlds. Solve an alignment before computing any error. |
| 1–2 objects per episode (sometimes more) | Baseline tracks only one |
| Real factory calibration in `meta/camera_calibration.json` (14-coeff distortion) | Better than the baseline's learned AnyCalib/MoGe estimates |
| Stereo pair, 75 mm baseline | Possible metric-depth source; the baseline ignores it |
| License: CC-BY-4.0 | Free to use with attribution. Still never commit it (size). |

---

## 7. Modifying the pipeline

### How the pipeline is built

- The host runs thin Python wrappers. Each step launches its own Docker image.
- Steps communicate **only through files** in `--output_dir`, with typed formats from `v2d_common`.
- Entrypoint: `baseline/reconstruction/modules/v2d_pipelines/run_ego_reconstruction.py`, which in HaMeR mode calls `run_ego_wilor.py`.

### Path A: building on the baseline

- All Path A work, edits and new code alike, goes inside `baseline/` on the `path-a` branch.
- Small, focused commits, one concern each.
- Every edited NVIDIA file gets a header comment, as Apache-2.0 requires:
  `# Modified for V2D Track 3 by <name>, <YYYY-MM>: <what changed>`
- New standalone steps go in a new module, `baseline/reconstruction/modules/v2d_t3_<name>/` (`--dev` only mounts that folder into containers). Follow NVIDIA's module contract: its own Docker image, a `run_download_weights` entry point if it needs weights, a `run_shell` entry point, and types from `v2d_common`.
- Compare against the baseline on the same episodes, using the same evaluator.

Known weak points of the baseline (good first targets):

| Symptom | Cause in code |
|---|---|
| Different mesh every run | SAM3D called with `seed=None` in HaMeR mode (`run_ego_wilor.py`) |
| Wrong object scale | Scale fitted to MoGe depth on a single frame (SAM3D + FoundationPose scale search) |
| Object flips ~180° | FoundationPose re-registers from scratch when mask IoU < 0.3, with no temporal or symmetry prior |
| Hands not touching the object | Hands and object fitted to depth independently; gsplat penetration loss weight is 0 by default |
| Jittery hands | HaMeR is per-frame; only gsplat refinement smooths |
| Only one object | Only the top Grounding DINO box is tracked |
| Late-appearing hand/object lost | All SAM2 tracks start on `--reference_frame` |
| Approximate intrinsics | MoGe forces the principal point to the image centre; the real calibration is ignored |

### Path B: a completely new approach

- All code lives in `path_b/` on the `path-b` branch, with its own entrypoint, stages and Docker images.
- It uses only the shared inputs (`assets/`) and the shared evaluator (`shared/`). It does not use or edit `baseline/`.
- Once the shared result format and evaluator exist, it must write that format so both paths are scored the same way.

### Before opening a PR

- [ ] Ran on at least the agreed validation episodes, from a clean output folder
- [ ] Scored with the shared evaluator (once available) and pasted the numbers into the PR, alongside the baseline numbers
- [ ] No data, weights, outputs or tokens in the diff (`git status`, `git diff --stat`)
- [ ] Modification notices added to edited NVIDIA files

---

## 8. What never goes into git

| Never commit | Instead |
|---|---|
| `baseline/reconstruction/data/` (weights, outputs), `assets/data/` (dataset) | Each person downloads; outputs stay local or on shared storage |
| `MANO_RIGHT.pkl` | Each person registers and downloads |
| `HF_TOKEN`, any `.env`, credentials | Environment variables |
| Model checkpoints (`*.pt`, `*.pth`, `*.ckpt`), `*.npz` | Shared storage (e.g. PSC Ocean) |
| Large videos / meshes | Git LFS only if essential, after team agreement |

These patterns are already in `.gitignore`. Check before committing with `git status`. If something sensitive was committed, tell the team immediately: deleting it in a new commit does not remove it from history.

---

## 9. Licensing

- **NVIDIA code:** Apache-2.0. **NVIDIA docs and skills:** CC-BY-4.0. See `baseline/LICENSE`, which must stay unchanged. The root `LICENSE` is a copy.
- **Our code** (`shared/`, `path_b/`, Path A's `v2d_t3_*` modules and marked edits): Apache-2.0, see `NOTICE`.
- **Track 3 dataset:** CC-BY-4.0 (attribution required).
- **Third-party models** (MANO, SAM3D, HaMeR, WiLoR, Grounding DINO, ...) have their own licenses. Some are gated or non-commercial.
- Winning submissions must release their code under Apache-2.0 (challenge FAQ). Before this repo or any branch goes public, scan the full history for secrets.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `Password authentication is not supported` on clone or push | Use `gh auth login` (§4.1), not your GitHub password |
| `airplane.mp4` is ~130 bytes / video won't open | It's an LFS pointer: `git lfs install && git lfs pull upstream` |
| `docker: permission denied` | Add yourself to the `docker` group, then log out and back in |
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit not installed or configured |
| Launcher refuses to start on the GPU | Blackwell `sm_120` is unsupported. Use an A6000 / L40S. |
| `RuntimeError` about no hands or no object detected | Pick a different `--reference_frame`, or rephrase `--object_prompt` |
| Edits have no effect | Cached outputs: delete the step's output folder. Also check you passed `--dev`. |
| Results differ between identical runs | Known: unseeded SAM3D (HaMeR mode), DROID-SLAM and GPU non-determinism. Compare over several runs. |
| SAM3D download or access error | Request access on its Hugging Face page and set `HF_TOKEN` |
| Imports or scripts still point at the old `reconstruction/` path (installed before the `baseline/` move) | Editable installs record absolute paths: re-run `./scripts/install_ego_reconstruction_packages.sh` from `baseline/reconstruction/` |
| A command from NVIDIA's docs or skills can't find a path | NVIDIA's paths are relative to `baseline/`: run it from there |
| Stuck in the vim editor during `git commit` / merge | `Esc`, then `:wq` (save) or `:cq` (abort). To avoid it: `git config --global core.editor nano` |

---

## 11. Links

- NVIDIA repo: https://github.com/nvidia-isaac/video_to_data (baseline: `release/0.2.0`; NVIDIA's README: `baseline/README.md`)
- NVIDIA docs: https://nvidia-isaac.github.io/video_to_data/
- Ego pipeline guide: `baseline/reconstruction/docs/ego_e2e_setup.md`
- Robotic grounding guide: `baseline/robotic_grounding/docs/SETUP.md`
- Track 3 dataset: https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main/track_3
- CHORD (grounding method) paper: https://arxiv.org/abs/2607.00033
- Challenge contact: v2d_challenge@nvidia.com
- PSC Bridges-2 user guide: https://www.psc.edu/resources/bridges-2/user-guide
