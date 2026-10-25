# Track 3 jobs on Babel

Slurm scripts for the Track 3 pipeline on CMU Babel. They assume the repo at
`/home/$USER/NVIDIA_V2D_Egocentric` and per-user data under `/data/user_data/$USER/v2d/`:

| Path | Holds |
|---|---|
| `sif/` | Apptainer images (`pull_sifs.sbatch`) |
| `recon_data/{outputs,weights}` | `baseline/reconstruction/data/{outputs,weights}` symlink here |
| `stereo_test/{seed,seed_fs,logs}` | stereo-depth seeds and job logs |
| `flash_runs/diag/<name>` | training runs |
| `venvs/{retarget,flash_chord}` | uv environments for load/retarget and FlashCHORD |
| `jobs/logs/` | training and load/retarget logs |

One-time setup for the stereo seed (`seed.sbatch`, `fs.sbatch`, `run.sbatch`), on a compute node:

    uv venv --python 3.11 /data/user_data/$USER/v2d/venvs/stereo
    uv pip install --python /data/user_data/$USER/v2d/venvs/stereo numpy scipy opencv-python-headless onnxruntime-gpu==1.20.0 \
      nvidia-cudnn-cu12==9.* nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
    python baseline/reconstruction/modules/v2d_foundation_stereo/lib/download_weights.py \
      --output_dir /data/user_data/$USER/v2d/recon_data/weights/foundation_stereo

The `nvidia-*` wheels are the CUDA 12 / cuDNN 9 libraries onnxruntime-gpu loads; `run.sbatch` puts
their `lib/` folders on `LD_LIBRARY_PATH` when it runs FoundationStereo on the GPU.

The calibration and stereo code is `tools/track3/stereo_seed.py`; the late-hand fix is
`--late-hands` in `tools/track3/run_baseline_reconstruction.py` (`--late_hand_prompts` in
`run_ego_wilor.py`).

Traps that cost most of the score (public ep41 drops from AUC 0.84 to under 0.4):

- **Seed folder.** `run.sbatch ... fs_sam3d` reconstructs from `stereo_test/seed_fs/ep<N>`, whose
  `depth/` is FoundationStereo (`depth_scale.json` says so, `median_scale` 1.0). `stereo_test/seed/ep<N>`
  is the stage-1 folder: its `depth/` is MoGe rescaled to stereo (`median_scale` about 0.7), which scored
  0.15 on ep41.
- **No 180° turn.** The calibration lists the head cameras as `rotation_deg: 180`, but the released
  videos are already upright and the calibration fits them as stored. On ep41 the seed's
  `anycalib/undistorted_intrinsics.json` has cx 994.80, cy 767.54; cx 1032.20, cy 751.46 means the
  intrinsics were turned.
- **Score in the scan's frame.** Our poses are in our mesh's frame; compare them with ground truth only
  through `score_playback.sh` (it uses the metric's own registration). Read `recon_ourmesh`.

GPU jobs go to `preempt` (requeued on preemption; every script resumes). `train_diag.sbatch` requeues itself on
TERM; it checks for a cancel first, but if a cancelled run still comes back, stop it with
`scancel --signal=KILL --full <job>`. CPU work goes to `maxlab-cpu`.
`preempt_qos` allows 100 queued and 24 running jobs per user, so keep one job per run.

## Order

| Step | Script | What it does |
|---|---|---|
| Setup | `pull_sifs.sbatch`, `weights.sbatch` | Images as `.sif`; weights that need their module image |
| R0–R2 seed | `seed.sbatch <ep>` (CPU), `fs.sbatch <ep> <start> <stop>` (CPU) | Calibrated stereo video; FoundationStereo depth in frame chunks |
| R3–R13 | `run.sbatch <ep> <object> fs_sam3d` | Reconstruction on the FoundationStereo depth, one object per job (~1 h) |
| R4 fix | `hands.sbatch <ep> <object>` | Rerun with `--late-hands` when a hand enters after frame 0 (namespace `fs_hands`) |
| R11 fix | `../write_unrefined_bundle.py <src_ns> <dst_ns> <ep>:<object> ...` (CPU, minutes) | Bundle without the joint refinement; use when the refined camera path is >6x the SLAM path |
| L–T | `scene_train.sh` (`LR_CPU=1` builds the scene on CPU) | Load + retarget, then a training run with the default recipe: 100M steps, object keypoint reward x2 (`train_diag.sbatch`, self-requeuing, L40S allowed) |
| R4 follow-up | `hands_next.sh <ep> ...` | After `hands.sbatch` finishes both objects: camera rule, bundle, scene and training |
| Score | `score_playback.sh <out.json> <ns,...> <ep,...>` | Official-metric playback scores for public episodes (no simulation) |
| Score a policy | `../score_rollout.py <run_dir> <ep> [scene_variant]` | Official scores of a training run's rollout, full or half speed |
| Meshes (CD-O) | `../fit_shape.py`, `../swap_mesh.py`, `../eval_mesh.py` | Fit mesh proportions to masks and depth over 40 frames; put the fitted meshes in bundles; CD-O against the public scans |
| Check | `scan.sbatch <ep> <object>` | Public episodes only: track with the released scan, to measure what a perfect mesh is worth |
| Safety net | `still_submit.sh` | Hold-still scenes and policies for the scored episodes |

Example for one two-object episode, no refinement, two seeds (the second reuses the scene):

    NS=fs_noref TAG=obj2x_s42 LR_CPU=1 SCORE=1 tools/track3/jobs/scene_train.sh 12
    NS=fs_noref TAG=obj2x_s43 SEED=43 SKIP_LR=1 SCORE=1 tools/track3/jobs/scene_train.sh 12

Keep each episode's best seed by own-reference score with `../select_policies.py`.

`l40s_probe.sbatch` is the memory check behind the L40S contact-buffer setting in `train_diag.sbatch`.
