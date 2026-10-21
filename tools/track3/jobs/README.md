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
    uv pip install --python /data/user_data/$USER/v2d/venvs/stereo numpy scipy opencv-python-headless onnxruntime-gpu==1.20.0
    python baseline/reconstruction/modules/v2d_foundation_stereo/lib/download_weights.py \
      --output_dir /data/user_data/$USER/v2d/recon_data/weights/foundation_stereo

The calibration and stereo code is `tools/track3/stereo_seed.py`; the late-hand fix is
`--late-hands` in `tools/track3/run_baseline_reconstruction.py` (`--late_hand_prompts` in
`run_ego_wilor.py`).

GPU jobs go to `preempt` (requeued on preemption; every script resumes). CPU work goes to `maxlab-cpu`.
`preempt_qos` allows 100 queued and 24 running jobs per user, so keep one job per run.

## Order

| Step | Script | What it does |
|---|---|---|
| Setup | `pull_sifs.sbatch`, `weights.sbatch` | Images as `.sif`; weights that need their module image |
| R0–R2 seed | `seed.sbatch <ep>` (CPU), `fs.sbatch <ep> <start> <stop>` (CPU) | Calibrated stereo video; FoundationStereo depth in frame chunks |
| R3–R13 | `run.sbatch <ep> <object> fs_sam3d` | Reconstruction on the FoundationStereo depth, one object per job (~1 h) |
| R4 fix | `hands.sbatch <ep> <object>` | Rerun with `--late-hands` when a hand enters after frame 0 (namespace `fs_hands`) |
| R11 fix | `../write_unrefined_bundle.py <src_ns> <dst_ns> <ep>:<object> ...` (CPU, minutes) | Bundle without the joint refinement; use when the refined camera path is >6x the SLAM path |
| L–T | `scene_train.sh` (`LR_CPU=1` builds the scene on CPU) | Load + retarget, then a 250M-step training run (`train_diag.sbatch`, self-requeuing, L40S allowed) |
| Score | `score_playback.sh <out.json> <ns,...> <ep,...>` | Official-metric playback scores for public episodes (no simulation) |
| Check | `scan.sbatch <ep> <object>` | Public episodes only: track with the released scan, to measure what a perfect mesh is worth |
| Safety net | `still_submit.sh` | Hold-still scenes and policies for the scored episodes |

Example for one two-object episode, no refinement:

    NS=fs_noref TAG=v2_250M LR_CPU=1 LR_EXTRA="--smooth-hands 2" SCORE=1 tools/track3/jobs/scene_train.sh 12

`l40s_probe.sbatch` is the memory check behind the L40S contact-buffer setting in `train_diag.sbatch`.
