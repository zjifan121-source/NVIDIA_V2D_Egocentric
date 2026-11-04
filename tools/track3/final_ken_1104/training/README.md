# training/: kq.sh queue driver (Edmond's de66fea3 protocol) and its queue helpers

This driver trained and selected the final policies for episodes 1, 3, 5, 8 and 10. Episode 6 is Edmond's bob1 entry
(scene eb1_ep6, policy ep6_bm_s42@100M). Edmond trained and released it, so nothing here touches ep6.

## What it does and why

- `kq.sh` is a small GPU queue that runs Edmond's training protocol from team commit `de66fea3`
  (`reports/policy_targeted_2026-10-28/TRAINING_GUIDE.md`, `campaign.py`, `fresh_keep.sbatch`) on one node, without Slurm arrays.
  It runs one FlashSAC training per GPU with the same command line as `fresh_keep.sbatch`:
  `scripts/train_flash_sac.py experiment=sharpa_flash_sac task.parquet=... reset.seed=S curriculum=tracking_100m_object2x
  training.checkpoint.save_replay=false logging.mode=offline ... training.total_environment_steps=100003840` plus the recipe overrides.
  It saves a policy checkpoint every 5,000,192 steps. After training it evaluates **every checkpoint >= 70M** with the kit's official
  evaluator (`assets/data/v2d_submission_kit/examples/run_policy_evaluation.py`, own-reference `add_auc`) and writes
  `sweep.json` (all checkpoints, best first) and `best.txt`.
- Why the sweep: TRAINING_GUIDE says "Evaluate every checkpoint from 70M, keep the best". The best checkpoint was not the last one
  in 262 of 371 runs, and it was 0.06 better on average. Single evaluations are also noisy. For example, checkpoint 100M of the ep5
  run `o_ep5_attcom_s42` scored 0.478 in the old driver's evaluation and 0.8465 in the kq.sh sweep.
- Recipes. These are the same overrides as `campaign.py` RECIPES at de66fea3. We checked them against `DEFAULT`, `BIG={10}`, the
  `KEN` substitutions and the `WIDE` extras.

  | recipe | overrides on top of COMMON (`task.motion_speed=1.0 sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 training.checkpoint.save_interval_environment_steps=5000192`) |
  |---|---|
  | **KEN** (used by all 5 finals) | object termination 0.1 m / 0.5 rad, replay 4,194,304 float32, `sim.njmax=1024 sim.nconmax=512` on every episode |
  | DEF | object termination 0.3 m / 1.5 rad, replay 1,048,576 float16, njmax 1024 / nconmax 512 (ep10: 2048 / 1024) |
  | WIDE | DEF + `action.normalized_mapping=linear training.network.initial_normalized_std=0.3 training.network.log_std_max=-0.5108256237659907 training.exploration.target_sigma=0.3` |
  | SWEEP | no training; only re-sweeps an existing run's checkpoints |

## Team-repo code this depends on (not copied here)

- **flash_chord at commit `1a6bae7b2682a40087cd70e5d68c6aa3e5f21b77`** (origin/edm_training, "Add a no-Slurm training queue and
  night run sheet"). It comes from a sparse checkout of `baseline/robotic_grounding/flash_chord`, `tools/track3` and
  `assets/data/v2d_submission_kit`, with the venv built by `uv sync --frozen --extra train` (`orch_setup.sh`).
- **Overlay `fc_src_de66`, put first on `PYTHONPATH`.** It is a copy of `flash_chord/src` at 1a6bae7b where two files are
  replaced by their **de66fea3** versions:
  - `flash_chord/lifecycle/reset.py`: git blob `d8d005c0`, +19 lines, the reset focus window, from de66fea3.
  - `flash_chord/training/flash_sac/replay.py`: git blob `1eca045a`, +25 lines, from commits 0f466fb5 "Keep non-finite transitions
    out of FlashSAC replay" and 6bc05388 "Drop float16-overflowing obs from FlashSAC replay".

  Checked on babel-s5-16 on 4 Nov: `diff -rq` against the repo src shows only these two files. All asset files are byte-identical;
  the overlay's asset folder only has three extra `__pycache__/*.pyc` files. Between 1a6bae7b and de66fea3, flash_chord also gained
  five curriculum YAMLs; the finals do not use them. Each final run's `train.log` imports from `/tmp/haisenk_v2d/fc_src_de66/flash_chord/...`.
- **The kit**: `examples/run_policy_evaluation.py` (sha256 `0bcc41f5...`) is used for the checkpoint sweep.
- **The team loader** `tools/track3/run_load_retarget.py` (at 1a6bae7b: blob `43d29ed0`). It writes the scene layout that every script
  here expects: `<scene>/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet` and
  `<scene>/score_input/meshes/episode_XXXXXX/`. The install scripts check the parquet columns `raw_motion_file`, `object_mesh_paths`
  and `object_urdf_paths`, and relocate them with `../tools/relocate_scene.py`.

Overlay as built (orchard, 3 Nov 22:33 EST, from the session log). The two de66 files were downloaded from GitHub; their git blob
hashes equal those at de66fea3, so `git show de66fea3:<path>` gives the same bytes:

```bash
B=/tmp/haisenk_v2d   # de66_reset.py / de66_replay.py = the two files at de66fea3
cp -a $B/v2d_train/baseline/robotic_grounding/flash_chord/src $B/fc_src_de66.tmp
cp $B/de66_reset.py  $B/fc_src_de66.tmp/flash_chord/lifecycle/reset.py
cp $B/de66_replay.py $B/fc_src_de66.tmp/flash_chord/training/flash_sac/replay.py
mv $B/fc_src_de66.tmp $B/fc_src_de66
```

## Files

| file | role | where it ran | md5 | how we know it is the version that ran |
|---|---|---|---|---|
| `kq.sh` | queue driver: train + sweep | orchard-flame-21 job 157612 (3-4 Nov), then babel-s5-16 job 10774619 (4 Nov) | `9d5a6253fdee63942a1b309c4ed23ae2` | Same md5 in the local copy and in babel-s5-16 `/tmp/haisenk_v2d/kq.sh`, which is the orchard file after the md5-verified move. Its mtime (3 Nov 22:33:07 EST) is before both driver starts (orchard 22:34:02, Babel 4 Nov 10:20:20), and it was not edited after. See "Exactness checks" below. |
| `kq_start.sh` | starts kq.sh as one `srun --overlap --whole` step (8 H100) on Babel | Babel login node, piped through `ssh babel "bash -s"` | `60a9e3447121a1d5c21a9a13b4b68073` | Never stored on Babel (sent over stdin), so this is the local file that was sent. The driver header (`kq_babel.txt`, `GPUs 0 1 2 3 4 5 6 7`) matches. |
| `kq_add_b.sh` | append queue lines (stdin) to `/tmp/haisenk_v2d/kq_babel.txt`; skips duplicates | Babel step | `ecbe3664f15307697a849b12a8bbcafe` | Same md5 as `~/v2d_ops/kq_add_b.sh` and `/tmp/haisenk_v2d/kq_add.sh`. Its header calls it `kq_add.sh`; the local copy has that name. |
| `kq_stat_b.sh` | GPU use, driver log, queue, per-run progress | Babel step | `fda55637b2ea9b4368a8a793c5a3e81d` | Same md5 as `~/v2d_ops/kq_stat_b.sh` (header name `kq_stat.sh`). |
| `install_scene_b.sh` | copy a built scene to `/tmp/haisenk_v2d/scenes/NAME`, relocate, check paths, print PARQUET_DIR; refuses to overwrite | Babel step | `e9fe7124994d9eb30ddc97a864ad8478` | Same md5 as `~/v2d_ops/install_scene_b.sh`. |
| `install_scene.sh` | orchard version: upload scene tgz, unpack, relocate, check, queue `RUNPREFIX_s<seed>` lines | local PC -> orchard | `f8bb925cd59990344865b5d0348f732b` | Local-only script. Last edited 4 Nov 03:09 EST, before it installed `k10_ep10_nest_b` and queued `n_ep10_nb_K_s42` (run start 03:43). |
| `kqstat.sh`, `kqq.sh` | orchard status helpers (sweep summary; queue state) | orchard node | `821e5d15...`, `7b0c3813...` | Same md5 as the moved orchard copies on babel-s5-16. |
| `orch_setup.sh` | orchard env: sparse clone at 1a6bae7b, uv 0.13.4, `uv sync --frozen --extra train`, smoke test | orchard step | `c1d1896c954f2e59e6b9ae59797d90d6` | Same md5 as `orchard_home/v2d_orchard/orch_setup.sh`. Its log (3 Nov 18:35 EST) shows commit 1a6bae7b, jax 0.10.2 on 7 GPUs, warp 1.14.0, `ORCH_SETUP_DONE`. |
| `_babel_kq_setup.sh` | Babel: link `/tmp/haisenk_v2d/v2d_train` -> `/data/user_data/haisenk/v2d/v2d_train` (same commit, py3.11 uv venv), smoke test with the overlay | Babel step of 10774619 | `f4e286225a1af4550b27255074009342` | Local copy (4 Nov 09:50); it was sent over stdin. |

The Babel venv is `/data/user_data/haisenk/v2d/v2d_train/baseline/robotic_grounding/flash_chord/.venv` (uv 0.13.4, CPython 3.11,
`pyvenv.cfg` dated 24 Oct 22:47). The clone's HEAD is 1a6bae7b, authored 24 Oct 22:06. `git status` and `git diff` show no local
changes under `flash_chord`, `tools/track3` or the kit; the only changes are data folders under `baseline/reconstruction` and
`assets/data/track_3`. The orchard venv used CPython 3.12 (uv sync on 3 Nov). Three finals trained on orchard (py3.12; ep3, ep5,
ep10) and two on Babel (py3.11; ep1, ep8).

## Commands as run

Orchard (ep3, ep5 and ep10 finals; orchard no longer exists, and the same paths were later reused on Babel):

```bash
bash orch_setup.sh            # inside a step of 157612. Robot meshes are Git LFS and were copied from Babel (fc_assets.tgz).
# fc_src_de66 overlay: see above
tmux new -d -s v2dkq "srun --jobid=157612 --overlap -n1 -c64 --mem=700G --job-name=v2d-orch-kq bash -c 'bash /tmp/haisenk_v2d/kq.sh /tmp/haisenk_v2d/kq.txt 0 1 2 3 4 5 6 > /tmp/haisenk_v2d/kq.out 2>&1'"
# ep10 scene + queue lines, from the PC:
bash install_scene.sh /c/v2d_demo/archive/k10_ep10_nest_b/scene.tgz k10_ep10_nest_b 10 "42 43" KEN 1 n_ep10_nb_K
```

Babel (ep1 and ep8 finals). `/tmp/haisenk_v2d` was moved from orchard to babel-s5-16 at the same path, so the scene and queue
paths are unchanged:

```bash
ssh babel "bash -s" < kq_start.sh                                  # login node -> step v2d-kq; log ~/v2d_ops/logs/kq_babel.out
ssh babel "srun --jobid=10774619 --overlap -n1 -c2 --mem=8G bash ~/v2d_ops/install_scene_b.sh <built scene dir> <NAME>"
printf '%s\n' "<queue line>" | ssh babel "srun --jobid=10774619 --overlap -n1 bash ~/v2d_ops/kq_add_b.sh"
ssh babel "srun --jobid=10774619 --overlap -n1 bash ~/v2d_ops/kq_stat_b.sh"
```

Queue lines of the finals (`NAME EP PARQUET_DIR SEED RECIPE PRIO`, copied from `kq.txt` and `kq_babel.txt`; `.../processed` stands
for `/tmp/haisenk_v2d/scenes/<scene>/assets/human_motion_data/ego_recon/processed`):

```
h_ep3_sc_K_s42   3  .../sc_ep3/.../processed/sequence_id=ep3_h5_h5_ep3_A1/robot_name=sharpa_wave            42 KEN 2   (kq.txt:58)
r_ep5_rl_K_s43   5  .../rl2_ep5/.../processed/sequence_id=ep5_hw_w7_attcom_ep5/robot_name=sharpa_wave       43 KEN 2   (kq.txt:78)
n_ep10_nb_K_s42  10 .../k10_ep10_nest_b/.../processed/sequence_id=ep10_h5_k10_ep10_nest_b/robot_name=sharpa_wave 42 KEN 1 (kq.txt:92)
b_ep1_wh_K_s48   1  .../w_hold_ep1/.../processed/sequence_id=ep1_hw_w_hold_ep1/robot_name=sharpa_wave        48 KEN 20  (kq_babel.txt:4)
rf_ep8_ds_K_s43  8  .../rf_ep8_ds/.../processed/sequence_id=ep8_ken_c_scored_attach/robot_name=sharpa_wave   43 KEN 10  (kq_babel.txt:10)
```

## The five final runs (driver logs `kq.out` / `kq_babel.out`, each run's `sweep.json`)

| final | host, start (EST) | train min | nefc overflow lines | own-ref add_auc per checkpoint 70/75/80/85/90/95/100M | packed checkpoint |
|---|---|---|---|---|---|
| ep1 `b_ep1_wh_K_s48` | Babel gpu3, 4 Nov 10:21 | 49 | 0 | .562 .763 .793 .804 .814 .819 **.830** | 100003840 (kq best). Then chosen over T2 `o_ep1_hold_s44` and the other `w_hold_ep1` seeds by the Kaggle-calibrated ep1 emulator: better in 16/16 cells, median 0.887 vs 0.847. See the ep1 part. |
| ep3 `h_ep3_sc_K_s42` | orchard gpu1, 4 Nov 00:48 | 33 | 0 | .936 .930 .945 .937 .937 **.947** .783 | 95006720 (kq best; Kaggle T5) |
| ep5 `r_ep5_rl_K_s43` | orchard gpu6, 4 Nov 02:05 | 38 | 0 | .886 .875 .888 .907 **.910** .909 .907 | 95006720: best by the anchored own-ref (AUC.py-style frame-0 alignment, `diag/anch/anch.py`): 85M .901, 90M .901, **95M .909**, 100M .902. Kaggle-tested as T8. kq.sh's plain best was 90M (0.9104 vs 0.9089). |
| ep8 `rf_ep8_ds_K_s43` | Babel gpu6, 4 Nov 11:07 | 51 | 0 | .978 .977 .984 .987 .978 **.990** .988 | 95006720 (kq best; then judged against T6 under 5 GT-free stand-ins, see the ep8 part) |
| ep10 `n_ep10_nb_K_s42` | orchard gpu0, 4 Nov 03:43 | 75 | 17 (KEN keeps njmax 1024 on ep10) | .957 .936 .971 .969 .977 .964 **.980** | 100003840 (kq best) |

Kaggle evidence for the trained finals, from `C:\v2d_demo\tests\ledger.md` (each single swap = bob1, AUC 0.75166, with one
episode replaced):

- T5, ep3 `h_ep3_sc_K_s42@95M`: 0.75305 (ep3 +0.028).
- T8, ep5 `r_ep5_rl_K_s43@95M`: 0.75799 (ep5 +0.127).
- The team's combined entry (bob1 + T2 + T5 + T8 + T6) scored AUC **0.76526**. This is the final AUC; the AUC board closed before
  the deadline.
- ep10 NEST `n_ep10_nb_K_s42@100M` was GT-checked only on MPPE: 3.42999 -> **3.33289** (submission 57988348).
- The ep1 (`b_ep1_wh_K_s48`) and ep8 (`rf_ep8_ds_K_s43`) finals were **not submitted**. Ken stopped submissions at 10:05 EST on
  4 Nov. They replace T2 and T6, which were Kaggle-tested on the same scene families.

## Exactness checks done on 4 Nov (read-only)

1. **kq.sh**: md5 is identical in three places (local, orchard -> babel-s5-16, and the file the driver ran). The orchard driver
   started at 22:34:02, 55 s after the file's mtime. Babel started the same file.
2. Each final run's `recipe.txt` is written by `kq.sh` `task()`. All five are byte-identical (md5 `15bde716...`) and equal
   `KEN ` + this kq.sh's KEN override string.
3. Each final policy's own metadata (`user.resolved_config_json` in the safetensors header) shows exactly what kq.sh passes.
   - Scene and seed: `task.parquet` = the queued scene, `reset.seed` = the queued seed.
   - Task and sim: motion_speed 1.0, sim.fps 20.254119873046875, njmax 1024 / nconmax 512.
   - Termination: wrist 0.3, object 0.1 / 0.5.
   - Replay and checkpoints: replay 4,194,304 float32, save interval 5,000,192, total 100,003,840, save_replay false.
   - Logging: offline, `run_name` and `output_dir` = `/tmp/haisenk_v2d/runs/<NAME>/train`.
   - The 11 curriculum stages (thresholds, voc scales, object_keypoints 40) are identical to `tracking_100m_object2x.yaml` at 1a6bae7b.

## Hard-coded paths and side effects (kept as they ran)

- `kq.sh`: `/tmp/haisenk_v2d` (repo link, runs, `PYTHONPATH=/tmp/haisenk_v2d/fc_src_de66`). Its header comment names orchard job 157612.
  `prune_states` deletes all but the newest two `train/state_*.safetensors` resume states inside each run dir.
- `kq_start.sh`: Babel job `10774619`, `/tmp/haisenk_v2d`, `/data/user_data/haisenk/v2d/v2d_train`, log `~/v2d_ops/logs/kq_babel.out`.
- `kq_add_b.sh`, `kq_stat_b.sh`, `install_scene_b.sh`: `/tmp/haisenk_v2d`, `~/v2d_ops/logs`.
  `install_scene_b.sh` calls `/tmp/haisenk_v2d/orchard_home/v2d_orchard/relocate_scene.py` (= `../tools/relocate_scene.py`).
- `install_scene.sh`: `ssh orchard`, job 157612, `/tmp/haisenk_v2d`, `$HOME/v2d_orchard/relocate_scene.py`, and a local temp file `/tmp/_inst_<NAME>.sh`.
- `kqstat.sh`, `kqq.sh`: `/tmp/haisenk_v2d`.
- `orch_setup.sh`: `/tmp/haisenk_v2d`, clone URL of the team repo.
- `_babel_kq_setup.sh`: `/tmp/haisenk_v2d`, `/data/user_data/haisenk/v2d/v2d_train`.
- These scripts handle no credentials.

## Not recovered / caveats

- Orchard (orchard-flame-21) is gone, so the orchard command lines cannot be re-run as written.
- How the ep3 and ep5 queue lines were appended on orchard is not recorded; the lines themselves above are exact. The exact ssh
  line used to run `_babel_kq_setup.sh` is not recorded either; its header says "run inside a step of 10774619".
- The run folders on babel-s5-16 `/tmp/haisenk_v2d` disappear when job 10774619 ends (about 6 Nov 00:03). Final policies, results
  and scenes are archived in `C:\v2d_demo\archive\<candidate>\` and in the HF bundle manifest. The driver logs and both queues
  (`kq.out`, `kq.txt`, `kq_babel.txt`, `kq_babel.out`) were copied on 4 Nov 14:47 to `release1104/notes/shared_tooling/driver_logs/`.
- Not staged: the old driver used for the T2/T3 parent runs, the team's `tools/track3/jobs/gpu_queue.sh` via
  `orch_run2.sh`. None of the five finals came from it.
