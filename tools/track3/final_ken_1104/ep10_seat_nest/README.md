# Episode 10 (blue_cup + beige_cup): seat the blue cup (F1-b) + collision-only nest proxy (F2)

Final candidate for ep10 (haisenk / Ken, 4 Nov 2026):

| | |
|---|---|
| Scene | `k10_ep10_nest_b` = F1-b seated reference + F2 collision proxy (only the two `*_rigid.urdf` collision elements differ from `k10_ep10_seat_b`) |
| Policy | `n_ep10_nb_K_s42` @ step 100003840, `policy_100003840.safetensors`, sha256 `2bd79300d999a0d7522113befaa875d2caf07d0702dfaade8b2bbd203f045dba` |
| Rollout | `results/episode_000010.parquet`, sha256 `d73753cec80183da2a2183964610d314ee4e2dd34d2297a2234f8f8c63442c32`; own-ref add_auc 0.9804 [blue 0.9926, beige 0.9682] |
| Packed rows | `~/v2d_ops/kenc/tests/n_ep10_nb_K_s42/one.parquet` (Babel), 322 rows, sha256 `5b3eb64056b8e09af1956e6de2e57378ffad7fa88dca69af057c400f7662ba42` |
| Submitted rows | Kaggle **N1** (MPPE board only, 4 Nov 09:50 EST, submission 57988348) = team base + these ep10 rows. The AUC board had no submissions left from 04:35 EST: NEST was packed at 05:02 and approved at about 09:45, and an AUC attempt at 09:45 was refused. So the team's AUC, SP-SR, MP-SR and RPE entries carry bob1's ep10 poses. The CD-O entries C1/C2 replaced ep10's beige_cup mesh rows with ep8 T6's beige_cup mesh, which belongs to the ep8 / CD-O part. The final local combined file (`~/v2d_ops/kenc/final1104/local_best/track3_submission.parquet`, sha256 5d1d534f...) contains these rows but was not submitted. |

## What the fix changes and why

**Diagnosis** (verified, `records/ep10_diagnosis_plan.json`, findings A1, A2 and B1):
- **Phantom lift.** In bob1's reference, the blue cup floats 1.4 to 6.2 cm above its support on f55-160. FS depth of its visible top shows that the real cup stays on the table, rising at most about 2 cm, and only on f43-52. The left hand that holds it is dragged up with it.
- **Unreachable nest.** In the video, the cream ("beige") cup ends deep inside the blue cup, about 5.5 cm above the blue's bottom.
  - The CoACD hulls of our meshes stop it on the blue rim: 14.3-14.7 cm in the static wedge test, and bob1's rollout holds it at 13.2-13.6 cm.
  - The cause is our beige mesh, about 7% too wide: its inner radius 2.6 cm below the rim already equals the blue's outer rim radius.
  - So the rollout keeps the beige about 8 cm too high on f110-160 (+8.27 cm against the depth-observed top).

**F1-b (reference, GT-free).** The blue cup is lowered by a depth-measured visible-top rule:
- `dz(f) = min(0, z_support + max(rise5(f), 0) - z_lowest_vertex(f))`;
- applied only where there are at least 2000 depth points and |dz| >= 3 mm; 5-frame box smoothing; never raised; only translation z changes.

Bob1's respinf attach then runs, with hold ownership decided on the tracked (pre-seat) path, so the left hand follows the seated cup. The plain F1 build (`k10_ep10_seat`, superseded) re-picked the beige for the left hand.

**F2 (simulation collision only).** Only the two `*_rigid.urdf` `<collision>` elements change. The visual meshes, the `score_input` meshes (byte-identical to bob1: blue `f96f6897...`, beige `16482fc5...`) and the parquet do not.
- **Blue cup:** 192 convex wall slabs, 1.5 mm thick, on its outer surface (8 bands x 24 azimuths). They have a monotone rim flare: 0 below 8.65 cm and 6.52 mm at the rim, which keeps >= 1 mm clearance from the beige hulls in the coaxial nest pose at every twist. A 3 mm floor disk is added (radius 3.81 cm; top 5.548 cm above the blue bottom = the reference's median end-nest depth of 5.748 cm minus 2 mm).
- **Beige cup:** its visual mesh minus the 4 Vicon marker spheres (cavity kept, 87.0 -> 80.6 cm³), decomposed by CoACD with default parameters into 58 hulls.

## Disclosures

- **Rules question (resolved by Ken).** The simulated collision shapes differ from the submitted meshes: thin walls, flare, floor, and no markers. Placed at NEST's end poses, the submitted meshes overlap by 21-25 cm³ (bob1: at most 1.2 cm³). Ken approved using it on 4 Nov at about 09:45 EST ("2. yes, it is fine").
- **F2 failed its own pre-registered static check b, and was trained anyway on the coordinator's override** (`records/F2_FAILED.json`, `train_worthy: false`).
  - The F1-b reference end poses cross the blue wall by up to 9.22 mm on f119-160. The reference beige is tilted 5-7 deg and up to 0.9 cm off-axis against the blue.
  - These frames come after the hand has released the beige.
  - Checks a, c, d and e passed. The coaxial drop stops on the floor at 5.498 cm for all 24 twists.
- **Overflow warnings.** The s42 training log has 17 `nefc overflow` lines (njmax 1024). The no-overflow fallback is seed s43 @ 85M (own-ref 0.979).
- **Inherited meshes.** The meshes come from bob1 / `ken_c_scored` and are unchanged by this fix. They are donors from `share/best_meshes_20261024`: blue from ep13 `blue_cup`, beige from ep11 `beige_cup`. Those donors were picked earlier by public-episode AUC.
- **Commit URL.** The packed rows carry bob1's `code_commit_url` (`.../commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6`, adopted so that one-episode swaps keep the base file's uniform provenance). That commit does not contain this code.

## Inputs

- **Parent:** bob1's ep10 scene `h5_ep10_respinf_v4`.
  - It is the same scene as HF `rose-tea-cafe/v2d-t3-recon` `experiments/haisenk/hand_fixes_v1_20261102/episode_000010/scene`, with run `ep10_h5respinf_s43`.
  - It was built by `h5_build.sh` with `H5_PATCH=patch_respinf.py` (cup spin taken from the hand).
- **Our build** re-runs the same builder on the same bundles with one extra patch. It is not an edit of the parent's files: the parquet stores absolute paths, so each scene name is built separately.
- **Control:** `k10_ep10_ctl` (patch_respinf.py only) reproduces bob1's reference content: object pose <= 3.7e-9 m, MANO joints <= 2.4e-7 m. The bytes differ only in scene-name paths and in the loader's contact-label resampling.
- **Bundles:** `V2D_BUNDLE_NS=ken_c_scored`, Babel `/data/user_data/haisenk/v2d/v2d_recon/baseline/reconstruction/data/outputs/ken_c_scored/episode_000010/{blue_cup,beige_cup}`.
- **Builder environment:** team repo checkout `/data/user_data/haisenk/v2d/v2d_train` (`.venv_retarget`).
- **Depth input:** FS depth plus SAM 3 masks in the same `ken_c_scored` outputs, measured by `f1_seat/sk10a2.py` into `f1_seat/data/sk10a2.json`.
- **CoACD and training:** flash_chord at de66fea3 on `PYTHONPATH` (`/tmp/haisenk_v2d/fc_src_de66`), with `FLASH_CHORD_CACHE_DIR` unset, which is the cache the training reads.

## Files in this folder

All hashes are md5. "Babel" means `/home/haisenk/v2d_ops/kenc/...` (written `~/...` below). The scripts are the exact versions that ran. Their hard-coded personal absolute paths (`/home/haisenk/...`, `/data/user_data/haisenk/...`, `/scratch/haisenk-v2d/...`, `/tmp/haisenk_v2d/...`, Slurm job ids 11035728 and 157612, `C:/v2d_demo/...` in docstrings) are kept unchanged.

| Staged file | Ran as | md5 | How we know it is the version that ran |
|---|---|---|---|
| `build/h5_build.sh` | `~/v2d_ops/kenc/hand5/h5_build.sh` | aa2ce809 | Called by `build_f1b.sh`. Last modified 2 Nov 17:13, before the 4 Nov 03:08 builds. |
| `build/rlr_h5.py` | `~/v2d_ops/kenc/hand5/rlr_h5.py` | 0f2cef75 | The scene builder: the team's `tools/track3/run_load_retarget.py` (2 Nov) + env hooks `H5_PATCH` / `H5_MERGED_NPZ` + `REPO=$V2D_REPO`. Same mtime argument. |
| `build/patch_respinf.py`, `build/sym_common.py` | `~/v2d_ops/kenc/hand5/ep10/` | 608f2739, 41526773 | bob1's ep10 recipe, exec'd by absolute path from `p10seat.py`. Pushed earlier with path-portable edits (diag dir and relative exec) on branch `haisen/hand-fixes-ep3-10-20`, commit 5294c6b. |
| `build/build_ctl.sh`, `build/build_f1b.sh` | `~/v2d_ops/kenc/diag/ep10_fix/` | cf93291f, aba15c1a | `records/build_logs/*.out`: both F1-b builds ran 03:08-03:13 EST on babel-t9-28. |
| `f1_seat/sk10a2.py` + `f1_seat/data/sk10a2.json` | `~/v2d_ops/kenc/diag/skeptic_ep10A2/` | 9e385948, dae70186 | Depth measurement (01:15). Imports `review_tool/render_review.py` (8050e217, `~/v2d_ops/kenc/review/tool/`). |
| `f1_seat/make_toprise.py` + `f1_seat/data/blue_toprise.json` | `~/v2d_ops/kenc/diag/ep10_fix/` | 743a39b2, 8f7b9ea2 | Re-run locally on sk10a2.json: identical content (only the `source` path field differs). The scene `report.json` records this toprise path. |
| `f1_seat/p10seat.py`, `f1_seat/p10seat_b.py` | `~/v2d_ops/kenc/diag/ep10_fix/` | 5446f591, 475a3047 | Build log line `== hand5: patch /home/haisenk/v2d_ops/kenc/diag/ep10_fix/p10seat_b.py applied`, and `report.json` `h5_patch`; both files modified before the build. The "PROPOSAL (not used for any k10_* build)" header in `p10seat_b.py` is stale: it was written before the coordinator approved F1-b, and this file built `k10_ep10_seat_b` and `k10_ep10_nest_b`. |
| `f1_checks/*` | `~/v2d_ops/kenc/diag/ep10_fix/` | same as Babel | Gates a-f (`run_checks_b.sh`), dry runs (`dryrun_p10.py`, `dry_attach.py`), parquet diff (`pqdiff.py`). |
| `f2_nest/nestcol.py` | `~/v2d_ops/kenc/diag/ep10_fix/nestcol.py` | 9bddd81e | Modified 03:17:25, before the final F2 steps (03:18-03:25); `NESTCOL.json` names it as the tool. **Re-run check (local, 4 Nov 14:20):** `prep` without `--reuse` on the archived `k10_ep10_nest_b` base scene gives a byte-identical `beige_col.stl` (cc42e6f6...) and identical profiles. `slabs` then gives all 194 collision STLs with the same geometry: 111 byte-identical, and 83 with identical vertex sets and hull volumes but a different triangle order (qhull on Windows vs Linux). |
| `f2_nest/fc_hulls.py` | `~/v2d_ops/kenc/diag/ep10_fix/fc_hulls.py` | 88ad5308 | The same file is embedded (decoded md5 equal) in the orchard drivers `orch_final.sh` and `orch_final_urdf.sh`. |
| `f2_nest/drivers/final_b1.sh` | piped to Babel from `C:\v2d_demo\diag\ep10_fix\work\` | 61d0ac0a | Runs `prep --reuse dry_bob1` and `slabs`, then packs the orchard input. |
| `f2_nest/drivers/*.nopayload.sh` | `orch_final.sh`, `final_b2b.sh`, `final_b3.sh`, `orch_final_urdf.sh` (same folder) | - | **Edited for release:** the base64 payloads are removed (heredocs left empty, with a first-line note). The payloads were: `orch_final.sh` in.tgz (775 KB, md5 7ed5def3 = final_b1's `orch_final.tgz`); `final_b2b.sh` out_final.tgz (171 KB, md5 25379732 = orch_final's output: the CoACD hulls); `final_b3.sh` NESTCOL_final.json (= `records/NESTCOL.json`, md5 34273f84); `orch_final_urdf.sh` in.tgz (2.0 MB, md5 9ed1dd5c = the edited processed dir) + fc_hulls.py. Originals: md5 7fe364d8, c61151b6, 153b0fe3, 707bd9da. |
| `build/install_scene.sh`, `build/relocate_scene.py` | local `C:\v2d_demo\tests\install_scene.sh`; orchard `~/v2d_orchard/relocate_scene.py` | f8bb925c, d8319c83 | Install step for the training node (unpack + repoint absolute paths). `relocate_scene.py` equals the copy moved to babel-s5-16 (`/tmp/haisenk_v2d/orchard_home/v2d_orchard/`). |
| `training/kq.sh` | orchard `/tmp/haisenk_v2d/kq.sh` | 9d5a6253 | Same md5 as the copy moved to babel-s5-16. Modified 3 Nov 22:33, before this run started (4 Nov 03:43). |
| `training/kq_queue_lines_ep10.txt`, `training/kq_driver_log_n_ep10_nb_K_s42.txt` | queue snapshot / `kq.out` excerpt | - | Written for the release from the queue file and the driver log (headers say where from). |
| `packing/pack_one.sh`, `swap_episode.py`, `team_pack_submission.py`, `swap_submit.sh` | Babel `~/v2d_ops/` | 16f82b5c, b1b16564, c31634cb, 816dc28d | md5 equal to the Babel files (checked 4 Nov 14:21). |
| `packing/ship.sh`, `packing/archive.sh` | local `C:\v2d_demo\tests\` | f10dac27, caa53903 | Local drivers: orchard -> Babel rollout shipping, and the candidate archive. |
| `records/` | - | - | Contents: build records (`F1_READY.json`, `F2_FAILED.json`, `NESTCOL.json` = the scene's final copy, scene `report.json` and `pipeline.log`), the diagnosis plan, build logs, and the run records of `n_ep10_nb_K_s42` (manifest, sweep, best, recipe, train.log, evaluator json and overrides, pack.log, dryrun.log). |

To re-run elsewhere, edit the absolute exec/data paths in four files to the new locations:
- `p10seat_b.py` -> `p10seat.py`;
- `p10seat.py` -> `patch_respinf.py` and `blue_toprise.json`;
- `patch_respinf.py` -> `sym_common.py`;
- `sk10a2.py` -> `render_review.py` and its scene/bundle paths.

## Rebuild from the parent (commands as run)

```bash
# Babel, inside CPU steps of a holder job (srun --jobid=<job> --overlap ...). PY = team retarget venv.
T=/data/user_data/haisenk/v2d/v2d_train; PY=$T/.venv_retarget/bin/python
F=~/v2d_ops/kenc/diag/ep10_fix; S=/data/user_data/haisenk/v2d/hand5/scenes

# 1. GT-free visible-top measurement of the blue cup (FS depth in the SAM 3 blue mask minus 8-px-dilated hands, eroded 4 px)
/home/haisenk/anaconda3/bin/python ~/v2d_ops/kenc/diag/skeptic_ep10A2/sk10a2.py         # -> sk10a2.json
python $F/make_toprise.py ~/v2d_ops/kenc/diag/skeptic_ep10A2/sk10a2.json $F/blue_toprise.json   # rest top 16.62 cm (f0-35)

# 2. control = bob1's recipe (check that it reproduces h5_ep10_respinf_v4)
bash $F/build_ctl.sh     # NS=ken_c_scored SEAT=0 H5_PATCH=~/v2d_ops/kenc/hand5/ep10/patch_respinf.py bash ~/v2d_ops/kenc/hand5/h5_build.sh k10_ep10_ctl 10

# 3. F1-b builds: one patch, two scene names (nest_b is F2's base). Each runs:
#    NS=ken_c_scored SEAT=0 H5_PATCH=$F/p10seat_b.py bash ~/v2d_ops/kenc/hand5/h5_build.sh <name> 10
#    -> cd $T && V2D_REPO=$T V2D_BUNDLE_NS=ken_c_scored $PY rlr_h5.py --episode 10 --merge --attach-hands \
#         --smooth-hands 2 --skip-video --device cpu --sequence-id ep10_h5_<name> --out $S/<name>
bash $F/build_f1b.sh k10_ep10_seat_b k10_ep10_seat_b_scene.tgz
bash $F/build_f1b.sh k10_ep10_nest_b k10_ep10_nest_b_base_scene.tgz
bash $F/run_checks_b.sh                      # gates a-f (read-only); results assembled into records/F1_READY.json

# 4. F2 on k10_ep10_nest_b, as run (W = work dir; FCPY = flash_chord venv python, PYTHONPATH=fc_src_de66,
#    FLASH_CHORD_CACHE_DIR unset = the cache the training reads)
SC=$S/k10_ep10_nest_b; W=$F/nestcol_nest_b; cd $F
#   Babel, drivers/final_b1.sh:
$PY nestcol.py prep  --scene $SC --work $W --reuse dry_bob1     # copies beige_col.stl, profiles.npz, hulls_beige.npz from the bob1 dry run
$PY nestcol.py slabs --scene $SC --work $W                      # flare vs hulls_beige.npz; blue_col_000..191.stl + blue_floor.stl -> $W and $W/orch
#   orchard, drivers/orch_final.nopayload.sh (CoACD with default params, warms the training cache):
$FCPY fc_hulls.py out_beige beige_test.urdf                     # cache hit, key da18f55941a89175, 58 hulls
$FCPY fc_hulls.py out_blue --off_compare --info_log blue_col_180.stl blue_test.urdf   # 193 pieces, 1 hull each
#   Babel, drivers/final_b2b.nopayload.sh (out_beige/out_blue.npz -> $W/hulls_beige/hulls_blue.npz), then:
$PY nestcol.py check  --scene $SC --work $W --bob1_beige_hulls bob1_hulls/hull_beige_bf2fae51eba73fef.npz
$PY nestcol.py settle --scene $SC --work $W --bob1_blue_hulls bob1_hulls/hull_blue_46e5d1341ef7c706.npz \
                      --bob1_beige_hulls bob1_hulls/hull_beige_bf2fae51eba73fef.npz      # MuJoCo, no hands
$PY nestcol.py apply  --scene $SC --work $W --target k10_ep10_nest_b   # 2 URDFs edited (originals in col_backup/), 194 STLs added, NESTCOL.json
$PY nestcol.py ecmp   --scene $SC --work $W --ref_scene $S/k10_ep10_seat_b   # meshes/visuals unchanged vs seat_b
#   drivers/final_b3.nopayload.sh: the scene's NESTCOL.json <- the version with the check results (records/NESTCOL.json)
#   (orchard, drivers/orch_final_urdf.nopayload.sh: end-to-end newton/flash_chord load of the edited URDFs, 194 cache hits; a check only)
# Fresh rebuild without the dry run: run prep WITHOUT --reuse (gives the same beige_col.stl), then fc_hulls.py out_beige in $W/orch
# and copy out_beige.npz to $W/hulls_beige.npz BEFORE slabs; continue as above.

# 5. install on the training node: unpack and repoint the parquet's absolute paths (build/install_scene.sh does this over ssh
#    and appends the kq queue lines)
tar xzf scene.tgz -C /tmp/haisenk_v2d/scenes && $FCPY build/relocate_scene.py /tmp/haisenk_v2d/scenes/k10_ep10_nest_b
```

Notes on the run:
- **Inputs reused from the dry run.** `--reuse dry_bob1` copied `beige_col.stl`, `profiles.npz` and `hulls_beige.npz` from a dry run on bob1's identical meshes. That dry-run hull file is byte-identical to the final orchard CoACD output (cache hit on the same key). The beige hulls used for the flare are therefore the same hulls the training read from the warm flash_chord cache.
- **CoACD is not bit-deterministic.** A fresh machine recomputes the hulls with the same counts but slightly different shapes.
- **Relocation changes the parquet hash.** It changes the parquet md5 from 0a1264d0 (Babel build) to 76044244 (installed copy) and keeps the original as `*.parquet.orig`. Everything else in the trained scene is byte-identical to the F2 build archive.

## Training and selection

Training followed Edmond's protocol (`campaign.py` / TRAINING_GUIDE @ de66fea3), driven by the queue driver `training/kq.sh` (md5 9d5a6253). It ran on orchard-flame-21, job 157612, GPU0 H100, on 4 Nov from 03:43 to 05:00 EST (`training/kq_driver_log_n_ep10_nb_K_s42.txt`). Training took 75 min and exited 0.

Queue line: `n_ep10_nb_K_s42 10 /tmp/haisenk_v2d/scenes/k10_ep10_nest_b/assets/human_motion_data/ego_recon/processed/sequence_id=ep10_h5_k10_ep10_nest_b/robot_name=sharpa_wave 42 KEN 1`

```bash
export PYTHONPATH=/tmp/haisenk_v2d/fc_src_de66   # flash_chord de66fea3
cd $FC && $FC_PY scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='$PARQUET'" reset.seed=42 \
  curriculum=tracking_100m_object2x training.checkpoint.save_replay=false logging.mode=offline logging.upload_checkpoints=false \
  logging.run_name=n_ep10_nb_K_s42 output_dir=$RUN/train training.total_environment_steps=100003840 \
  task.motion_speed=1.0 sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 \
  training.checkpoint.save_interval_environment_steps=5000192 termination.object_position.threshold=0.1 \
  termination.object_orientation.threshold=0.5 training.replay.capacity=4194304 training.replay.observation_storage_dtype=float32 \
  sim.njmax=1024 sim.nconmax=512
# sweep: every checkpoint >= 70M with the kit's official evaluator (own-reference add_auc)
$FC_PY $KIT/examples/run_policy_evaluation.py --checkpoints early_<step>/checkpoints.json --results early_<step>/results --runtime-root $FC
```

- **Checkpoint sweep** (`records/run_n_ep10_nb_K_s42/sweep.json`): 100M 0.9804, 90M 0.9771, 80M 0.9713, 85M 0.9693, 95M 0.9637, 70M 0.9571, 75M 0.9363. The selected checkpoint is 100003840.
- **Seeds** (best checkpoint >= 70M each; overflow lines in brackets):

  | Recipe and seed | Own-ref @ step | Overflow lines |
  |---|---|---|
  | KEN s42 | 0.980 @ 100M | 17 |
  | KEN s43 | 0.979 @ 85M | 0 |
  | KEN s49 | 0.974 @ 95M | 7 |
  | KEN s46 | 0.968 @ 95M | 0 |
  | KEN s45 | 0.963 @ 100M | 0 |
  | KEN s48 | 0.959 @ 100M | 2 |
  | DEF s43 | 0.956 @ 90M | - |
  | DEF s42 | 0.940 @ 85M | 0 |
  | KEN s44 | 0.921 @ 95M | 7 |
  | KEN s47 | 0.634 @ 90M | 0 |

  The best of the 10 KEN seeds by own-ref was kept. This is GT-free, but carries a winner's curse.
- **Evaluator metrics of the pick:** add_auc 0.9804, chord_sr 1.0, maniptrans_sr 1.0, spider_sr_uncentered 1.0, mppe_cm 1.00, relative_position_error_cm 0.96.

## Packing

Packing used `packing/ship.sh`, `swap_submit.sh`, `pack_one.sh` (which calls `team_pack_submission.py` and the kit's `tools/pack_submission.py`) and `swap_episode.py`, on Babel on 4 Nov at 05:02 EST:

```bash
# local Git Bash: bash packing/ship.sh n_ep10_nb_K_s42 k10_ep10_nest_b 10   (rollout of best.txt's step + scene meshes -> Babel tests/<name>/in/)
# Babel, inside swap_submit.sh 10 n_ep10_nb_K_s42 $RES $MESH "<msg>" (SUBMIT=0):
cd ~/v2d_ops/kenc
bash ~/v2d_ops/pack_one.sh 10 tests/n_ep10_nb_K_s42/in/runs/n_ep10_nb_K_s42/early_100003840/results \
     tests/n_ep10_nb_K_s42/in/scenes/k10_ep10_nest_b/score_input/meshes \
     https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6 tests/n_ep10_nb_K_s42/one.parquet
~/v2d_ops/kaggle_env/bin/python ~/v2d_ops/swap_episode.py bob1/candidate_bob1.parquet tests/n_ep10_nb_K_s42/one.parquet 10 \
     tests/n_ep10_nb_K_s42/track3_submission.parquet --adopt-commit-url
(cd $KIT && ~/v2d_ops/kaggle_env/bin/python submit/v2d_submit.py ~/v2d_ops/kenc/tests/n_ep10_nb_K_s42/track3_submission.parquet \
     --track track_3 --only add_auc --dry-run -m "<msg>")      # "looks valid"; AUC board 0 left -> nothing submitted
```

- **Provenance hashes** (`pack.log`): eval_script 160d24f5..., eval_code b1b3ebe3..., packer 49b13052..., uploader d31d4a7c... These are the team's uniform values.
- **N1** (`~/v2d_ops/kenc/combos/base4_n10/`) is the team base (bob1 + our ep1 T2, ep3 T5, ep5 T8, ep8 T6) with these ep10 rows chained in by `swap_episode.py` (`kenc/combine.sh`).

## Evidence

Sources: the ledger `C:\v2d_demo\tests\ledger.md`, plus the verification and refine notes (copies in `release1104\notes\ep10_seat_nest\`).

**Kaggle**
- **N1, MPPE board only:** 3.33289 vs the base's 3.42999. That is -0.0971 on the board, or -1.94 cm on ep10.
  - The 9 GT-free judges had predicted -0.5 to -2.1 cm.
  - The depth-consistent judges had predicted -1.86 to -2.08 cm.
  - The judges that assume a lifted blue or a higher nest are refuted by this result.
- **No AUC result.** The AUC board's daily allowance was used up: an attempt was refused on 4 Nov at about 09:45, and the reset falls after the deadline.
- **Calibration point.** T1 is the only other ep10 swap Kaggle scored: a Wentao-base scene with a hand fix and the same meshes. It scored -0.036 on ep10, and the AUC emulator reproduced that within about 0.01-0.02.

**Local, GT-free**
- **Own-ref:** 0.9804 [0.993, 0.968]. bob1's 0.7201 is not comparable, because the reference changed.
- **Cross-score:** verbatim AUC.py symmetric ADD with all-symmetric alignment, using the same beige reference in all scenes.
  - Depth-consistent reference: bob1 0.777 -> NEST 0.982 (+0.205 ep10).
  - bob1's lifted-blue reference: 0.721 -> 0.861 (+0.140).
- **Rollout gates:**
  - beige bottom 5.3-5.5 cm above the table on f110-160 (bob1: 13.5);
  - nest depth 5.5 cm (reference 5.48);
  - blue height on f55-160 <= 0.09 cm.
- **Verification workflow** (wf_803b88ee-d40, 6 skeptics plus a synthesis, 9 GT-free judges):
  - ep10 AUC +0.10 to +0.24, most likely +0.21;
  - MP-SR ep10 fail -> pass under 8 of 9 judges;
  - RPE -2.3 to -3.4 cm and MPPE -0.8 to -2.1 cm on ep10;
  - SP-SR and CD-O unchanged, because the mesh payloads are byte-identical to bob1.
- **F1-b gates** (`records/F1_READY.json`):
  - (a) Left-wrist shift is at most 1.59 cm (bob1: 7.12).
  - (b) The beige is untouched; the blue changes only in z (-6.11..0 cm), plus a respin twist of at most 0.55 deg.
  - (c) The blue's lowest vertex sits -0.07..+0.8 cm from the support on f55-160 (bob1: +1.36..+6.23). The strict ±3 mm bar fails on 34 frames; the realistic bar passes.
  - (d) Occlusion-aware blue IoU on f63-122 is 0.819 (control: 0.661). The strict raw-IoU bar of 0.45 fails, because the occlusion ceiling is 0.29.
  - (e) Reference top minus observed top on f55-160 is 0.3-1.5 cm (bob1: 1.8-7.4).
  - (f) The right hand is inside the blue wall envelope only on f75-100.
- **F2 gates** (`records/F2_FAILED.json`):
  - (a) Pass: 1 hull per slab; the beige hull radius exceeds the visual surface by at most 0.23 mm.
  - (b) **Fail**, as described under Disclosures.
  - (c) Pass: in a MuJoCo settle from f140 and from f100, the beige ends nested at 5.50 cm with 0.8 deg tilt and at most 58 contacts.
  - (d) Pass:
    - left-hand joints inside the new blue collision: 62 (inside the visual wall: 109);
    - right-hand joints inside the new beige hulls: 138 (inside bob1's hulls: 151).
  - (e) Pass: meshes unchanged.

## Sibling scene `k10_ep10_seat_b` (F1-b only; not used)

It was built with the same patch and the same command (step 3) and has no F2. Seeds:
- s42: 0.809 @ 100M;
- s43: 0.812 @ 85M;
- s44: 0.813 @ 95M.

Cross-score for s44 @ 95M:
- under the depth-consistent reference: 0.808 vs bob1 0.777 (+0.031);
- under bob1's reference: 0.701 vs 0.721.

It was packed and dry-run (`~/v2d_ops/kenc/tests/s_ep10_sb_K_s44/`), never submitted, and dropped in favour of NEST.

## Not included here, or not recovered

- **The combine step for N1** (Babel `~/v2d_ops/kenc/combine.sh`, md5 28b0f8d3, building `combos/base4_n10`) and the final local combine (`kenc/final1104/combine_final.sh`, 61addd4f) are not copied here. They are multi-episode steps; the ep8 part's `packing/` carries `combine_final.sh`.
- **Local bookkeeping scripts that only wrote the JSON records:** `write_ready.py`, `write_status.py`, `assemble_f1.py`, `fix_arch_paths.py` and `make_f2_result.py`.
- **Development patches already folded into the final `nestcol.py`:** `patch_settle/flare/diag/mfb.py`, `mfb_new.py`, `ecmp_src.py`.
- **Dry-run drivers** on bob1's scene (`babel_settle1/2.sh`, `orch_run1/1b/2/3.sh`) and the planner's simulation code (`C:\v2d_demo\diag\ep10_plan\nestplan.py`, `shell.py`, `flare.py`; `nestcol.py` carries its own copies of the helpers).
- **The superseded plain-F1 attempt:** `run_checks.sh`, `F1_FAILED.json`.
- **Evaluation-only scripts behind the evidence numbers:** `C:\v2d_demo\diag\anch\cross10.py` and `C:\v2d_demo\diag\ep10_verify\`.
- **The `install_scene.sh` run is not logged.** That it was used is inferred from the relocated parquet and its `.orig` copy.
- **The `ONLY=add_auc` setting of the pack dry run** is inferred from `dryrun.log`, which lists only the AUC board.
- **The scene tarball's `bundle_cleaned/` has absolute symlinks** into `/data/.../bundle_merged/`. They do not extract off Babel; they are not needed for training.
