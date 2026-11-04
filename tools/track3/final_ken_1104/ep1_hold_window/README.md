# Episode 1: hold-window hand fix (scene `w_hold_ep1`, policy `b_ep1_wh_K_s48` @ 100,003,840 steps)

Part of the Ken / haisenk final release for V2D Track 3 (our scored episodes 1, 3, 5, 6, 8 and 10), staged 4 Nov 2026.

| | |
|---|---|
| Final ep1 choice (bundle we decided on) | scene **`w_hold_ep1`** + policy **`b_ep1_wh_K_s48` at step 100003840** (sha256 `6aaab5d6…51152`), rollout sha256 `b3fbfe3b…`, packed `one.parquet` sha256 `7150f4e5…` |
| Kaggle status of that choice | **Not submitted.** Ken stopped all Kaggle submissions on 4 Nov at about 10:05 EST; the choice was made afterwards with GT-free local judges only (section 4). |
| ep1 rows that are actually on the leaderboard | **T2 = `o_ep1_hold_s44` at step 100003840 on the same scene `w_hold_ep1`** (policy sha256 `969b9cef…`). These rows went in as swap test T2 (3 Nov 22:44 EST) and are the ep1 rows of the team's combined entry (AUC 0.76526). T2 is documented here as well (sections 4.1 and 5.1), because it used the same scene code. |

Everything below is GT-free unless it says "Kaggle". Files referenced as `scene_build/…`, `diagnosis/…`, `selection/…`
are in this folder. The data (scene, policies, rollouts, packed parquet) is not in git: see `bundle_manifest.json`
and section 6.

## 1. What the fix changes and why

**Diagnosis (confirmed, 3 Nov, `diagnosis/`, review in `notes/ep1_hold_window/r2_ep1_diagnosis_and_review_20261103.json`).**
In ep1 the left hand takes the white pot's handle at about f30 and keeps it until about f305; the right hand lifts the
lid off the pot (f45-67) and puts it back (f250-272). The team loader's `attach_hands` ("nearest within reach") only
attaches a hand while the object is lifted more than 2 cm above its frame-0 height, so on Wentao's ep1 bundle it
attaches the left hand to the pot only on [127,221) and the right hand to the lid only on [47,59) and [255,272).
GT-free evidence shows the holds are much longer: the left hand's MANO contact labels cover the pot over f28-308, the raw
(unattached) left wrist is one stable grip in the pot frame over f30-305 (phase medians within 1.9 cm of the lift
median, 0.4 cm deviation within each table phase), and the lid moves faster than 0.5 cm/frame over f45-66 and f251-272
(its descent below the frame-0 height does not count as "lifted"). In addition, Wentao's delivered hands are our *old
cleaned* hands attached to our *old* object tracks, so against his objects the delivered grip wobbles (left wrist
1.5 cm median / 2.6 cm p90 off a fixed grip during the lift); the as-delivered control trains to own-ref 0.275.

**Fix (only the hand wrist arrays change; objects, camera and meshes stay Wentao's byte for byte).**
1. `wprep.py rawhands`: copy Wentao's ep1 bundle and replace every `hand_left_*` / `hand_right_*` array with our raw
   pre-cleanup hands (`ken_c_scored_attach/bundle_merged`), re-expressed in his world:
   `wrist_to_world = his camera_to_world @ raw wrist_to_camera`.
2. Team loader `run_load_retarget.py` (commit `5294c6b`) with hand-only cleanup against his objects:
   `--no-rest-hold --no-object-clean --no-level --attach-hands --smooth-hands 2`.
3. `H5_PATCH=patch_whold.py` (exec'd by the loader's opt-in hook before `cleaned_bundle()`) redefines `attach_hands`:
   the team rule runs as before, then evidence-backed windows override the per-frame owner:
   left hand -> `white_pot` (slot 0) on [32,304); right hand -> `white_pot_lid` (slot 1) on [45,68) and [250,274).
   The grip is the median hand-in-object offset over each hold's interior (rotation mean + median translation)
   instead of the team's offset at the single closest-wrist frame. Grip deviation over the left hold drops from
   19.3 mm / 17.7 deg (team grip) to 12.2 mm / 8.5 deg (median grip). Windows and grip mode can be overridden with
   the env vars `EP1N_FORCE` (JSON) and `EP1N_GRIP`; the defaults are what was built.
4. Table: Wentao's `support_v1` table, copied from the as-delivered control scene `w_ctl_ep1` by `wsupport.py` after
   checking that the frame-0 object poses are identical (diff 0). His `install_support.py` pins the bundle hash, so it
   refuses a bundle whose hands were replaced (`ValueError: Bundle mismatch: result.npz`).

Independent review (3 Nov 01:2x EST): only the 8 `hand_{left,right}_wrist_*` npz keys differ from the raw re-clean,
only on [32,304) left and [45,68)+[250,274) right; fingers, betas, camera, objects, valid flags and contact labels are
unchanged; Wentao's object poses are identical (max abs 0). Known and **not** fixed: the thumb and index tips sit
3.6-4.1 cm above the 1.4 cm handle (no thumb-finger opposition), and the hand rigidly follows Wentao's pot jitter during
the lift (17 mm RMS high-frequency). Joint centres sit 5-7 mm inside the handle in 262 frames (under the 1 cm limit).

## 2. Exact inputs

| Input | Identity | Where |
|---|---|---|
| Upstream bundle | Wentao `scored20_contactshape_proxy_mix_v1_20261103`, `scenes/episode_000001/bundle` + `support_v1/support_in_bundle_world.json` + `tools/install_support.py` | HF `rose-tea-cafe/v2d-t3-recon` revision `fb11a3ce`, `experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103/` (our copy was Babel node-local `/scratch/haisenk-v2d/wentao_mix_20261103/`) |
| Raw hands | `ken_c_scored_attach/bundle_merged/result.npz`, episode 1, sha256 `a44038b7…92ea` | Babel `/data/user_data/haisenk/v2d/v2d_train/baseline/reconstruction/data/outputs/load_retarget/episode_000001/ken_c_scored_attach/bundle_merged/` |
| Loader | `tools/track3/run_load_retarget.py` at commit `5294c6b2c70c0cf88cd31e08f004f71cbc2d7f8a` (branch `haisen/hand-fixes-ep3-10-20`, pushed 2 Nov), sha256 `c3301fe7…7a31`; the hook diff is `scene_build/run_load_retarget_hooks_5294c6b.diff` | team repo; Babel source copy `~/v2d_ops/kenc/hand5/ship_git/tools/track3/run_load_retarget.py` has the same sha256 |
| Loader environment | `baseline/` and `assets/` (retarget scripts; MANO hand weights, which are licence-restricted and not part of this release) and `tools/track3/{episodes.json,support_stacked.py}` from `edm_training` @ `1a6bae7b`, venv `.venv_retarget` (py3.11) | Babel `/data/user_data/haisenk/v2d/v2d_train` (see `scene_build/setup.sh`) |
| Prepared bundle (output of step 1) | `result.npz` sha256 `115a0657…0510` + Wentao's meshes, `HANDFIX_PREP.json` | HF `experiments/haisenk/hand_fixes_wentao_v1_20261103/episode_000001/bundle/` (uploaded 3 Nov; LFS sha256 re-checked 4 Nov) and Babel `/data/user_data/haisenk/v2d/hand5_backup_20261103/hand5w/bundles/w_hold_ep1/` |
| Control scene (table source) | `w_ctl_ep1` = Wentao's ep1 bundle as delivered, support installed by his `install_support.py` (support usda sha256 `3486b2c7…`) | Babel `/data/user_data/haisenk/v2d/hand5_backup_20261103/hand5w/scenes/w_ctl_ep1` |
| Scene as built | `w_hold_ep1`, built 3 Nov 2026 00:52 EST on Babel node-local `/scratch/haisenk-v2d/hand5w/scenes/` (babel-u5-16) | HF `experiments/haisenk/hand_fixes_wentao_v1_20261103/episode_000001/scene/` and Babel `/data/.../hand5_backup_20261103/hand5w/scenes/w_hold_ep1` (+ `.log`, `.support.log`) |
| Scene as trained (final) | the same files, parquet paths rewritten by `relocate_scene.py` to `/tmp/haisenk_v2d/scenes/w_hold_ep1` | babel-s5-16 `/tmp/haisenk_v2d/scenes/w_hold_ep1`; tarball `C:\v2d_demo\archive\b_ep1_wh_K_s48\scene.tgz` (sha256 `7b385446…`) |

How the staged scripts are known to be the ones that ran:
- `patch_whold.py` sha256 `35a66f1a…` (md5 `c10d7ad6…`) equals the `/scratch` file that was exec'd (fetched right after
  the build, and in the `/data` backup of `/scratch/.../r2/ep1/`), the home copy `~/v2d_ops/kenc/hand5w/r2/ep1/`, and the
  md5 recorded by the 3 Nov review. The build log contains `== hand fixes: patch /scratch/haisenk-v2d/hand5w/r2/ep1/patch_whold.py applied`
  and `== ep1n attach: forced {'left': [[0, 32, 304]], 'right': [[1, 45, 68], [1, 250, 274]]}`; `report.json` has
  `"h5_patch"` and the `ep1n_*` keys.
- `wbuild2.sh`, `wprep.py`, `wsupport.py` equal the Babel copies `~/v2d_ops/kenc/hand5w/*` (sha256 re-checked 4 Nov
  14:19 EST); their mtimes (3 Nov 00:33-00:34) are before the build (00:52). `build_w_hold_ep1.sh` is the exact step
  script that ran the build (local `C:\v2d_demo\hand5w\r2\ep1\s5.sh`, same sha256 `5b2e5590…`).
- The control scene `w_ctl_ep1` was built at 00:03-00:31 by `build_r2.sh` with **`wbuild.sh`** (the earlier version;
  both staged). For `MODE=control` the two versions do the same thing (the diff only touches the rawhands support branch).
- Scene files: the 21 regular files of the trained copy on babel-s5-16, the orchard copy used for T2 (handoff
  `w_hold_ep1.tgz`) and the archive `scene.tgz` are identical by sha256; `report.json`, `bundle_cleaned/result.npz`,
  the support usda, the URDFs and the as-built parquet (`.parquet.orig`) equal the `/data` backup, and the LFS files
  equal the HF copy.

## 3. Rebuild the scene from its parent

The scripts hard-code the paths they ran with (section 7). Place them as below or edit the paths.

```bash
# 0. Loader tree (what setup.sh did on 3 Nov 00:02): run_load_retarget.py from commit 5294c6b + tools/track3/hand_fixes,
#    episodes.json + support_stacked.py from edm_training@1a6bae7b, baseline/ and assets/ symlinked to that checkout.
mkdir -p ~/v2d_ops/kenc/hand5w/r2/ep1 && cp scene_build/{wbuild.sh,wbuild2.sh,wprep.py,wsupport.py,build_r2.sh,setup.sh} ~/v2d_ops/kenc/hand5w/
cp scene_build/patch_whold.py ~/v2d_ops/kenc/hand5w/r2/ep1/
bash ~/v2d_ops/kenc/hand5w/setup.sh            # needs ~/v2d_ops/kenc/hand5/ship_git = files of commit 5294c6b

# 1. Wentao's release at revision fb11a3ce into /scratch/haisenk-v2d/wentao_mix_20261103 (RECONSTRUCTED command: no record
#    of the original download call was found; the path layout below is what wprep.py/wbuild*.sh expect).
hf download rose-tea-cafe/v2d-t3-recon --repo-type dataset --revision fb11a3ce \
   --include "experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103/*" --local-dir /scratch/haisenk-v2d/wentao_mix_20261103

# 2. Control scene (gives the support_v1 table): Wentao's bundle as delivered + his install_support.py.
bash ~/v2d_ops/kenc/hand5w/wbuild.sh w_ctl_ep1 1 control        # ran via build_r2.sh on 3 Nov 00:03

# 3. The fixed scene = scene_build/build_w_hold_ep1.sh (ran 3 Nov 00:52 EST):
cp ~/v2d_ops/kenc/hand5w/r2/ep1/patch_whold.py /scratch/haisenk-v2d/hand5w/r2/ep1/patch_whold.py
H5_PATCH=/scratch/haisenk-v2d/hand5w/r2/ep1/patch_whold.py bash ~/v2d_ops/kenc/hand5w/wbuild2.sh w_hold_ep1 1 rawhands
#    -> wprep.py 1 rawhands /scratch/haisenk-v2d/hand5w/bundles/w_hold_ep1          (prepared bundle, HANDFIX_PREP.json)
#    -> run_load_retarget.py --bundle <that> --episode 1 --no-rest-hold --no-object-clean --no-level --attach-hands \
#         --smooth-hands 2 --skip-video --device cpu --sequence-id ep1_hw_w_hold_ep1 --out /scratch/haisenk-v2d/hand5w/scenes/w_hold_ep1
#    -> wsupport.py <w_ctl_ep1> ep1_hw_w_ctl_ep1 <w_hold_ep1> ep1_hw_w_hold_ep1          (copies the support usda)

# 4. After moving the scene (orchard 3 Nov 18:48, then babel-s5-16 /tmp/haisenk_v2d/scenes/w_hold_ep1 on 4 Nov):
python scene_build/relocate_scene.py /tmp/haisenk_v2d/scenes/w_hold_ep1     # keeps the original parquet as .orig
```

Check: `report.json` must show `"h5_patch"`, `ep1n_forced_windows` as above, `attached_0 [["left", 32, 304, 2.9]]`,
`attached_1 [["right", 45, 68, 3.0], ["right", 250, 274, 1.6]]`, `hand_left_attached_frames 272`,
`hand_right_attached_frames 47`. Expected hashes are in `bundle_manifest.json` (scene item). A rebuild re-samples the
MANO contact labels and `object_mesh_radius` (loader behaviour noted on 2 Nov), so a rebuilt parquet may not be
byte-identical; the trained scene itself is in the bundle.
Shortcut: start from the prepared bundle on HF (`hand_fixes_wentao_v1_20261103/episode_000001/bundle/`) and run only the
loader step; or use the built scene from HF / the final bundle directly.

## 4. Training and selection of the final policy

**Training (final policy `b_ep1_wh_K_s48`).** Driver `kq.sh` (Edmond's protocol from `TRAINING_GUIDE.md` /
`campaign.py` at de66fea3; the copy on babel-s5-16 `/tmp/haisenk_v2d/kq.sh`, sha256 `9360ba9b…`, identical to the
local orchard copy `C:\v2d_demo\orchard\kq.sh`), run on Babel babel-s5-16 (H100 job 10774619), GPU 3,
4 Nov 10:21-11:10 EST (49 min, exit 0, 0 nefc-overflow lines). Queue line
(`training/kq_queue_lines_ep1.txt`):
`b_ep1_wh_K_s48 1 /tmp/haisenk_v2d/scenes/w_hold_ep1/assets/human_motion_data/ego_recon/processed/sequence_id=ep1_hw_w_hold_ep1/robot_name=sharpa_wave 48 KEN 20`.
Equivalent command (flash_chord venv from `edm_training` @ `1a6bae7b`, flash_chord source of commit `de66fea3` on
`PYTHONPATH=/tmp/haisenk_v2d/fc_src_de66`):

```bash
export PYTHONPATH=/tmp/haisenk_v2d/fc_src_de66 CUDA_VISIBLE_DEVICES=3 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline
cd $FC && $FC/.venv/bin/python scripts/train_flash_sac.py experiment=sharpa_flash_sac "task.parquet='<scene>/assets/human_motion_data/ego_recon/processed/sequence_id=ep1_hw_w_hold_ep1/robot_name=sharpa_wave'" \
  reset.seed=48 curriculum=tracking_100m_object2x training.checkpoint.save_replay=false logging.mode=offline \
  logging.upload_checkpoints=false logging.run_name=b_ep1_wh_K_s48 output_dir=<run>/train training.total_environment_steps=100003840 \
  task.motion_speed=1.0 sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 \
  training.checkpoint.save_interval_environment_steps=5000192 termination.object_position.threshold=0.1 \
  termination.object_orientation.threshold=0.5 training.replay.capacity=4194304 training.replay.observation_storage_dtype=float32 \
  sim.njmax=1024 sim.nconmax=512
```

**Checkpoint sweep** (kq.sh: kit `examples/run_policy_evaluation.py`, own-reference add_auc, every checkpoint >= 70M;
`training/sweep.json`): 100M **0.8300** [pot 0.913, lid 0.747], 95M 0.8188, 90M 0.8137, 85M 0.8040, 80M 0.7931,
75M 0.7634, 70M 0.5619. The packed rollout is the 100M evaluation (`early_100003840`).

**Seed sweep on the same scene** (12 new seeds, best checkpoint >= 70M, own-ref): s45 0.347, s46 0.218, s47 0.727 (95M),
**s48 0.830 (100M)**, s49 0.244, s50 0.804 (100M), s51 0.715, s52 0.194, s53 0.290, s54 0.328, s55 0.697, s56 0.796 (100M).
Only 3 of 12 seeds reach 0.75: the seed lottery on ep1 is large.

**Pre-registered selection rule** (written 12:12 EST, after the kq own-ref sweeps had named the candidates but before
any new rollout was scored on the emulator; `notes/ep1_hold_window/refine1104_ep1_notes.md`, section RETRY).
Baseline T2 = `o_ep1_hold_s44` @100M (`adopt_roll.parquet`). A seed replaces T2 only if, on the
registration-aware emulator of the kit's AUC.py (`selection/syn_lib.py`, truths built as in `selection/emu_boards.py`):
1. emulated AUC > T2's (strictly) in >= 13 of the 16 Kaggle-consistent cells
   (GT-pot stand-in trimmed t in {0.70, 1.00, 1.20, 1.50, 2.07} cm x truth in {ref, jb, seat}, plus t0.00_seat);
2. median emulated RPE <= T2's and median emulated MPPE <= T2's over the 16 cells;
3. MP-SR passes in all 16 cells.
Among passing seeds the highest 16-cell median AUC wins (ties at 4 dp: larger MP-SR lid margin). Sanity gate: every new
rollout's reference equals T2's reference exactly (max abs diff 0 for all compared draws).
- 12:19 EST (`selection/s1_select.py` -> `selection/results/s1_select.json`, `selection.json`): s48 @100M passes
  (16/16 cells, dAUC median +0.0377, range +0.0258..+0.0518), s50 @100M passes (+0.0212), s47 @95M fails (1/16).
- 13:22 EST final re-run with s51-s56 added (`selection/ep1_final_select.py`, identical truth/cell/mechanism code,
  reproduces the morning numbers exactly; `selection/results/ep1_final_select.json`, `ep1_final.json`): qualifiers
  s48 @100M (median 0.8869, lid margin min 0.43 cm), s56 @100M (0.8725, 0.18), s50 @100M (0.8702, 0.38) -> **s48 @100M**.
- Reported, not deciding: anchored own-ref (`selection/anch.py`): s48 0.847 vs T2 0.803; plain own-ref 0.830 vs 0.787.
- Mechanism (`s1_select.json` "mech"): s48 lifts the lid to 16.5 cm above the support (reference 16.6, T2 13.2), its
  twist push is 1.25 cm (T2 2.64) and its lid rest offset over f100-236 is 0.62 cm (T2 2.19).

**Disclosure.** The judge was calibrated on public-leaderboard feedback for T2: the 16 cells are the truth cells in
which the emulator reproduces T2's measured Kaggle MP-SR flip (t0.00 under ref and jb is excluded), and the trim range
of the stand-in GT pot agrees with a calibration on T2's measured changes. In these cells the emulator gives, for T2
minus bob1, AUC +0.030..+0.067 (measured +0.049), RPE -1.23..-1.74 cm (-1.64), MPPE -0.59..-0.92 cm (-0.90) and
MP-SR fail -> pass (measured flip). The selection itself uses only our own references, depth fits and rollouts (no GT).
s48 is the best of 12 seeds (winner's curse), and the emulator truths share our reference's errors.

### 4.1 T2 (the submitted ep1 rows)

`o_ep1_hold_s44` trained on orchard (flame-21) on the same scene with the team driver `tools/track3/jobs/gpu_queue.sh`
at `edm_training` @ `1a6bae7b` (byte-identical to that commit; *not* the de66fea3 flash_chord), seed 44, overrides
`experiment=sharpa_flash_sac curriculum=tracking_100m_object2x task.motion_speed=1.0 sim.fps=20.254119873046875
termination.wrist_position.threshold=0.3 training.checkpoint.save_interval_environment_steps=5000192 sim.njmax=1024
sim.nconmax=512 training.total_environment_steps=100003840` (other settings = experiment defaults). Packed rollout = the
old driver's evaluation of the final 100M checkpoint (own-ref 0.787 [0.900, 0.675]; same run's sweep: 90M 0.779,
95M 0.778, 100M re-evaluated 0.770).

**Packing (both).** `~/v2d_ops/pack_one.sh 1 <results dir> <scene>/score_input/meshes <code commit URL> <out>` (the
team wrapper's `relabel` from a copy of `tools/track3/pack_submission.py` at `~/v2d_ops/team_pack_submission.py`, then
the kit's `tools/pack_submission.py` with config and sample narrowed to episode 1), then chained
into bob1 with `swap_episode.py --adopt-commit-url` (`combine_final.sh` for the final local-best file). The s48 call is
in `packing/pack_args.txt`; `packing/pack.log` shows `checkpoint_sha256 6aaab5d6…` and
`code_commit_url …/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6` (bob1's commit, which does not contain these scripts).

## 5. Evidence

### 5.1 Kaggle (T2, same scene; ledger `C:\v2d_demo\tests\ledger.md`, receipts 57969833-57969844)

| Board | bob1 | bob1 + ep1 T2 | Change | ep1 change (x20) |
|---|---|---|---|---|
| AUC | 0.75166 | 0.75413 | +0.00247 | **+0.049** |
| SP-SR | 0.900 | 0.900 | 0 | unchanged |
| MP-SR | 0.450 | 0.500 | +0.05 | **fail -> pass** |
| RPE (cm) | 4.34503 | 4.26320 | -0.08183 | -1.64 cm |
| MPPE (cm) | 3.61223 | 3.56705 | -0.04518 | -0.90 cm |
| CD-O (cm) | 0.40993 | 0.41020 | +0.00027 | pot about +0.01 cm; lid payload identical to bob1's |

T2's rows are the ep1 rows of the team's combined entry, which teammate Wentao submitted on 4 Nov 03:43 EST as bob1 +
our four adopted swaps (AUC 0.76526, SP-SR 0.95, MP-SR 0.55, RPE 4.10520, MPPE 3.42999, CD-O 0.39579; its AUC equals
the sum of the four swap deltas). Attribution caveat: T2 changed the whole episode relative to bob1 (Wentao's
fb11a3ce objects incl. his pot mesh, his support table, our re-attached hands, a new seed and policy); only a model of
AUC.py's mesh registration reproduces +0.049, so part of the gain probably comes from how the pot mesh registers, not
from the hold window alone.

### 5.2 Local (GT-free)

| Comparison | Numbers | Source |
|---|---|---|
| Fix vs as-delivered control on Wentao's base (Babel, team recipe via `h5_train.sh`, final checkpoint, own-ref) | `w_hold_ep1` s42/s43 **0.770 / 0.524** vs `w_ctl_ep1` s42 **0.275** | ship_w2 README / SUMMARY.json (HF `hand_fixes_wentao_v1_20261103`) |
| Same scene on orchard (old driver) | s44 0.787 (= T2), s42 0.469 | handoff README |
| Final s48 @100M vs T2 (16 emulator cells) | AUC better in 16/16; median 0.8869 vs 0.8471 (+0.0377, range +0.026..+0.052); RPE median 2.936 vs 3.328 cm (16/16 better); MPPE 1.785 vs 2.142 cm (16/16 better); MP-SR 16/16 (lid margin min 0.43 vs 0.01 cm) | `selection/results/selection.json` |
| Noise yardstick | T2's own checkpoint re-evaluated: 0/16 cells better, median -0.012 | same |
| Own-ref | s48 0.830 [0.913, 0.747] (anchored 0.847) vs T2 0.787 [0.900, 0.675] (anchored 0.803) | `results/episode_000001.json`, `s2_anch.log` |
| Expected effect | about +0.002 on the 20-episode mean AUC if the emulator gain transfers (untested on Kaggle) | `C:\v2d_demo\diag\refine1104\wrapup\FINAL.md` |

Remaining ep1 losses (s48, t1.00_ref cell): lid phase C about 0.05 (the reference lid floats about 3.5 cm above the
support), pot D/E about 0.022, lid D/E about 0.03 (the returning lid lands 3-5 cm short of its seat in 20 of 21 returned
draws). Two follow-up fixes (lid seat `rf_ep1_ls`, post-return event hold `rf_ep1_ev`) failed their pre-registered gates
and were not used.

## 6. Bundle (data, not in git) - full list with sha256 in `bundle_manifest.json`

| Item | sha256 | Location(s) |
|---|---|---|
| Scene `w_hold_ep1` (as trained; 21 files + 6 dangling `bundle_cleaned/` symlinks) | parquet `6b1ed3b2…` (as built `31d10a3f…`) | babel-s5-16 `/tmp/haisenk_v2d/scenes/w_hold_ep1` (job 10774619, node /tmp, gone when the job ends ~6 Nov 00:03 EST); `C:\v2d_demo\archive\b_ep1_wh_K_s48\scene.tgz` (`7b385446…`) |
| Policy `b_ep1_wh_K_s48` step 100003840 | `6aaab5d64f317cf5690f89f296114b12e5aebe33121fe950b925094ce1e51152` | Babel `/data/user_data/haisenk/v2d/refine1104/ep1/adopt_s48/train/policy_100003840.safetensors` (checked 4 Nov 14:21 EST); babel-s5-16 run dir; local archive |
| Rollout `early_100003840/results/episode_000001.parquet` (+ `.json`, `hydra/`) | `b3fbfe3b…` (json `6758a34f…`) | `/data/.../adopt_s48/early_100003840/results/`; babel-s5-16 run dir; local archive |
| Packed `one.parquet` (670 rows) | `7150f4e5764d15ad30824bda8720054288a63055eb32ea4cd21a1e82529d3276` | Babel `~/v2d_ops/kenc/final1104/b_ep1_wh_K_s48/one.parquet`; local archive `babel/one.parquet` |
| train.log, recipe.txt, sweep.json, best.txt, sweep rollouts 70-95M, 95M policy (context) | see manifest | `/data/.../adopt_s48/`, babel-s5-16 run dir |
| T2 policy / rollout / one.parquet / swap-test file / receipt (submitted rows) | `969b9cef…`, `bf5e57c4…`, `27dba334…`, `d7c31578…`, `73ff20ca…` | `C:\v2d_demo\archive\o_ep1_hold_s44\`, `C:\v2d_demo\handoff\ep1_ep5_20261104\episode_000001\` |
| Parents: prepared bundle, as-built scene, control scene, raw hands | `115a0657…`, `31d10a3f…`, `3486b2c7…`, `a44038b7…` | HF `hand_fixes_wentao_v1_20261103/episode_000001/`, Babel `/data/.../hand5_backup_20261103/`, `/data/.../v2d_train/...` |
| Emulator inputs | see manifest | `C:\v2d_demo\diag\ep1_verify\synthesis\in\`, `...\refine1104\ep1\in_wh\`, `...\wrapup\ep1_final\in_wh\` |

## 7. Files in this folder

| Path | What | Provenance |
|---|---|---|
| `scene_build/patch_whold.py` | the H5_PATCH (new `attach_hands`) | exact file exec'd 3 Nov 00:52 (sha256 `35a66f1a…`) |
| `scene_build/build_w_hold_ep1.sh` | the build step that ran | = local `hand5w\r2\ep1\s5.sh` |
| `scene_build/wbuild2.sh`, `wprep.py`, `wsupport.py` | build wrapper, hand preparation, support copy | = Babel `~/v2d_ops/kenc/hand5w/` copies that ran |
| `scene_build/wbuild.sh`, `build_r2.sh` | earlier wrapper + the call that built the control `w_ctl_ep1` | = Babel copies |
| `scene_build/setup.sh` | assembles the loader tree (commit 5294c6b loader + edm_training 1a6bae7b assets) | = Babel copy |
| `scene_build/relocate_scene.py` | rewrites absolute paths after moving a scene | = copy used on orchard and babel-s5-16 |
| `scene_build/run_load_retarget_hooks_5294c6b.diff` | the 8-line opt-in hook diff (already pushed in 5294c6b) | `git show 5294c6b` |
| `diagnosis/ev.py`, `diag2.py`, `diag3.py`, `geo_b.py`, `rev.py`, `steps/s*.sh`, `outputs/*.txt` | GT-free evidence for the windows (contact labels, grip stability in the object frame, lid speed, handle geometry) and the independent review, with their 3 Nov outputs | Babel `~/v2d_ops/kenc/hand5w/r2/ep1/`, local `C:\v2d_demo\hand5w\r2\` |
| `selection/syn_lib.py` | registration-aware emulator of AUC.py + loaders | `C:\v2d_demo\diag\ep1_verify\synthesis\py\` |
| `selection/emu_boards.py`, `emu_boards_ext.py` | the status-audit emulator (16 + 2 cells) the selection copies | `C:\v2d_demo\diag\ep_status_1104\ep1_work\` |
| `selection/s1_select.py`, `ep1_final_select.py`, `make_final_json.py` | the pre-registered seed selection (morning) and the final re-run with s51-s56 | `C:\v2d_demo\diag\refine1104\` |
| `selection/anch.py`, `an_common.py` | anchored own-ref (reported, not deciding) | `C:\v2d_demo\diag\anch\`, `...\ep5_skepA1\` |
| `selection/results/*` | outputs: `selection.json`, `s1_select.{json,log}`, `s2_anch.log`, `ep1_final.json`, `ep1_final_select.{json,log}`, `emu_boards*.json` | as produced |
| `training/*` | kq queue lines (s45-s56), driver log lines for s48, `recipe.txt`, `sweep.json`, `best.txt` | Babel `/tmp/haisenk_v2d/kq_babel.txt`, `~/v2d_ops/logs/kq_babel.out`, run dir |
| `packing/pack_args.txt`, `pack.log` | the pack call for s48 and its output | Babel `~/v2d_ops/kenc/final1104/b_ep1_wh_K_s48/` |
| `bundle_manifest.json` | every data item with location and sha256 | written for this release |
| `SHA256SUMS` | hashes of the files in this folder | written for this release |

Notes staged next to the code (release `notes/ep1_hold_window/`): `refine1104_ep1_notes.md` (4 Nov refine record with the
pre-registered rule, the failed follow-up fixes and the remaining-loss analysis), `ep1_status_audit_1104.md` (4 Nov
status audit: Kaggle T2 table, emulator calibration), `r2_ep1_diagnosis_and_review_20261103.json` (3 Nov diagnosis,
build record and independent review, verbatim from `C:\v2d_demo\hand5w\r2_result.json`).

Common tools used but staged elsewhere or already public: `kq.sh` (training driver), `pack_one.sh`,
`swap_episode.py`, `combine_final.sh`, team `tools/track3/pack_submission.py`, kit `tools/pack_submission.py` and
`examples/run_policy_evaluation.py`, Wentao's `install_support.py`, the team loader (`run_load_retarget.py` @5294c6b).

**Hard-coded personal absolute paths (kept as they ran):** `scene_build/*.sh` and `wprep.py` use
`/scratch/haisenk-v2d/...` (Babel node-local scratch), `/data/user_data/haisenk/v2d/...`, `$HOME/v2d_ops/kenc/hand5w`,
`~/v2d_ops/kenc/hand5/ship_git`; `diagnosis/diag3.py` and `diagnosis/steps/*.sh` use `/scratch/...` and `/data/...`;
`selection/*.py` use `C:\v2d_demo\diag\...` (Windows) for their inputs and outputs (`syn_lib.py` reads `../in` next to
its own folder); `selection/an_common.py` globs an ep5 scene under `C:\v2d_demo\diag\ep5_objA\data` **at import time**,
so `anch.py` fails on another machine until that path exists or is edited; `make_final_json.py` also reads
`d6_potevent.log` files that are not staged (they are in Babel `/data/user_data/haisenk/v2d/refine1104/ep1/retry_1104/`).

## 8. Known gaps (honest list)

- The final choice s48 was never scored by Kaggle. Its advantage over T2 rests on the GT-free emulator (section 4).
- The scene was built on node-local Babel scratch (babel-u5-16), which is no longer reachable. The build is reconstructed
  from the scripts, the build log, `report.json` and the `/data` backup, all consistent; the original download command for
  Wentao's release was not recorded (step 1 in section 3 is reconstructed).
- The gain of the fix over the delivered control mixes two changes (raw hands re-attached to his objects, and the forced
  windows / median grip). The raw-hands-only build `w_raw_ep1` exists but, as far as the records show, was never trained.
- T2's Kaggle gain is not attributable to the hold window alone (section 5.1).
- The emulator needs the organisers' metric code (kit `metric_code/track_3/AUC.py`, copied as `in/AUC_kit.py`,
  md5 `93a20d94…`), which is not redistributed here.
- The packed files carry `code_commit_url` = bob1's commit `1689cf63`, which does not contain this folder.
- A rebuild re-samples contact labels and `object_mesh_radius`, so the processed parquet may differ slightly from the
  trained one; use the bundled scene for exact reproduction.
