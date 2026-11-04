# Episode 3: support at the stereo-depth table + pot centre of mass

**Final ep3 entry:** scene `sc_ep3`, policy `h_ep3_sc_K_s42` @ 95,006,720 env steps.

- **Episode:** scored episode 3 (`white_pot` + `wooden_spoon`), sequence `ep3_h5_h5_ep3_A1`, robot `sharpa_wave`, 313 frames. The pot is object slot 0, so AUC.py uses it as the frame-0 anchor.
- **Kaggle test T5:** the AUC board scored 0.75305 against the bob1 baseline's 0.75166. That is +0.00139 on the board, or **+0.028 on episode 3** (x20). It was submission 57974215, made 4 Nov 2026 01:29 EST, on the AUC board only.
- **Final local judge (4 Nov 13:45 EST):** kept this entry. The later alternative that was trained, the spoon-track scene `rf_ep3_f3w`, failed its pre-registered judge. Notes are in `../../../../../notes/ep3_support_com/` in the staging tree.

## 1. What the fix changes and why

`sc_ep3` makes two physics-setup edits on a copy of bob1's ep3 scene. Object poses, hand motion, contact labels and every mesh stay byte-identical (checked, see section 3).

1. **Support raised.** The table proxy `support_0` (a horizontal cylinder, radius 0.298 m, 1 cm high) goes from a top at **0.9878 m to 0.9990 m (+1.12 cm)**.
2. **Pot centre of mass moved.** The `white_pot` URDF `<inertial><origin>`, which is the centre of mass the simulator uses, moves from the mesh origin `0 0 0` to the area-weighted shell centroid of the pot's own visual mesh, **`-0.008451 -0.062887 0.049392` m** (8.0 cm). Mass (0.3 kg) and inertia are unchanged. The spoon is untouched.

### Confirmed diagnosis

The overnight diagnosis (workflow wf_fac9fc07-8e2) produced each finding, and an independent skeptic re-checked it. See `diagnosis/ep3_diagnosis_B1_B2.json`.

**Support too low (ep3-B2).** The scene builder placed `support_0` at the lowest still-frame vertex. That vertex belongs to the pot's end-rest pose, which is itself wrong: the pose is sunk 1-2.6 cm into the real table and tilted 4.4 deg.
- Our FoundationStereo depth puts the real table **+1.87 cm (pot) and +1.96 cm (spoon) above `support_0`** at the start spots, with a 2.4 deg tilt and 2.4 mm rms. Both objects' start-rest bottoms sit on that table.
- In the sim, both objects therefore dropped 14.35 mm at step 0.
- AUC.py anchors the whole scene on the pot's frame-0 pose, so every lifted pose shifted +1.435 cm against the reference.
- The skeptic's verdict ("partly") corrected the finding: the objects do not float; the support is too low.

**Pot centre of mass outside its base (ep3-B1).** flash_chord imports the URDF inertial block (`add_urdf(..., ignore_inertial_definitions=False)`), so the sim centre of mass is the mesh origin.
- That origin is 8.04 cm from the shell centroid, towards the handle. With the pot flat, it lies about 3.3 cm outside the base footprint.
- A no-hands MuJoCo settle tips the pot to 23 deg within 0.25 s with the centre of mass at the origin. With it at the shell centroid, the pot stays at 0.3 deg.
- 25 of 28 earlier ep3 rollouts ended with the pot tipped.
- The same pot mesh and the same rule are in the adopted ep5 scene (`w7_attcom_ep5`).

### Support height rule (no tuned numbers)

`top = min(depth-measured table under the resting objects at frame 0, lowest frame-0 resting bottom - 1 mm)`

Applied to ep3:
- The depth table under the objects at f0 is 1.0065-1.0081: pot 1.0065 local / 1.0071 global, spoon 1.0074 / 1.0081 (`depth_table/out/run.log`).
- The pot's frame-0 mesh bottom is 1.0000 (`scene_build/records/build_outputs_orchard.txt`).
- So the cap applies: **0.9990**.

Raising the support to the full depth-table height would push the pot up at reset, because its f0 pose is tilted 2.3 deg against the real table, and the anchor would then shift the other way. The skeptic's corrected fix states this rule ("clipped to the lowest f0 resting bottom ... never above it"). The 1 mm margin is inferred from the recorded numbers, which are consistent across the three scenes built that night (ep1 1.0207 -> 1.0197, ep3 1.0000 -> 0.9990, ep8 cup disk 0.972 -> 0.971). No document states the margin itself.

**Effect on the reference** (GT-free render review, `checks/compare_review_numbers.out`):
- Silhouette IoU against our SAM 3 masks is unchanged: pot 0.821 (f0-20 0.945), spoon 0.587 (0.712).
- Median rest height above the support fell: pot **+1.22 -> +0.10 cm**, spoon **+1.95 -> +0.83 cm**.

## 2. Exact inputs

| | name | where | identity |
|---|---|---|---|
| parent scene | `b1_ep3` = bob1's ep3 scene = our hand-fix v1 build `h5_ep3_A1` (2 Nov 2026) | `babel-s5-16:/tmp/haisenk_v2d/scenes/b1_ep3` (moved from orchard-flame-21, same path); original `Babel:/data/user_data/haisenk/v2d/hand5/scenes/h5_ep3_A1`; HF `rose-tea-cafe/v2d-t3-recon` `experiments/haisenk/hand_fixes_v1_20261102/episode_000003/scene` | the Babel copies are md5-identical, except that `b1_ep3`'s parquet has relocated path strings; support USD md5 `0ee9f14e7dfbed38c55480f93a964a2c` |
| intermediate | `sf_ep3` (support only) | `babel-s5-16:/tmp/haisenk_v2d/scenes/sf_ep3`; tarball `C:\v2d_demo\cupfix\sf_scenes.tgz` | support USD md5 `bc823a849edfb0339d7ec8730973ec9f` |
| **final scene** | **`sc_ep3`** | `babel-s5-16:/tmp/haisenk_v2d/scenes/sc_ep3` = `C:\v2d_demo\archive\h_ep3_sc_K_s42\scene.tgz` (all 31 non-PNG files md5-identical) | tarball sha256 `154922eb3c784d5f3253c6692bd370c90c3b4a59c13a03425131693f666b7978`; `white_pot_rigid.urdf` md5 `b1577693c7a1fff05442dc2c06149a8f` |
| depth input | our `ken_c_scored` reconstruction of episode 3: FoundationStereo depth, SAM 3 object and hand masks | `Babel:/data/user_data/haisenk/v2d/v2d_recon/baseline/reconstruction/data/outputs/ken_c_scored/episode_000003` | 313 depth frames |

The parent comes from our 2 Nov hand-fix v1 release: GitHub personal branch `haisen/hand-fixes-ep3-10-20` @ 5294c6b (the H5 hooks in `run_load_retarget.py` plus `tools/track3/hand_fixes/`), and HF `experiments/haisenk/hand_fixes_v1_20261102/`. bob1 adopted it for ep3. This record of the parent's origin is taken from our 2 Nov push/upload notes; the HF copy was not re-downloaded here. The Babel copies were md5-checked.

## 3. Rebuild the scene from its parent (`scene_build/`)

```bash
R=<this folder>; PY=<flash_chord venv python (numpy, scipy, trimesh, pyarrow)>; export RELOCATE=$R/scene_build/relocate_scene.py
cd /tmp/haisenk_v2d/scenes                                                    # holds the parent b1_ep3
$PY $R/scene_build/restspots.py b1_ep3 0,30,280,312                           # measurement: f0 bottoms (pot 1.0000)
$PY $R/scene_build/scenefix.py b1_ep3 sf_ep3 --support-top support_0:0.9990   # was run 00:12 EST 4 Nov
$PY $R/scene_build/comfix.py fix sf_ep3 sc_ep3 white_pot                      # was run 00:20 EST 4 Nov
```

`scene_build/build_sc_ep3.sh` is the same sequence as a script, with checks and expected md5s.

- **What the tools touch.** `scenefix.py` edits only the support USD and `comfix.py` only the pot URDF. Both copy the input scene first, refuse to overwrite an existing output, and finish by running `relocate_scene.py`, which repoints the absolute paths in the parquet, URDF, USD and MTL files at the new copy.
- **Recorded outputs.** `scene_build/records/build_outputs_orchard.txt` holds the verbatim orchard outputs: `top_before 0.9878 -> top_after 0.999`, `white_pot: inertial origin [0. 0. 0.] -> [-0.008451 -0.062887 0.049392]`.

### Exact script versions, and how we know

- **`scene_build/scenefix.py`** (md5 `7f19b50d0f7c63441187b63cb45a6ce6`, 7,901 B) is the exact version that built `sf_ep3`. The session log shows:
  - the file written at 05:10:54Z;
  - piped to orchard and run on `b1_ep3` at 05:12:13Z, with no change in between;
  - first edited at 05:12:52Z, to handle symlinks for ep1's scene;
  - edited again at 06:49Z (USD regex for any indentation).

  The deployed copy (`C:\v2d_demo\cupfix\scenefix.py` = `babel-s5-16:/tmp/haisenk_v2d/tools/scenefix.py`, md5 `d3ca0eeb...`) is that later version. Its three-line diff is in `scene_build/scenefix_later_edits.diff`. Neither edit changes the ep3 result: both versions give the same support USD (checked).
- **`scene_build/comfix.py`** (md5 `2aa37dac3570d0aeaa3ec5b95e7ef0a4`) was written at 05:19:10Z and used at 05:20:00Z. It is identical to the copy on orchard and Babel and was never edited afterwards.
- **`scene_build/relocate_scene.py`** (md5 `d8319c83...`) is byte-identical to the `~/v2d_orchard/relocate_scene.py` that both tools called by default.
- **Rebuild check** (`checks/rebuild_verification/`). These scripts with these arguments, applied to the parent, reproduce the archived `sc_ep3` file by file. The only differences are local path strings in the parquet path columns and in `scenefix_report.json`.
- **Unchanged motion.** The scene parquet equals bob1's original (`*.parquet.orig`) in all 63 non-path columns.

## 4. The depth-table measurement that set the height (`depth_table/`)

`depthtab.py` and `followup.py` ran on Babel on 3-4 Nov, 23:51-00:00 EST, in workflow wf_ed1d2237-bda. The commands are in `depth_table/run_depthtab_ep3.sh`.

Inputs:
- bob1's ep3 scene, identical to the parent above;
- our FoundationStereo depth (`depth_m = 65535/raw - 1`);
- our SAM 3 object and hand masks;
- the bundle camera mapped into the sim world.

Method:
- Table pixels are taken outside the object masks and silhouettes and outside the dilated hand masks, then back-projected.
- A RANSAC plane is fitted to 990k points over 33 rest frames.
- Local planes are fitted on 2-12 cm rings around each object.
- `followup.py` adds per-vertex clearance and tilt.

Result (`depth_table/out/`, plus the structured summary in `depth_table/depthtab_ep3_workflow_result.json`):
- **Global plane:** z = 0.00292x + 0.04231y + 1.00499, tilted 2.43 deg, rms 2.4 mm, 91% inliers.
- **Start rests:** table minus support is +1.87 cm under the pot (f0-48) and +1.96 cm under the spoon (f0-36).
- **Object depth:** agrees with our meshes within 4-8 mm, so no object needs to move along the camera ray.
- **Root cause:** the support height 0.9878 equals the end-rest pot bottoms (median 0.9894), which is the wrong pose.

`depthtab.py` imports `review_tool/render_review.py` (md5 `8050e217...`), our render/review tool, unchanged since 3 Nov 22:30 EST, before this run. The raw table points `ep3_table_points.npz` (57 MB) are listed in the bundle (section 9), not stored here.

## 5. Training and checkpoint selection (`training/`)

The policy was trained with Edmond's protocol, run by `training/kq.sh` (md5 `9d5a6253...`; the orchard copy was last written 3 Nov 22:33 and never changed).

Code and recipe:
- flash_chord code: de66fea3 via `PYTHONPATH=/tmp/haisenk_v2d/fc_src_de66`, on the team repo `edm_training` @ 1a6bae7b.
- Recipe **KEN**: object position / orientation termination 0.1 m / 0.5 rad, replay 4,194,304 float32, njmax 1024, plus `task.motion_speed=1.0 sim.fps=20.254... termination.wrist_position.threshold=0.3`.
- Seed 42, curriculum `tracking_100m_object2x`, 100,003,840 env steps, checkpoints every 5,000,192.

Run:
- The queue line, the expanded command and the environment are in `training/train_command.txt`.
- It ran on orchard-flame-21 GPU 1 from 00:48 to 01:21 EST on 4 Nov, uninterrupted. Exit 0, 0 nefc overflows.

Selection:
- kq.sh evaluates every checkpoint from 70M with the kit's `examples/run_policy_evaluation.py`, using our own reference, and keeps the best `add_auc` (`training/sweep.json`).
- Best checkpoint: **95,006,720: 0.9468 [pot 0.988, spoon 0.905]**. Others: 80M 0.945, 90M 0.937, 85M 0.937, 70M 0.936, 75M 0.930; 100M collapsed to 0.783.
- Policy `policy_95006720.safetensors`, sha256 `45c4736b1feab1e88189530c4cccbf672f19d6d8d50a7bfca4e321ed4828b531`. Rollout `episode_000003.parquet`, sha256 `77a60882...`.
- Evaluator output and hydra config are in `training/eval_95006720/`.

Other seeds on `sc_ep3` (`training/kq_driver_log_ep3.txt`):

| run | own-ref (best checkpoint) |
|---|---|
| KEN s43 | 0.952 @100M |
| KEN s44 | 0.955 @80M |
| KEN s45 | 0.793 (pot failure) |
| WIDE s42 | 0.772 |

s42 stayed the pick because it is the run measured on Kaggle. For comparison, seeds on the unfixed bob1 scene scored 0.574-0.914, and bob1's own ep3 run scored 0.914.

## 6. Packing (`packing/`)

The verbatim commands and outputs are in `packing/t5_commands.txt`. The steps were:

1. `ship.sh` copied the rollout and the scene's `score_input/meshes` to Babel. These are bob1's meshes, unchanged.
2. `swap_submit.sh` ran:
   - `pack_one.sh`, which narrowed the kit config and sample to episode 3, relabelled the bodies to the roster names with the team packer's `relabel()` (`team_pack_submission.py`), and ran the kit's `tools/pack_submission.py` with commit URL `.../commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6`;
   - `swap_episode.py`, which spliced the 626 ep3 rows into bob1's file and left the other 8,494 rows byte-identical;
   - a kit dry run, then a submission with `--only add_auc`.
3. `archive.sh` archived everything to `C:\v2d_demo\archive\h_ep3_sc_K_s42\`.

Result:
- `one.parquet`: 626 rows, sha256 `4a3d974d5ebce4ce529174b4f2828b7fe4789c4c250fdcd2d423bac6002cc98d`. Its `checkpoint_sha256` is `45c4736b...`.
- These rows were reused unchanged in the team's combined entry and in `final1104/local_best`.
- **Commit URL.** The rows carry bob1's code commit URL (`1689cf63...`). Every Kaggle file must have uniform provenance columns, so `pack_one.sh` was given bob1's URL; `swap_episode.py --adopt-commit-url` enforces the same value. That commit is the one bob1's file cites. It does not contain the scene edits above, which is what this folder adds.

## 7. Evidence

**Kaggle** (`C:\v2d_demo\tests\ledger.md`; receipt `packing/records/kaggle_receipt_T5_final.json`):

| entry | AUC | vs bob1 0.75166 | ep3 effect |
|---|---|---|---|
| T5 = bob1 with only ep3 replaced by this entry | **0.75305** | +0.00139 | **+0.028** |
| combined team entry: bob1 + ep1 T2 + **ep3 T5** + ep5 T8 + ep8 T6 (a teammate's row-level merge of our four swap files) | 0.76526 | +0.0136 | predicted 0.76524 from the four single swaps (they add exactly) |

- **Combined entry, other boards:** SP-SR 0.950, MP-SR 0.550, RPE 4.10520, MPPE 3.42999, CD-O 0.39579.
- **T5 alone:** measured only on the AUC board, never alone on the others. The ep3 status audit (`notes/ep3_support_com/ep3_status_audit_1104.md`) separates ep3+ep8 together from the combined entries: SP-SR 0, MP-SR 0, RPE -0.092, MPPE -0.033 on the boards. That is derived, not measured.
- **CD-O:** T5 changes no mesh. The later CD-O-only entry C2 swapped ep3's `white_pot` mesh rows for ep5's T8 pot; that belongs to the ep5/CD-O records, not to this part.

**Local, GT-free**, with our own reference unless marked:

| | bob1 ep3 (`ep3_h5A1_s42`) | **this entry** |
|---|---|---|
| own-ref add_auc (official evaluator) | 0.914 [pot 0.983, spoon 0.845] | **0.947 [0.988, 0.905]** |
| anchored own-ref (AUC.py-style pot f0 alignment) | 0.915 | **0.948** |
| anchor at reset | whole scene 1.435 cm low (14.35 mm drop) | 0.20 deg, 0.03 cm |
| own-ref pot / spoon position error | 0.76 / 2.16 cm | 0.55 / 1.57 cm |
| own-ref chord_sr / maniptrans_sr | - | 1 / 1 |

- **4 Nov 13:45 judge** (`notes/ep3_support_com/ep3_final_decision_1104.json`):
  - The verbatim kit AUC.py on this entry's rollout, under four GT-free pseudo-truths: 0.948 (OC, our adopted `sc_ep3` reference), 0.917 (HYBW, a hybrid with Wentao's spoon track in the stir windows), 0.918 (F3W) and 0.912 (W, Wentao's spoon track throughout). SP-SR and MP-SR were 1 under all four.
  - The challenger `rf_ep3_f3w` (Wentao's spoon centre and axis in the stir windows) failed: its best seed scored OC 0.9410 < 0.9477, and its family stir-depth residual was 14.06 mm against a 13.30 mm limit.
  - **Decision: KEEP T5.**

## 8. Not done, caveats, honest limits

- **Pot end rest not lifted.** Plan F2's lift of the pot's end-rest frames was not applied. In f259-312 the reference pot still sits 0.8-1.5 cm below the raised support and tilted 4.3-4.9 deg. The simulated pot ends flat on the support, so our own reference penalises the policy there.
- **Support kept horizontal.** The real table is tilted 2.4 deg, so the support is off by up to about ±0.75 cm at its 30 cm rim. ep1 and ep5 got a world re-level; ep3 did not.
- **Spoon centre of mass** was left at the mesh origin (the second half of plan F1).
- **Remaining loss.** About 0.033 of ep3's own-reference loss in the stir windows (f85-215) remains: our spoon FoundationPose track is about 12.5 mm off in depth. The fix tried for it, `rf_ep3_f3w`, failed its judge.
- **Recorded margins.** The `margin` values printed by `comfix.py check` are not signed distances; read only the inside/outside flags. The 3.3 cm outside figure comes from the diagnosis, measured with the pot flat.
- **Receipt file.** `packing/records/kaggle_receipt_T5_final.json` was transcribed from `cat` of the Babel receipt, which shows state COMPLETE and score 0.75305. The local archive copy (`C:\v2d_demo\archive\h_ep3_sc_K_s42\babel\`) was taken before polling and still says PENDING.
- **`swap_submit.sh` version.** The copy here is the current one. At T5 time its submission guard read `-le 8`; it was changed to `-le 3` at 01:38 EST. Only the guard differs.
- **Rebuild check scope.** The check rebuilt the parent from `sf_ep3` plus the original support USD, not from a full download of `b1_ep3`. File-by-file md5s on Babel show this is equivalent.

## 9. Bundle items (large data, not in this folder; upload targets for `final_ours_20261104/`)

| item | location | sha256 | size |
|---|---|---|---|
| scene `sc_ep3` (directory) | `babel-s5-16:/tmp/haisenk_v2d/scenes/sc_ep3` (also tarred: `C:\v2d_demo\archive\h_ep3_sc_K_s42\scene.tgz`) | tgz `154922eb3c784d5f3253c6692bd370c90c3b4a59c13a03425131693f666b7978` | 16,159,487 B tgz |
| policy `h_ep3_sc_K_s42` @ 95006720 | `C:\v2d_demo\archive\h_ep3_sc_K_s42\policy_95006720.safetensors` (= `babel-s5-16:/tmp/haisenk_v2d/runs/h_ep3_sc_K_s42/train/policy_95006720.safetensors`) | `45c4736b1feab1e88189530c4cccbf672f19d6d8d50a7bfca4e321ed4828b531` | 5,138,784 B |
| packed ep3 rows `one.parquet` | `Babel:~/v2d_ops/kenc/tests/h_ep3_sc_K_s42/one.parquet` (= archive `babel/one.parquet`) | `4a3d974d5ebce4ce529174b4f2828b7fe4789c4c250fdcd2d423bac6002cc98d` | 171,792 B |
| rollout (evaluator output) | `C:\v2d_demo\archive\h_ep3_sc_K_s42\results\episode_000003.parquet` + `.json` | `77a60882e148f23a95bf43a1959c70c81b00296cc1b323863cc680d68e78a62f` / `0a016df9aa85f280d75e46d711637970b3bf1b0b01af4fd8446e4768b838caf7` | 90,869 B / 2,351 B |
| T5 full file (evidence) | `Babel:~/v2d_ops/kenc/tests/h_ep3_sc_K_s42/track3_submission.parquet` | `7fd980f052e89e71b181aa1ca88fedb691ec79fe8f665c817c9df7b8818e3c97` | 2,589,569 B |
| optional: depth-table points | `Babel:~/v2d_ops/kenc/depthtab/ep3/out/ep3_table_points.npz` | `83b563440a769fe70c6bfea6e5f9543c26c97986e95d34ffca3b02ed72562f3e` | 57,097,157 B |

## 10. Files in this folder

- **`scene_build/`**: `scenefix.py` (exact version), `comfix.py`, `relocate_scene.py`, `restspots.py`, `build_sc_ep3.sh`, `scenefix_later_edits.diff`. The `records/` subfolder holds the support USD before and after, the pot URDF before and after, `scenefix_report.json` and the verbatim orchard outputs.
- **`depth_table/`**: `depthtab.py`, `followup.py`, `run_depthtab_ep3.sh`, `review_tool/render_review.py`, `out/` (JSON and logs), `depthtab_ep3_workflow_result.json`.
- **`diagnosis/`**:
  - `ep3_diagnosis_B1_B2.json`: findings, skeptic verdicts and plan F1/F2.
  - `b1_pot_com/`: `com_tip.py` and its output; pot-tip census; the skeptic's MuJoCo settle (`ep3com_phys.py`, `phys_bob1.*`) and re-census (`ep3com_census.py`, `census.*`).
  - `b2_support/`: the skeptic's own depth-table check `dtchk.*`, the AUC.py anchor counterfactual `skb2.*` and the support-at-table counterfactual `skb2b.*`.
- **`checks/`**: the render-review logs and numbers for bob1 and the support-fixed scene, `compare_review_numbers.py` and its output, and `rebuild_verification/`, added during release staging.
- **`training/`**: `kq.sh`, `train_command.txt`, `kq_driver_log_ep3.txt`, `recipe.txt`, `sweep.json`, `best.txt`, `manifest.json`, `train.log`, `eval_95006720/`.
- **`packing/`**: `ship.sh`, `swap_submit.sh`, `pack_one.sh`, `swap_episode.py`, `team_pack_submission.py` (a copy of the team packer as loaded by `pack_one.sh`, md5 `c31634cb...`), `archive.sh`, `t5_commands.txt`, and `records/` with the pack, dry-run and submit logs and the receipt.
- **`SHA256SUMS.txt`**: hashes of every file here.

### Hard-coded personal paths (kept as run; adjust before re-use)

- `/tmp/haisenk_v2d/...`, the orchard node layout, now on babel-s5-16: `kq.sh`, `ship.sh`, `archive.sh`, `build_sc_ep3.sh`.
- `~/v2d_orchard/relocate_scene.py`, the `RELOCATE` default in `scenefix.py` and `comfix.py`. Override it with `RELOCATE=...`.
- `~/v2d_ops/kenc/review/tool`, the `sys.path` entry in `depthtab.py`, `followup.py` and `dtchk.py`.
- `/data/user_data/haisenk/v2d/...`, Babel: `pack_one.sh` (kit), `swap_submit.sh`, `render_review.py`, `ep3com_phys.py`, `dtchk.py`, `skb2.py`, `run_depthtab_ep3.sh`.
- `/home/haisenk/v2d_ops/...`: the `kaggle_env` python and `team_pack_submission.py` in `pack_one.sh` and `swap_submit.sh`; `/home/haisenk/anaconda3/bin/python` in `run_depthtab_ep3.sh` and `render_review.py`.
- `/scratch/haisenk-v2d/...`, the Babel node scratch copy of bob1's scene: `run_depthtab_ep3.sh`.
- Local Windows paths `C:\v2d_demo\...` / `/c/v2d_demo/...`: `archive.sh`.
- Host aliases `orchard` and `babel`, and Slurm job ids 157612, 11035728 and 10774619: `ship.sh`, `archive.sh` and the recorded commands.
