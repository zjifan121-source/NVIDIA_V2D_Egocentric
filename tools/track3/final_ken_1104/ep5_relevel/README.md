# Episode 5: re-level to the depth-measured table, drop the partial spoon disk, trim the pot 1.0 cm (scene `rl2_ep5`, run T8)

These are the scored episode-5 rows of team Gravity Aligned for the NVIDIA V2D Challenge Track 3. The final
candidate is **scene `rl2_ep5`, policy `r_ep5_rl_K_s43` at 95,006,720 steps**. It is Kaggle swap test T8.

The same ep5 rows are in:

- the team's combined entry (AUC 0.76526) and the later entries built on it;
- our final local-best file (`final1104`, not submitted).

## 1. What the fix changes and why

**Diagnosis.** It was confirmed with GT-free evidence only: our stereo depth and our rollouts.

1. **The world's gravity is 5.5 deg off the real table.** We fitted the table from our own FoundationStereo depth
   (every 2nd frame of 0-24, 260k points, RANSAC then least squares, residual 1.6 mm). It is tilted **5.48 deg**
   against the scene's +z. The floor is tilted 6.08 deg, and table and floor differ by only 1.07 deg. So the error is
   in the world's gravity estimate (GeoCalib), not in the table.
   - The simulator's gravity points along -z, so at reset it re-seats the resting anchor pot. In the adopted T3
     rollout the pot is 3.0 deg off at frame 0 and 5.2 deg off at frame 10 (`diagnosis/outputs/anchorseat_*.txt`).
   - The Kaggle scorer aligns the whole scene on the anchor's frame-0 pose, so that re-seat rotates every object.
2. **The spoon tips off a partial support.** In the parent scene the spoon rests on `support_1`, a disk of radius
   9.1 cm that lies under the handle half only. It came with the copied v2 control support.
   - In the simulator the spoon tips 9 deg by frame 1 and 14 deg by frame 2, and stays tipped until the pickup (mean
     13.5 deg over f0-30).
   - This costs about 0.022 of spoon AUC in f0-35 in every policy trained on that scene (diagnosis finding
     `ep5-B1`).

**Fix** (`scene_build/`):

1. **Re-level** (`relevel.py`). The whole world is rotated 5.4786 deg about the measured table point under the pot's
   frame-0 footprint, so that the measured table normal becomes +z.
   - It rotates every world-frame field of the processed parquet: object poses, MANO joints/trans/orient, contacts,
     and robot wrists and frames.
   - Relative poses change by at most 9e-8 m.
2. **Drop the partial disk.** `support_1` is removed.
3. **Trim the pot.** The pot mesh is shortened by 1.0 cm with the rim fixed, from 13.34 to 12.34 cm along the frame-0
   vertical (`scenefix.py`).
   - The pot's frame-0 bottom (z 1.0007) becomes coplanar with the spoon's (1.0107).
   - The single table support goes 1 mm under both: top 1.0097.

Effect on the rollout:

- The anchor's frame-0 re-seat drops from 3.05 to 1.21 deg. It is reduced, not removed.
- The spoon's resting error over f0-35 drops from 2.10 to 0.60 cm mean ADD.

About the 1.0 cm trim: it comes from the coplanar-rest rule. It was not fitted to depth. The depth evidence points the
same way, toward a pot bottom that is too deep:

- point-to-plane ICP of the real depth points on the pot lifts its bottom 1.4 cm (`depth_table/out/run_dt5b.out`);
- mesh-vs-depth residuals: inner bottom -2.1 cm, handle +0.05 cm (`C:\v2d_demo\diag\ep5_skepA1\run_sil5b.out`, quoted
  in `notes/ep5_relevel/ep5_status_audit_1104.md`).

**Parent (T3 layer, built 3 Nov, `scene_build/parent_w7_attcom_ep5/`).** The parent scene `w7_attcom_ep5` is Wentao's
`scored20_lead_sprint_v2_20261103` ep5 bundle with two changes; its objects are unchanged:

1. **Hands.** Our raw hands are re-expressed in his world (`w7prep.py rawhands`), then cleaned and attached to *his*
   new spoon track by the team loader. His hands were attached to our old spoon track, so the spoon in the wrist frame
   was off by a median of 21.6 deg.
2. **Pot COM.** The pot URDF centre of mass moves 8.09 cm to the pot's shell centroid (`w7edit.py --com`). With the COM
   at the mesh origin, the pot tipped about 25 deg onto its handle at rest.

The details are in `scene_build/parent_w7_attcom_ep5/README_ep1_ep5_handoff.md`.

## 2. Inputs

| what | name | where | sha256 |
|---|---|---|---|
| reconstruction bundle under the parent | Wentao `scored20_lead_sprint_v2_20261103`, `scenes/episode_000005/bundle` | HF `rose-tea-cafe/v2d-t3-recon` `experiments/wentacc/...`; Babel `/scratch/haisenk-v2d/wentao_v2/...` | (see the handoff README) |
| our raw hands | `ken_c_scored_attach/bundle_merged/result.npz`, ep5 | Babel `/data/user_data/haisenk/v2d/v2d_train/baseline/reconstruction/data/outputs/load_retarget/episode_000005/` | not recorded here |
| parent scene | `w7_attcom_ep5` (sequence id `ep5_hw_w7_attcom_ep5`) | bundle item `w7_attcom_ep5.tgz` = handoff `episode_000005/scene/w7_attcom_ep5.tgz` = babel-s5-16 `/tmp/haisenk_v2d/archive/o_ep5_attcom_s42/scene.tgz` | tgz `ae61a76f…`; processed parquet `da451d2a…` |
| table plane | `depth_table/out/ep5_depthB.json` → `planes_start.table` | measured by `depth_table/dt5b.py` from our FoundationStereo depth (`ken_c_scored/episode_000005`) | `2b6225d9…` |
| final scene | `rl2_ep5` | bundle item `scene.tgz`; see `bundle_manifest.json` | tgz `c0159b0d…`; parquet `f3badc29…` |

Plane, in the parent's sim world: unit normal `(-0.02000265, -0.09335555, 0.99543188)` and
`z = 0.020094445 x + 0.093783968 y + 1.015406946`.

## 3. Rebuild

### Parent from Wentao's bundle

This ran on Babel, 3 Nov 18:32-18:49 EST:

1. `bash parent_w7_attcom_ep5/w7build.sh`. It makes `w7_com_ep5` (control arm) and `w7_attcom_ep5`:
   - `w7prep.py 5 rawhands`
   - `run_load_retarget.py --no-rest-hold --no-object-clean --no-level --attach-hands --smooth-hands 2`

   For the rawhands bundle, Wentao's `install_support.py` refused with "Bundle mismatch", so the build stopped before
   the support and COM steps.
2. `bash parent_w7_attcom_ep5/w7fix.sh` completes it:
   - `wsupport.py` copies the support USD of the control scene `wv2_ctl_ep5`, after checking that frame-0 object poses
     are equal;
   - `w7edit.py --com` moves the pot COM.
3. `parent_w7_attcom_ep5/add_ep5_orchard.sh` (3 Nov 19:15 EST) extracted the scene on orchard and repointed it with
   `relocate_scene.py`. That copy is the one the re-level ran on: its processed parquet has sha256 `da451d2a…`, the same
   in the bundle tgz and on babel-s5-16.

The loader is the team's `tools/track3/run_load_retarget.py` (handoff: commit 5294c6b, no hook set for ep5).

### `rl2_ep5` from the parent

This is the verified two-command form:

```
bash scene_build/build_rl2_ep5.sh <SCENES_DIR containing w7_attcom_ep5> [python]
# = relevel.py w7_attcom_ep5 rl_ep5 -0.02000265 -0.09335555 0.99543188 0.020094445 0.093783968 1.015406946 white_pot --drop-other-supports
#   scenefix.py rl_ep5 rl2_ep5 --obj white_pot:-1.00 --support-top support_0:1.0097
```

What actually ran on orchard, 4 Nov 01:47-01:49 EST, is in `scene_build/history/build_rl2_ep5_asrun.md`. It was three
steps:

1. `relevel.py`, as above.
2. The older `scenefix.py`. It rewrote the pot meshes, then crashed on the support step with a USD indentation regex
   bug (`history/rl2_ep5.log`).
3. An inline snippet that set the `support_0` top to 1.0097, then `relocate_scene.py`.

The older scenefix.py and the snippet are in `history/`, rebuilt exactly from the session log.

**Verified.** The two commands, run on the bundle's parent, reproduce the archived `rl2_ep5`
(`checks/rebuild_verification/`):

- all 68 parquet columns are numerically bit-identical;
- the meshes are vertex-identical and the STL is byte-identical;
- the support USD is text-identical;
- only absolute path strings differ.

### Recomputing the plane (optional)

`depth_table/run_dt5b.sh` holds the exact command, run on Babel CPU on 4 Nov ~00:08 EST. It needs our ep5 depth PNGs
and `review_tool/render_review.py` at its hard-coded path.

## 4. Training and selection

**Training.** Edmond's protocol ran unchanged through `kq.sh` (`../training/kq.sh`, sha256 `9360ba9b…`):

- flash_chord de66fea3 via `PYTHONPATH`;
- recipe KEN;
- 100,003,840 steps;
- curriculum `tracking_100m_object2x`;
- seed 43.

It ran on orchard GPU 6, 02:05-02:44 EST on 4 Nov (38 min, 0 nefc overflows). The queue line, the full command and
the logs are in `training/`.

**Checkpoint selection.**

- `kq.sh` evaluated every checkpoint of 70M or more with the kit evaluator (own reference). Its plain best was 90M at
  0.9104.
- We picked **95M** by the *anchored* own-ref score (`selection/anch.py`). This score aligns the rollout on the pot's
  frame-0 pose, as the Kaggle AUC.py does:

  | | 95M | 90M |
  |---|---|---|
  | anchored | **0.9093** [pot 0.921, spoon 0.898] | 0.9012 |
  | plain | 0.9089 | 0.9108 |

- At pick time (02:47 EST) seeds 42 and 43 had finished. Seed 44 finished later and did not beat the pick, so 95M is
  the best of all 21 checkpoints of seeds 42/43/44 by anchored score (`selection/results/`):
  - s42: best 0.8995;
  - s44: best 0.9068, collapsing at 70-80M and at 100M.
- Before the swap test, `anch_phase.py` checked that T8 has no per-segment regression against the T3 run. The only
  negative segment is pot f192-239 (-0.017).
- Later seeds s45 and s46 on the same scene were judged with a rule written down before scoring (notes,
  `refine1104_ep5_notes.md` section C). Neither cleared it, so T8 stays.

**Packing.** `pack_one.sh` and `swap_episode.py` (`../packing/`) produced:

- `one.parquet`: 576 rows, sha256 `11183810…`;
- the T8 file: bob1 with its ep5 rows replaced.

`packing/` holds the commands, logs and receipt. The final combined file reused this exact `one.parquet`.

## 5. Evidence

These numbers come from `C:\v2d_demo\tests\ledger.md`. Kaggle swap tests are bob1 (AUC 0.75166) with one episode
replaced; the episode change is the board change × 20.

| entry | ep5 rows | AUC | change on ep5 | other boards |
|---|---|---|---|---|
| bob1 (Edmond) | bob1 ep5 | 0.75166 | - | SP 0.900, MP 0.45, RPE 4.34503, MPPE 3.61223, CD-O 0.40993 |
| T3 (3 Nov 22:58) | `o_ep5_attcom_s42`@95M on `w7_attcom_ep5` (parent) | 0.75434 | +0.054 | SP 0.95, MP 0.45, RPE 4.29122, MPPE 3.53753, CD-O 0.40638 |
| **T8** (4 Nov 02:49, submission 57976828) | **`r_ep5_rl_K_s43`@95M on `rl2_ep5`** | **0.75799** | **+0.127 vs bob1, +0.073 vs T3** | AUC board only |
| team combined (Wentao, 4 Nov 03:43 EST) | bob1 + ep1 T2 + ep3 T5 + **ep5 T8** + ep8 T6 | 0.76526 | sum of the four swaps | SP 0.950, MP 0.550, RPE 4.10520, MPPE 3.42999, CD-O 0.39579; the same combination with T3 instead of T8 scored 0.76161 / RPE 4.11736 |
| N1 (MPPE only) | team combined + ep10 NEST | - | - | MPPE 3.33289 |
| C2 (CD-O only) | T8 pot copied into the ep1/3/6/8 pot rows | - | - | CD-O 0.38183 → 0.36781; T8 pot vs donor pot -0.124 cm per object |

**Local, GT-free own-reference:**

| | T3 run | T8 run |
|---|---|---|
| plain | 0.8861 | 0.9089 [0.9195, 0.8982] |
| anchored | 0.8697 | 0.9093 |
| anchor frame-0 re-seat | 3.05 deg | 1.21 deg |

With the AUC.py-style anchored score, T8's gain over T3 was predicted before the submission.

## 6. Hard-coded absolute paths (kept as they ran)

- `relevel.py`, `scenefix.py` and `history/scenefix_asrun_*.py`: `RELOCATE` defaults to
  `~/v2d_orchard/relocate_scene.py`. `build_rl2_ep5.sh` sets `RELOCATE` to the staged copy.
- `parent_w7_attcom_ep5/add_ep5_orchard.sh`: `/tmp/haisenk_v2d` and `$HOME/v2d_orchard` (orchard layout).
- `depth_table/dt5b.py`: `/home/haisenk/v2d_ops/kenc/review/tool/render_review.py`.
- `depth_table/run_dt5b.sh`: `/data/user_data/haisenk/...` (scene and depth) and `/home/haisenk/anaconda3/bin/python`.
- `review_tool/render_review.py`: `/data/user_data/haisenk/v2d/...` defaults.
- `parent_w7_attcom_ep5/*`: Babel paths under `/scratch/haisenk-v2d/hand5w`, `/scratch/haisenk-v2d/wentao_v2`,
  `/data/user_data/haisenk/v2d/...` and `$HOME/v2d_ops/kenc/...`.
  - Check-only helpers that are called but not staged: `cmp_pq.py`, `g6dump.py`, `pen0.py`, `seatcheck.py`.
  - Wentao's `install_support.py` comes from his release.
- `selection/`: local Windows paths.
  - `anch.py` and `anch_phase.py` add `C:\v2d_demo\diag\ep5_skepA1` to `sys.path`.
  - `an_common.py` globs `C:\v2d_demo\diag\ep5_objA\data\...` at import, so that folder, or an edited `ROOT`, must
    exist.
  - `anch_run.sh` pulls rollouts from orchard over ssh. Orchard is gone; its files are now on babel-s5-16
    `/tmp/haisenk_v2d`.

## 7. Not recovered, and caveats

- The script that built the control scene `wv2_ctl_ep5` is not in the handoff. This is the scene whose support USD was
  copied into the parent. It was presumably the same as `w7build.sh`'s `control` mode under another name. Its
  resulting support is recorded in the parent scene (`records/parent_w7_attcom_ep5_support.usda`), which is all the
  rebuild needs.
- T8 itself went only to the AUC board. Its SP-SR, MP-SR, RPE and MPPE effects are measured only inside the team's
  combined entries above.
- The absolute ep5 Kaggle AUC is unknown; only the deltas are exact.
- `notes/ep5_relevel/ep5_status_audit_1104.md` calibrates the remaining pot height. It suggests the hidden pot may be
  another 0.5-1.0 cm shorter. That was not acted on: the AUC board was closed.
- `scene_build/history/scenefix_asrun_20261104T0649Z.py` and `support_top_inline_20261104T0649Z.py` were rebuilt from
  the session log, not copied from disk. The orchard copy of the old scenefix.py was overwritten 24 s after the run.
  Section 4 of `history/build_rl2_ep5_asrun.md` explains why they are exact, and the rebuild check shows the result is
  the same.

## 8. Files staged here and where they came from

`SHA256SUMS.txt` lists every file. "= Babel" means the sha256 was checked on 4 Nov ~14:30 EST against the copy on
Babel or babel-s5-16 that ran.

| staged file | source | check |
|---|---|---|
| `scene_build/relevel.py` | `C:\v2d_demo\cupfix\relevel.py` | = babel-s5-16 `/tmp/haisenk_v2d/tools/relevel.py` (`0520b6c9…`); the version that ran at 01:47 EST |
| `scene_build/scenefix.py` | `C:\v2d_demo\cupfix\scenefix.py` | = babel-s5-16 `tools/scenefix.py` (`65f7d926…`); the support regex was fixed 24 s after the rl2 build (see `history/`) |
| `scene_build/relocate_scene.py` | `C:\v2d_demo\orchard\relocate_scene.py` | = babel-s5-16 `orchard_home/v2d_orchard/relocate_scene.py` (`d1f55474…`); also staged by the shared and ep1 parts |
| `scene_build/build_rl2_ep5.sh` | written for this release | reproduces the archived scene (`checks/`) |
| `scene_build/history/*` | rebuilt from the session log; `rl2_ep5.log` read from babel-s5-16 `/tmp/haisenk_v2d/scenes/rl2_ep5.log` | see `build_rl2_ep5_asrun.md` |
| `scene_build/parent_w7_attcom_ep5/{w7prep.py, w7edit.py, w7build.sh, w7fix.sh, wsupport.py, README_ep1_ep5_handoff.md}` | `C:\v2d_demo\handoff\ep1_ep5_20261104\code\tools\track3\hand_fixes\` | = the handoff `SHA256SUMS.txt`, which says they are byte-identical to the Babel copies that ran |
| `scene_build/parent_w7_attcom_ep5/add_ep5_orchard.sh` | `C:\v2d_demo\orchard\add_ep5.sh` | = babel-s5-16 `orchard_home/v2d_orchard/add_ep5.sh` (`63233285…`) |
| `depth_table/dt5b.py` | read from Babel `~/v2d_ops/kenc/diag/ep5_objA/dt5b.py` | the copy that ran (`82ae9593…`); the local copy lacks only the final blank line |
| `depth_table/review_tool/render_review.py` | `C:\v2d_demo\review\tool\render_review.py` | = Babel `~/v2d_ops/kenc/review/tool/render_review.py` (`864350cf…`) |
| `depth_table/out/ep5_depthB.json`, `run_dt5b.out` | `C:\v2d_demo\diag\ep5_objA\` | json = Babel `dt5b_base/ep5_depthB.json` (`2b6225d9…`) |
| `diagnosis/{anchorseat.py, restspots.py}` | `C:\v2d_demo\cupfix\` | = babel-s5-16 `tools/` (`c83a8f40…`, `2e7a2e04…`) |
| `diagnosis/outputs/*.txt` | tool output copied from the session log | - |
| `selection/{anch.py, anch_run.sh, an_common.py}` | `C:\v2d_demo\diag\anch\`, `C:\v2d_demo\diag\ep5_skepA1\an_common.py` | file times are earlier than the 02:47 EST ranking, so these are the versions that ran |
| `selection/anch_phase.py` | rebuilt from the session log (02:48 EST heredoc) | the later local copy only adds an optional `EDGES` env var |
| `selection/results/*` | session log; `C:\v2d_demo\diag\ep_status_1104\ep5_work\anch_rl2.jsonl` | - |
| `training/*`, `packing/*.log`, `packing/kaggle_receipt_T8.json`, `records/archive_manifest_*.json` | `C:\v2d_demo\archive\r_ep5_rl_K_s43\` | `train.log`, `sweep.json` and the eval json = babel-s5-16 `runs/r_ep5_rl_K_s43/`; `one.parquet` = Babel `~/v2d_ops/kenc/tests/r_ep5_rl_K_s43/` |
| `records/{relevel_report.json, W7EDIT.json, parent_loader_report_w7_attcom_ep5.json, rl2_ep5_support.usda}` | extracted from the archived `scene.tgz` (`c0159b0d…`) | - |
| `records/parent_w7_attcom_ep5_support.usda` | extracted from the parent `w7_attcom_ep5.tgz` | shows `support_1`, the partial disk that the re-level drops |
| `records/ep5_diagnosis_plan.json` | `C:\v2d_demo\diag\ep5_diagnosis_plan.json` | diagnosis workflow findings and verdicts: ep5A-1 tilt, ep5-B1 spoon disk |
| `checks/rebuild_verification/*` | written for this release | - |
