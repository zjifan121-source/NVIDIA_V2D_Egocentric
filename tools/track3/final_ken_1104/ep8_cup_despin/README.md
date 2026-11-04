# Episode 8 (beige_cup + white_pot): cup stretched to the depth table, then the cup's fake spin removed

Part of `tools/track3/final_ken_1104/` (Ken's scored episodes 1, 3, 5, 6, 8, 10; NVIDIA V2D Challenge Track 3).
This folder holds the code that built our episode-8 scenes, plus the records of how the final policy was chosen and packed.
Large files (scenes, policies, rollouts, packed parquets) are not here. `bundle_manifest.json` lists them with hashes.

| | Kaggle-scored ep8 rows (T6) | Decided final candidate |
|---|---|---|
| scene | `sc_ep8` (step 1 below) | `rf_ep8_ds` (steps 1-3 below) |
| policy | `h_ep8_sc_K_s42` @ 100,003,840 (sha256 `0ff463d1…`) | `rf_ep8_ds_K_s43` @ 95,006,720 (sha256 `3d33ff6b…`) |
| packed one.parquet | sha256 `2e5696d4…` (438 rows) | sha256 `6da1e7bf…` (438 rows) |
| status | submitted as swap test T6. Its rows are in the team's best entry (AUC 0.76526, 4 Nov 03:43 EST) and in the later entries built on that base (e.g. N1 on MPPE) | **never submitted**. Ken stopped all submissions at ~10:05 EST on 4 Nov, and the AUC board had no submissions left that day |

Episode 8: the right hand lifts the beige cup, pours twice (f55-72 and f128-155) and sets it down (the hold is f46-185).
The left hand holds the white pot and lifts it from about f95 to f165. There are 219 frames. Slot 0 is beige_cup, which the kit README
treats as axially symmetric. Slot 1 is white_pot.

## 1. What the fixes change and why

### Step 1: `sc_ep8` (ledger T6). Cup mesh stretched 2.8 cm, cup support disk, pot COM

- **Diagnosis.** Our GT-free depth-table study is `diagnosis/depth_table/dt8.py` → `dt8.log`. It uses our own
  FoundationStereo depth and was run on bob1's ep8 scene on 3 Nov at 23:52 EST.
  - In bob1's scene the cup's mesh bottom hovers 4.62 cm above the sim support at rest.
  - Under the cup, the measured table is 1.81 cm above that support. The support is a single disk at pot level.
  - The cup bottom still sits 2.82 cm above the measured table: 2.815 cm at the start rest (annulus 2.79, ring 2.83) and 2.78 cm at the end rest.
  - The cup's visible surface agrees with the depth: median +0.17 cm, silhouette IoU 0.88.
  - So the pose is right. The single-view SAM 3D mesh lacks the cup's base, and the table under the cup is too low.
  - Separately, the pot URDF had its inertial origin at the mesh origin. That is 8.0 cm from the pot's shell centroid and outside its base footprint.
- **Fix.** All three changes are GT-free; poses, hands and contact labels are untouched.
  - **Cup mesh.** `scenefix.py --obj beige_cup:2.8` stretches the mesh 2.8 cm downward along the frame-0 world vertical, with the rim fixed. The cup bottom goes from z 1.000 to 0.972. The tool rewrites the processed OBJ, the visual/collision STL, the score mesh and the bundle_cleaned mesh.
  - **Cup disk.** `--add-disk cupdisk:-0.003:0.008:0.085:0.971` adds a 1 cm support disk under the cup with its top at the measured table, z 0.971.
  - **Pot COM.** `comfix.py fix … white_pot` moves the pot's URDF inertial origin to the area-weighted shell centroid, (-0.008451, -0.062887, 0.049392). This is the same rule as the adopted ep5 scene.
  - **Effect.** Cup rest float goes from 4.62 cm to 0.10 cm. Rest IoU is 0.869 (bob1 0.876).
- **Provenance caveat**, from `notes/ep8_cup_despin/verify_ep8_10.md`:
  - T6 was built on 4 Nov 00:12-00:20 EST. That is before the ep8 skeptic verdicts.
  - The later B1 verdict advised against sizing the mesh extension from the depth table alone. T6 did exactly that.
  - Kaggle then measured T6 at +0.068 for ep8.
  - The pot-COM change was applied as a rule. No ep8 failure had been attributed to it, and the 4 Nov refine (D3) still lists the pot's lift sag as "cause not established".

### Steps 2-3: `rf_ep8_ds`. Cup despun about its axis, right hand re-attached spin-free

- **Diagnosis ep8-B2.** It was confirmed by an adversarial verdict ("partly": the mechanism holds, the size was overstated); see `diagnosis/ep8_B2_diagnosis_extract.json`.
  - FoundationPose spins the near-symmetric cup about its own axis: -125.7° over f58-68 and -79.7° over f128-154.
  - The raw HaMeR wrist turns only +0.7° and +6.2° over the same frames. The likely cause is aliasing on 4 marker bumps about 90° apart.
  - The team's `attach_hands` rides the reference right hand rigidly on the cup. The reference hand therefore orbits and spins with the fake spin: up to 27°/frame, 8 frames more than 15 cm from the video wrist, max 19 cm / 140°.
- **New per-frame evidence on the T6 rollout** (`diagnosis/d1_decomp.py` → `d1_out.json`):
  - The policy copies the fake spin: achieved twist over f55-72 is -111.5° against -110.7° in the reference.
  - Cup centroid error jumps from 0.24 cm (f61) to 5.70 cm (f65) inside the spin burst. Best lag is 0 frames.
  - Over the hold, ADD correlates 0.77 with |reference twist|, against 0.38 with pour tilt.
  - Lost credit in the burst frames is 0.0103 episode AUC (`d3_gain.json`).
- **Fix** (planned fix F2, "no per-episode numbers"):
  - **Despin.** Replace the cup's rotation about its mesh symmetry axis (PCA axis through the surface centroid) with frame 0's: `T'[t] = T[t]·S(-θ_t)`, where θ_t is the twist of R0ᵀRt about that axis. Position, axis direction and the axis line are unchanged. The kit README says axial spin of the cups is unscored.
  - **Spin-free re-attach.** Re-attach the right hand spin-free: it follows the cup's translation and tilt, not its spin. The grip is still taken at the closest frame, and the blends use the continuous rotation branch.
  - **How it is built.** The despin is built from bob1's exact merged bundle with the team's `run_load_retarget.py` plus a cleanup patch (`sym_common` mode `despin`). Then **only** the cup pose/root columns, the 17 right-hand columns and `bundle_cleaned/result.npz` are merged into a copy of `sc_ep8`.
  - **What stays the same.** T6's stretched cup mesh, cupdisk, pot COM, pot track and left hand stay byte-identical. So the CD-O mesh rows are identical to T6's.
- **Pre-registered gates G1-G8 all pass** (`gates.json` 10:38 EST → `gate_results.json` 10:45 EST):
  - 36/38 files are identical and exactly the 21 intended columns changed.
  - Pure spin: centroid, axis and residual twist all 0.0.
  - The hand is never more than 15 cm from the video wrist (the T6 scene, whose hands are bob1's, had 8 such frames). Max 6.51 cm; wrist rotation step over f55-72 7.0°/frame against 27.1.
  - Contacts are 2567 against 2530 link-frames.
  - Rest seating moves by at most 0.061 cm.
  - Modelled end-rest side effect is 0.00155, within its limit of 0.00513.

## 2. Exact inputs

| input | identity |
|---|---|
| parent scene | bob1 ep8 = Edmond's `candidate_bob1` episode 8. It is our `ken_c_scored_attach` build (Kaggle bob1 AUC 0.75166). Copies: `b1_ep8` (C:/v2d_demo/orchard/b1scenes.tgz; orchard and babel-s5-16 `/tmp/haisenk_v2d/scenes/b1_ep8`) and `eb1_ep8` (`/data/user_data/haisenk/v2d/edmond_bob1/relocated/eb1_ep8`; HF `experiments/edmond/bob1_submission_20261103`). Every mesh/URDF/USD/bundle md5 is identical; see `scene_build/scene_md5_lists_20261104.txt` |
| bob1 merged bundle (despin input) | `/data/user_data/haisenk/v2d/NVIDIA_V2D_Egocentric/baseline/reconstruction/data/outputs/load_retarget/episode_000008/ken_c_scored_attach/bundle_merged`, result.npz md5 `c75cb24c63d3555f1ad5a013900f274f` |
| scene code repo | team repo `edm_training` @ `1a6bae7b2682a40087cd70e5d68c6aa3e5f21b77` (Babel `/data/user_data/haisenk/v2d/v2d_train`, `.venv_retarget`, Python 3.11). The MANO weights come from its data dir and are not included (licence) |
| training code | flash_chord @ `de66fea3` via `PYTHONPATH=/tmp/haisenk_v2d/fc_src_de66` (Edmond's protocol), driver `kq.sh` (md5 `9d5a6253…`, shared, not in this folder) |
| intermediate scenes | `sf_ep8` → `sc_ep8` (T6) → `rf_ep8_ds`. md5 lists are in `scene_build/scene_md5_lists_20261104.txt` |

## 3. Rebuild from the parent (exact command sequence)

```bash
R=<absolute path of this folder>                          # tools/track3/final_ken_1104/ep8_cup_despin
# --- step 1: sc_ep8 = T6 scene (CPU; any Python with numpy, scipy, trimesh, pyarrow) ---
S=<bob1 ep8 scene dir>                                    # b1_ep8 / eb1_ep8
export RELOCATE=$R/scene_build/1_sc_ep8/relocate_scene.py # the tools default to ~/v2d_orchard/relocate_scene.py
python -I $R/scene_build/1_sc_ep8/scenefix.py $S sf_ep8 --obj beige_cup:2.8 --add-disk cupdisk:-0.003:0.008:0.085:0.971
python -I $R/scene_build/1_sc_ep8/comfix.py fix sf_ep8 sc_ep8 white_pot
#   original run: orchard, 4 Nov 00:12 / ~00:20 EST, /tmp/haisenk_v2d/scenes/{b1_ep8 -> sf_ep8 -> sc_ep8}

# --- step 2: despin build (Babel CPU step; scene_build/2_despin/feas.sh is the script that ran, 4 Nov 01:51-01:57 EST) ---
D=/scratch/haisenk-v2d/diag/ep8_plan; T=/data/user_data/haisenk/v2d/v2d_train
cp -aL <load_retarget>/episode_000008/ken_c_scored_attach/bundle_merged $D/bundle_base     # md5 c75cb24c
#   sym_common_ep8.py = ~/v2d_ops/kenc/hand5/ep10/sym_common.py with only its diag path changed (sed in feas.sh)
#   patch_despin_ep8.py sets _H5_MODE = "despin" and execs sym_common_ep8.py (absolute path inside: edit it)
cd $T && V2D_REPO=$T H5_PATCH=$D/patch_despin_ep8.py timeout 900 $T/.venv_retarget/bin/python \
   $R/scene_build/2_despin/rlr_h5.py --bundle $D/bundle_base --episode 8 --attach-hands --smooth-hands 2 \
   --skip-video --device cpu --sequence-id ep8_ken_c_scored_attach --out $D/despin
#   (the original run used the same file from ~/v2d_ops/kenc/hand5/rlr_h5.py, md5 0f2cef75)
#   This ends with "ValueError: string is not a file ... ep8_ken_c_scored_attach.obj" in check_output,
#   AFTER every scene file is written. That is expected for --bundle without --merge; see despin.log.
#   Parquet md5 2ae8488e19c5bce2cfbbcc758347bd27, bundle_cleaned/result.npz md5 266a5213cf9968ce68237d953a5bd642.
#   A control build (no H5_PATCH) reproduced bob1's scene: cup/pot/wrist poses identical (0.000 cm, 0.0 deg);
#   only the re-sampled contact-label / mesh-radius columns differ, which is known build noise (feas.out).

# --- step 3: merge + pre-registered gates (scene_build/3_merge_install/b1_copy_src.sh, b2_build.sh; 10:38-10:45 EST) ---
O=/data/user_data/haisenk/v2d/refine1104/ep8            # merge_gates.py asserts this output prefix (edit to rebuild elsewhere)
cp -a /tmp/haisenk_v2d/scenes/sc_ep8 $O/src_sc_ep8      # the installed sc_ep8 that T6 trained on
mkdir -p $O/despin_src $O/code                          # b2_build.sh received the next two files base64-encoded (same bytes)
cp $R/scene_build/3_merge_install/merge_gates.py $O/code/; cp $R/scene_build/3_merge_install/gates.json $O/
cp $D/despin/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave/*.parquet $O/despin_src/despin.parquet
cp $D/despin/bundle_cleaned/result.npz $O/despin_src/bundle_cleaned_result.npz
cp -a $O/src_sc_ep8 $O/rf_ep8_ds
$T/.venv_retarget/bin/python -I $O/code/merge_gates.py $O $O/src_sc_ep8/bundle_merged \
   /data/user_data/haisenk/v2d/edmond_bob1/relocated/eb1_ep8/assets/human_motion_data/ego_recon/processed/ep8_ken_c_scored_attach_beige_cup.obj 0.01027
#   -> gate_results.json "all_pass": true; merged parquet md5 b09c4578bb3ca0689320cad7085f314c (before relocation)

# --- step 4: install + queue (scene_build/3_merge_install/b3_install_queue.sh, babel-s5-16, job 10774619, 10:46 EST) ---
bash ~/v2d_ops/install_scene_b.sh $O/rf_ep8_ds rf_ep8_ds   # copy to /tmp/haisenk_v2d/scenes/rf_ep8_ds + relocate + path check
printf "rf_ep8_ds_K_s43 8 $PQ 43 KEN 10\n" | bash ~/v2d_ops/kq_add_b.sh                # also s42 and s44
```

`scene_build/1_sc_ep8/rebuild_check_20261104.txt` records a local re-run of step 1 on 4 Nov from the staged tools.
- Every cup and pot mesh, STL, URDF, support USD, bundle mesh and score mesh came out byte-identical to the trained-on `sc_ep8`.
- Of the 66 parquet columns, 63 match and only the three path columns differ, by design. The trained-on file also carries two Hive partition-key columns (sequence_id, robot_name) that relocation wrote into it.
- The original step-1 command lines were not saved. They are reconstructed from `sf_ep8.log` (scenefix's printed report) and from comfix's result. The re-run is what verifies them.

## 4. Training and selection of the final policy

- **Training.** `kq.sh` runs Edmond's protocol for queue line `rf_ep8_ds_K_s43 8 <parquet dir> 43 KEN 10`; `training/kq_queue_lines_ep8.txt` has the full list:
  - `train_flash_sac.py experiment=sharpa_flash_sac curriculum=tracking_100m_object2x training.total_environment_steps=100003840 reset.seed=43`
  - plus the KEN overrides in `training/rf_ep8_ds_K_s43/recipe.txt`: motion_speed 1.0, sim.fps 20.2541, wrist termination 0.3, object termination 0.1 m / 0.5 rad, replay 4,194,304 float32, njmax 1024, checkpoints every 5,000,192 steps.
  - It ran on babel-s5-16 (H100) on 4 Nov from 11:07 to 11:58 EST and exited 0 with 0 nefc-overflow lines.
- **Checkpoint sweep.** kq.sh evaluates every checkpoint from 70M with the kit evaluator (`run_policy_evaluation.py`, own-reference add_auc) and keeps the best (`sweep.json`, `best.txt`).
  - s43: 95,006,720 = **0.9902** [cup 0.984, pot 0.996]. 90M/100M give 0.978/0.988, and every checkpoint from 70M to 100M scores 0.977-0.990.
  - The other seeds: s42 0.988 @ 85M, s44 0.987 @ 90M.
  - These numbers are measured against the despun reference, so they are not comparable with T6's 0.973.
- **Seed and scene choice.** This is GT-free, with a rule written before scoring (`selection/decision_rule_retry.json`, 12:12 EST, before the rf rollouts were fetched).
  - `selection/judge_ep8.py` scores each rollout with the verbatim kit scorer `AUC.py` (sha256 `3f1a98997f25e521…`; through `emu8.py`) against five stand-in truths: REF, REF_DS, REF_B1, SEAT and RAY.
  - `selection/decide_rf.py` applies the rule and writes `verdict_rf.json`. All five clauses pass.
  - The rf seed with the best REF score was chosen: K_s43 @ 95M.
  - An adversarial review approved it with caveats (`notes/ep8_cup_despin/refine1104_ep8_review.md`).

## 5. Packing

```bash
# packing/pack_new.sh, babel-s5-16, 4 Nov 13:47 EST (it also packed ep1)
bash ~/v2d_ops/pack_one.sh 8 /tmp/haisenk_v2d/runs/rf_ep8_ds_K_s43/early_95006720/results \
  /tmp/haisenk_v2d/scenes/rf_ep8_ds/score_input/meshes \
  https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6 \
  ~/v2d_ops/kenc/final1104/rf_ep8_ds_K_s43/one.parquet
# -> 438 rows, sha256 6da1e7bf92856491de3bbca8ba47f1c2dee60ef0163d429c9bcecf43e3d46611, checkpoint_sha256 3d33ff6b...
bash ~/v2d_ops/kenc/final1104/combine_final.sh local_best 1:<ep1 one> 3:<ep3 one> 5:<ep5 one> \
  8:~/v2d_ops/kenc/final1104/rf_ep8_ds_K_s43/one.parquet 10:<ep10 one>      # see packing/final1104_local_best_manifest.txt
# -> local_best/track3_submission.parquet sha256 5d1d534f... (kit --dry-run only; NOT submitted)
```

- `pack_one.sh` (shared, md5 `16f82b5c…`) is the team wrapper plus the kit packer narrowed to one episode. The mesh dir is the scene's own `score_input/meshes`, the T6 meshes.
- The packed rows carry bob1's code commit URL (`1689cf63…`), so all rows have the same provenance. That commit does not contain this folder's code.

## 6. Evidence

**Kaggle** (from `C:\v2d_demo\tests\ledger.md`; each swap test = bob1 with ONE episode replaced, so the ep8 change = dAUC × 20):

| test | ep8 rows | AUC | dAUC | ep8 change | note |
|---|---|---|---|---|---|
| bob1 | bob1 | 0.75166 | - | - | base |
| T4 (4 Nov ~00:40, submission 57972783) | cf_ep8 V1 (`superseded_v1_cf_ep8/cupfix.py`, +4.62 cm cup bottom), f_ep8_cf_K_s42 @ 70M | 0.75402 | +0.00236 | +0.047 | superseded |
| T6 (4 Nov ~01:50) | **sc_ep8**, h_ep8_sc_K_s42 @ 100M (own-ref 0.973 [0.953, 0.993]) | **0.75505** | **+0.00339** | **+0.068** | adopted |
| team entry (Wentao, 03:43 EST) | bob1 + ep1 T2 + ep3 T5 + ep5 T8 + **ep8 T6** | 0.76526 | | | team best on all 6 boards at 06:25: SP-SR 0.950, MP-SR 0.550, RPE 4.10520, MPPE 3.42999, CD-O 0.39579 |

- Cross-check (`notes/ep8_cup_despin/verify_ep8_10.md`): two combined team entries scored 0.76161 (with ep8 T6) and 0.76057 (with ep8 T4). The difference, ×20 = +0.021 for ep8, equals T6 − T4.
- **CD-O.** T6's stretched cup is derived to have moved CD-O by -0.00957 (-0.345 cm on that object). C1 (submission 57983875) copied T6's ep8 beige_cup rows into ep10 and scored CD-O 0.38183. C2 replaced the ep8 pot rows with ep5's pot.
- **rf_ep8_ds has no Kaggle number.**

**Local, GT-free** (`selection/JUDGED.json`, `verdict_rf.json`; add_auc of the ep8 rows under five stand-in truths):

| run (checkpoint) | REF | REF_DS | REF_B1 | SEAT | RAY | cup ADD median, burst f62-69 |
|---|---|---|---|---|---|---|
| T6 = h_ep8_sc_K_s42 (100M) | 0.95691 | 0.95733 | 0.96219 | 0.93936 | 0.94406 | 4.76 cm |
| **rf_ep8_ds_K_s43 (95M)** | **0.98102** | **0.98145** | 0.97660 | **0.96076** | 0.96789 | 0.81 cm |
| family mean, KEN4 (sc_ep8) → RF3 (rf_ep8_ds) | 0.95826 → 0.97931 | 0.95826 → 0.97941 | 0.95858 → 0.97950 | 0.94021 → 0.95966 | 0.94870 → 0.96675 | 4.27 → 0.67 cm |

- All 15 rf × stand-in scores beat the best of the 5 sc_ep8 rollouts under the same stand-in.
- SP-SR and MP-SR are 1 everywhere. MPPE family mean improves 1.060 → 0.851 cm under REF.
- Frames with more than 3 cm cup centroid error: 7-15 per sc_ep8 rollout against 0 in every rf rollout (`perframe_rf.json`).
- A plain re-implemented scorer (reviewer) agrees: family 0.9737 → 0.9910.

## 7. Files here

| path | what |
|---|---|
| `scene_build/1_sc_ep8/scenefix.py` | mesh stretch with rim fixed, support edits, path relocation (`--obj`, `--support-top`, `--add-disk`) |
| `scene_build/1_sc_ep8/comfix.py` | URDF inertial origin moved to the shell centroid (`fix`), and a COM vs footprint report (`check`) |
| `scene_build/1_sc_ep8/restspots.py` | cup footprint centres at the rests. The cupdisk centre (-0.003, 0.008) is the midpoint of the f0 (-0.0013, -0.0022) and end-rest (-0.0047, +0.0175) footprints |
| `scene_build/1_sc_ep8/relocate_scene.py` | repoints absolute paths in a copied scene (`RELOCATE` target) |
| `scene_build/1_sc_ep8/sf_ep8.log`, `sc_ep8_files.md5`, `rebuild_check_20261104.txt` | scenefix's report from the original run; the md5 list of the installed sc_ep8; the 4 Nov rebuild check |
| `scene_build/2_despin/rlr_h5.py` | team `tools/track3/run_load_retarget.py` @ 1a6bae7b plus the H5_MERGED_NPZ/H5_PATCH hooks; the diff is in `rlr_h5_vs_team_run_load_retarget_1a6bae7b.diff.txt`. The same hooks are on branch `haisen/hand-fixes-ep3-10-20` (5294c6b) |
| `scene_build/2_despin/sym_common_ep8.py`, `patch_despin_ep8.py` | the exact patch files used: `attach_hands` redefined, mode `despin` |
| `scene_build/2_despin/feas.sh`, `feas.out`, `despin.log`, `despin_build_files.md5.txt` | the build driver as run (control + despin builds and comparisons), its output, the build log, and file md5s |
| `scene_build/2_despin/chk_scene.py`, `cmp_pq.py`, `cmp_scene.py` | checks called by feas.sh (twist, wrist steps, pose diffs) |
| `scene_build/3_merge_install/*` | merge + gates (`merge_gates.py`, `gates.json` pre-registered, `gate_results.json`), the Babel step scripts b1-b3, install/queue helpers and logs |
| `scene_build/scene_md5_lists_20261104.txt` | md5 of every file in b1_ep8, eb1_ep8, sf_ep8, sc_ep8, rf_ep8_ds (as built and as installed) |
| `scene_build/superseded_v1_cf_ep8/cupfix.py` | built cf_ep8 (T4, +4.62 cm). **Not** in the final chain; kept because T4's rows were Kaggle-submitted |
| `diagnosis/` | the ep8-B2 extract; D1 spin decomposition, D2 despin side effect, D3 gain (`d*_*.py/json`); `depth_table/` (dt8.py + its imported `render_review.py`, dt8.log, ep8_depthtab.json) |
| `training/` | rf_ep8_ds_K_s43 and T6 recipe/best/sweep/archive manifests, the rf train.log, the evaluator config and metrics of the packed rollout, queue lines |
| `selection/` | judge_ep8.py + emu8.py (needs the kit's AUC.py, not copied here), decide_rf.py, the rule, judge outputs, verdict, per-frame and non-symmetric risk checks |
| `packing/` | pack_new.sh, pack args/logs (rf and T6), combine_final.sh and the local-best combine manifest |
| `bundle_manifest.json`, `PROVENANCE.tsv` | large items with hashes and locations; the source and md5 of every file in this folder |

Notes (outside the code tree, `notes/ep8_cup_despin/`): the 4 Nov refine notes, the adversarial review, READY.json, the ep8 status audit, and the ep8/ep10 verification.

## 8. Hard-coded personal paths (kept as is)

| path | appears in |
|---|---|
| `/home/haisenk/v2d_ops/...` | dt8.py (render_review import); combine_final.sh and pack_one.sh (kaggle_env python) |
| `~/v2d_orchard/relocate_scene.py` | default `RELOCATE` of scenefix.py, comfix.py and cupfix.py |
| `/data/user_data/haisenk/v2d/...` | feas.sh, b1/b2/b3 scripts, the `merge_gates.py` output assert, render_review.py |
| `/scratch/haisenk-v2d/diag/ep8_plan` | feas.sh, patch_despin_ep8.py, sym_common_ep8.py (diag dir) |
| `/tmp/haisenk_v2d` | b1/b3, install_scene_b.sh, kq_add_b.sh, pack_new.sh |
| `C:\v2d_demo\...` | diagnosis/ and selection/ scripts (local analysis machine) |

## 9. Caveats and what could not be recovered

- **Step-1 provenance.**
  - The original command lines for sf_ep8/sc_ep8 were not logged. They are reconstructed and verified by the byte-identical rebuild.
  - `scenefix.py`'s file time (01:49 EST) is later than the 00:12 build, so the exact file used then cannot be proven. Only its behaviour on this input is verified.
  - How the 8.5 cm disk radius was chosen is not recorded. It covers both cup footprints (r_all 6.4 cm) and stays clear of the pot base.
- **Despin build.**
  - The despin build crashed in its final `check_output` step, after every scene file was written. It therefore has no report.json or score_input. The merge used only its parquet columns and `bundle_cleaned/result.npz`, and the gates are the acceptance record.
  - The despin build directory is node-local on babel-t9-28 `/scratch`. Its two used files are kept on `/data` (`despin_src/`).
  - `merge_gates.py` refuses any output path outside `/data/user_data/haisenk/v2d/refine1104/ep8/`.
- **Gates.**
  - G3 measures the centroid with the build's own sampler. With an exact area-weighted centroid the reviewer got 0.078 cm against the 0.05 cm limit, which is under 1 mm.
  - G7 is a model.
- **Host.** T6's seeds trained on orchard (Python 3.12). The rf seeds trained on babel-s5-16 (Python 3.11) with the same code commits, and no same-host control was run.
- **Seed pick.** K_s43 was picked by its REF score, so there is a small winner's curse. All three rf seeds pass.
- **Not tested on Kaggle.**
  - rf_ep8_ds was never submitted, so its gain is GT-free only (lesson T7: a better GT-free fit once failed on Kaggle, though that was a mesh change and these meshes are unchanged).
  - If the hidden scorer treated beige_cup as non-symmetric and the real cup spun like FoundationPose, K_s43 would score 0.742 against T6's 0.961 (`selection/nonsym_summary.json`). The kit README says cup axial spin is unscored.
- **Installed parquet.** The installed `rf_ep8_ds` parquet's `.orig` is bob1's original parquet, inherited through relocation. It is not a pre-relocation copy of rf's own parquet; that copy is on `/data` (md5 `b09c4578…`).
- **Not staged.**
  - The kit's `AUC.py` (kit code).
  - `kq.sh`, `pack_one.sh` and `swap_episode.py` (shared training and packing tools).
  - test2/test3 and seatrest.py: a seat-rest variant `despin_seat` that was **not** used.
  - early*.py and disk_clear.py (planning probes).
  - The ep8 skeptic work dirs.
  - The T6 submission id: no local receipt; the ledger has only the score.
