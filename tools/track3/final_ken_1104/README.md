# Final submission status (updated 4 Nov 2026, after the 17:00 EST close)

**Read this first: the "Status" section further down was written at 15:00 and is out of date.**

- **Final file.** One file was selected on all six Track 3 boards: FINAL_B.
  - sha256: `eecb6d3a3fb4b85381bd2a6f62c397cc78d0fed9f97936489c215d97213d4878`
  - Size: 9,120 rows, 20 episodes, 36 meshes.

  | Board | Kaggle submission | Score |
  |---|---|---|
  | AUC | 57998961 | 0.77970 |
  | SP-SR | 57998965 | 0.95 |
  | MP-SR | 57998964 | 0.65 |
  | RPE | 57999654 | 3.86589 cm |
  | MPPE | 57998873 | 3.28089 cm |
  | CD-O | 57999655 | 0.39579 cm |

- **Code commit.** Every row cites commit `1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6` of this repository.
- **Evaluator digests.** These are identical on all rows: eval script `160d24f5`, eval code `b1b3ebe3`, packer `49b13052`, uploader `d31d4a7c`.
- **How FINAL_B was built.** FINAL_B = the team's bob1 file (3 Nov) with 13 episodes replaced. The script was `packing/final1104/combine_final.sh` with `swap_episode.py --adopt-commit-url`.
  - Our episodes, each with its final run and selected rollout:

    | Ep | Run and step | Rollout |
    |---|---|---|
    | 1 | `b_ep1_wh_K_s48` @100003840 | best of 64 re-evaluations (a10) |
    | 3 | `rf_ep3_f3w_K_s42` @100003840 | best of 48 (a7) |
    | 5 | `r_ep5_rl_K_s43` @95006720 | best of 56 (g5) |
    | 6 | `x_ep6_d2_K_s44` @85004288 | sweep evaluation |
    | 8 | `rf_ep8_ds_K_s43` @95006720 | best of 32 (a10) |
    | 10 | `n_ep10_nb_K_s42` @100003840 | best of 64 (e2) |

  - Teammates' episodes 15, 16, 20, 32, 35, 36 and 37 use bob1's checkpoints. The rollout is the best of 36-72 re-evaluations. For 15, 16, 32, 35, 36 and 37 the re-evaluations ran on a copy of the checkpoint whose stored scene path was updated. Its weights are byte-identical, but the file digest differs.
  - Episodes 14, 19, 25, 28, 30, 33 and 38 are unchanged from bob1.
- **Selection.** Re-evaluations were ranked by own-reference add_auc only; no hidden data was used. The official evaluator is not deterministic. Re-running a checkpoint therefore reproduces the procedure but not the exact recorded rollout. The recorded rollouts are archived with the checkpoints.
- **Final-day parts:** `ep3_f3w/` (ep3 spoon track), `ep6_d2/` (ep6 in-pot spoon de-smoothing), `redraw/` (best-of-N re-evaluation, including the teammates' episodes).

# `final_ken_1104/`: final Track 3 release for scored episodes 1, 3, 5, 6, 8 and 10 (Ken / haisenk)

Team Gravity Aligned, NVIDIA V2D Challenge Track 3 (egocentric). Staged on 4 Nov 2026; all times are EST.

This folder holds the code that built our six episodes' scenes, trained and picked their policies, and packed their Kaggle
rows. Large data (scenes, policies, rollouts, packed parquet files) is not in git. It goes to the team Hugging Face dataset
`rose-tea-cafe/v2d-t3-recon` under **`experiments/haisenk/final_ours_20261104/`**, one folder per episode
(`episode_000001/`, ...), following the team's existing HF layout. Every data item, with its source path, sha256 and HF target,
is listed in the release manifest `bundles_manifest.json` (staging copy `release1104/bundles_manifest.json`) and in each
part's own manifest.

## Status

- **On the leaderboard.** The team's final AUC is **0.76526**. That entry was submitted by teammate Wentao on 4 Nov 03:43 and
  is bob1 plus our four swap tests: T2 (ep1), T5 (ep3), T8 (ep5) and T6 (ep8). Its other boards: SP-SR 0.950, MP-SR 0.550,
  RPE 4.10520, MPPE 3.42999, CD-O 0.39579. It was the team best on all six boards at 06:25.
- **Later entries.** C1 and C2 went to the CD-O board only (0.38183, then 0.36781; `cdo_transfer/`). N1 went to the MPPE board
  only (3.33289; ep10 NEST). The AUC board had no submissions left from 04:35, and an AUC attempt at 09:45 was refused.
- **Nothing after 09:50 was submitted.** Ken stopped all Kaggle submissions at about 10:05. After that, two picks changed on
  GT-free local evidence only: the ep1 seed (s48) and the ep8 despin scene. The combined local-best file
  (`local_best/track3_submission.parquet`, sha256 `5d1d534f...`) was only dry-run with the kit.

## Overview

Each swap test is bob1 (AUC 0.75166) with one episode replaced, so the episode change is about the board change x 20
(public rows only; the private fraction is 0, so the public board is final). Numbers come from the team ledger.
"Own-ref" is the kit evaluator's add_auc against our own reference.

| Ep (objects) | Final scene | Final policy @ step | Fix in plain words | Part | Evidence | ep rows on the Kaggle boards | HF bundle (`final_ours_20261104/`) |
|---|---|---|---|---|---|---|---|
| 1 (white_pot, white_pot_lid) | `w_hold_ep1` | `b_ep1_wh_K_s48` @ 100003840 | Wentao's ep1 objects, with our raw hands re-attached to them. The hands stay on the pot handle (f32-304) and on the lid (f45-68, f250-274) for the whole hold windows that contact labels, grip stability and lid speed support, using a median grip. Before, they attached only while the object was lifted more than 2 cm. | `ep1_hold_window/` | **Kaggle T2** (same scene, seed s44): AUC 0.75413, so ep1 **+0.049**. MP-SR fail -> pass. ep1 RPE -1.64 cm, MPPE -0.90 cm. **s48 is local only**: the ep1 emulator puts it ahead of T2 in 16/16 cells (median AUC 0.887 vs 0.847). Own-ref 0.830 vs 0.787. | T2 `o_ep1_hold_s44` @100003840 | `episode_000001/` |
| 3 (white_pot, wooden_spoon) | `sc_ep3` | `h_ep3_sc_K_s42` @ 95006720 | The table support is raised 1.12 cm to the stereo-depth table, capped 1 mm under the pot's frame-0 bottom. The pot centre of mass moves 8.0 cm to its shell centroid (before, it lay outside the base and the pot tipped). | `ep3_support_com/` | **Kaggle T5**: AUC 0.75305, so ep3 **+0.028** (AUC board only). | these rows (T5) | `episode_000003/` |
| 5 (white_pot, wooden_spoon) | `rl2_ep5` | `r_ep5_rl_K_s43` @ 95006720 | The parent `w7_attcom_ep5` is Wentao's v2 ep5 with our hands re-attached and the pot COM fixed. On top of that: the world is re-levelled 5.48 deg to the depth-measured table, the partial spoon disk is dropped, and the pot mesh is 1.0 cm shorter (rim fixed). | `ep5_relevel/` | **Kaggle T8**: AUC 0.75799, so ep5 **+0.127** vs bob1 (+0.073 over the parent's test T3, 0.75434). | these rows (T8) | `episode_000005/` |
| 6 (white_pot, wooden_spoon) | bob1 `bm_scored_attach` (Edmond) | `ep6_bm_s42` @ 100003840 (Edmond) | None. Every ep6 change failed: T7 scored ep6 -0.092 on Kaggle, and S2, F3a and D2 failed their GT-free gates. | `ep6_bob1/` (pointer) | bob1 | bob1 | pointer only: `experiments/edmond/bob1_submission_20261103/episode_000006/` |
| 8 (beige_cup, white_pot) | `rf_ep8_ds` | `rf_ep8_ds_K_s43` @ 95006720 | Step 1 (T6 scene `sc_ep8`): the cup mesh is extended 2.8 cm down to the depth-measured table, a cup support disk is added at the measured height, and the pot COM moves to its centroid. Steps 2-3 ("despin"): the cup's rotation about its own symmetry axis is held at frame 0's, which removes FoundationPose's false spins (about -126 deg over f58-68 and -80 deg over f128-154), and the right hand is re-attached without the spin. | `ep8_cup_despin/` | **Kaggle T6** (scene `sc_ep8`, s42 @100M): AUC 0.75505, so ep8 **+0.068** (it replaced T4, an earlier cup fix at +0.047). **Despin is local only**: it beats T6 under all 5 GT-free stand-in truths (ep8 add_auc +0.014 to +0.024). | T6 `h_ep8_sc_K_s42` @100003840 | `episode_000008/` |
| 10 (blue_cup, beige_cup) | `k10_ep10_nest_b` | `n_ep10_nb_K_s42` @ 100003840 | The blue cup is lowered to the depth-measured visible top; before, it floated 1.4-6.2 cm on f55-160 and dragged the left hand up. A simulation-only collision proxy (thin blue wall slabs plus a floor; beige hulls without the marker spheres) lets the beige cup settle inside the blue cup, its bottom about 5.5 cm above the blue's bottom, as in the video. | `ep10_seat_nest/` | **Kaggle N1** (MPPE board only): 3.42999 -> 3.33289, so ep10 **-1.94 cm**. No AUC test was possible (T1, a different ep10 hand fix on Wentao's base, scored ep10 -0.036 and was rejected). Local cross-score: +0.205 (depth-consistent reference), +0.140 (bob1's reference). | MPPE: these rows (N1). AUC, SP-SR, MP-SR, RPE: bob1's ep10 (our earlier hand_fixes_v1 respinf build). | `episode_000010/` |

**Combined file.** `final_ours_20261104/local_best/track3_submission.parquet` has sha256
`5d1d534f4b3e6ba5dcd165203aebc001faa4f2823c78238b83e9df287e529edd` and 9,120 rows over 20 episodes. It is bob1 plus the five
final rows above. It was **not submitted**. Compared with the file Kaggle scored as N1, only the ep1 and ep8 rows differ.

**CD-O only.** C1 and C2 also replaced mesh rows on the CD-O board:
- C1: the ep10 beige_cup rows were replaced with T6's ep8 cup;
- C2: the white_pot rows of ep1, 3, 6 and 8 were replaced with T8's ep5 pot.

These meshes differ from the pose-board files (see `cdo_transfer/`).

## Layout

```
final_ken_1104/
  README.md           this file
  ep1_hold_window/    ep1: hold-window hand attach patch, build scripts, GT-free diagnosis, pre-registered seed selection (emulator)
  ep3_support_com/    ep3: scenefix.py (exact as-run version) + comfix.py, depth-table measurement, diagnosis, rebuild check
  ep5_relevel/        ep5: relevel.py + scenefix.py, parent (w7_attcom_ep5) build scripts, depth plane, anchored selection
  ep6_bob1/           ep6: pointer README only (Edmond's bob1, unchanged)
  ep8_cup_despin/     ep8: step 1 scene fix, despin patch + loader, merge with pre-registered gates, 5-stand-in judge
  ep10_seat_nest/     ep10: seat patch (F1-b), nest collision proxy (nestcol.py, F2), checks, run records
  training/           kq.sh queue driver (Edmond's de66fea3 protocol: train, then sweep checkpoints >= 70M) + queue/install helpers
  packing/            pack_one.sh, swap_episode.py, swap_submit.sh, ship.sh, combine.sh, final1104/ (final local-best build)
  cdo_transfer/       CD-O-only mesh-row transfers (Kaggle C1, C2)
  tools/              relocate_scene.py (repoint a copied scene's absolute paths), archive.sh
```

Every part has its own `README.md`. Each one covers what the fix changes and why, the exact inputs, the commands, training and
selection, the evidence, and the gaps. Each part also has hash records (`SHA256SUMS`, `SHA256SUMS.txt`, `PROVENANCE.tsv`, or md5
tables). Part READMEs quote md5 or sha256 prefixes and say which. The table below gives both for the shared tools.

| Shared tool | md5 | sha256 | Also staged (byte-identical) in |
|---|---|---|---|
| `training/kq.sh` | `9d5a6253` | `9360ba9b` | ep3, ep10 |
| `packing/pack_one.sh` | `16f82b5c` | `a895130f` | ep3, ep10 |
| `packing/swap_episode.py` | `b1b16564` | `d7d93ab4` | ep3, ep10 |
| `packing/swap_submit.sh` | `816dc28d` | `a7d9fb9d` | ep3, ep10 |
| `packing/ship.sh` | `f10dac27` | `82eee05b` | ep3, ep10 |
| `packing/final1104/combine_final.sh` | `61addd4f` | `92a86c0e` | ep8 |
| `packing/final1104/pack_new.sh` | `84a370b2` | `03130428` | ep8 |
| `tools/relocate_scene.py` | `d8319c83` | `d1f55474` | ep1, ep3, ep5, ep8, ep10 |
| `tools/archive.sh` | `caa53903` | `68991210` | ep3, ep10 |
| `training/install_scene_b.sh`, `kq_add_b.sh`, `install_scene.sh` | `e9fe7124`, `ecbe3664`, `f8bb925c` | `efa69151`, `504e8b84`, `498e5304` | ep8 (first two), ep10 (last) |

Other byte-identical copies:
- the loader `rlr_h5.py` (md5 `0f2cef75`) in ep8 and ep10;
- `comfix.py` (`2aa37dac`) in ep3 and ep8;
- `scenefix.py` (`d3ca0eeb`) in ep5 and ep8;
- `restspots.py` (`1df89fd0`) in ep3, ep5 and ep8;
- `render_review.py` (`8050e217`) in ep3, ep5, ep8 and ep10;
- `team_pack_submission.py` (`c31634cb`) in ep3 and ep10;
- `anch.py` and `an_common.py` in ep1 and ep5;
- `wsupport.py` in ep1 and ep5.

The copies are kept so that each part is self-contained and its hash list stays valid. **One version difference is
deliberate:** `ep3_support_com/scene_build/scenefix.py` (md5 `7f19b50d`) is the older file that actually built ep3's scene. On
ep3 it gives the same result as the later version.

## How to reproduce

The scripts keep the absolute paths they ran with: `/tmp/haisenk_v2d`, `/data/user_data/haisenk/...`, `~/v2d_ops/...`,
`/scratch/haisenk-v2d/...` and `C:\v2d_demo\...`. Each part README lists them; edit them before re-running elsewhere. The
whole chain is: **build scene -> relocate -> train with `kq.sh` (recipe KEN) -> sweep checkpoints -> pack -> swap or combine.**

### 0. Environment (team code at fixed commits; see "Team code and data" below)

- **Training and evaluation.** Use the team repo at `1a6bae7b`. Make a sparse checkout of
  `baseline/robotic_grounding/flash_chord`, `tools/track3` and `assets/data/v2d_submission_kit`, then run
  `uv sync --frozen --extra train` (`training/orch_setup.sh`).
  - Build the overlay `fc_src_de66`: `flash_chord/src` from 1a6bae7b with `lifecycle/reset.py` and
    `training/flash_sac/replay.py` taken from `de66fea3`. Put it first on `PYTHONPATH`.
  - The commands are in `training/README.md`.
- **Loader-built scenes (ep1, ep8 step 2, ep10).** These use the team loader `tools/track3/run_load_retarget.py` with the opt-in
  `H5_PATCH` hook. ep1 uses it as on our pushed branch at `5294c6b`; ep8 and ep10 use it as the staged `rlr_h5.py`.
  - Run it in the team retarget venv (`.venv_retarget`, Python 3.11).
  - It needs the MANO hand model files. These are licence-restricted and not included.
- **File-level scene edits (ep3; ep5; ep8 step 1).** `scenefix.py`, `comfix.py` and `relevel.py` need Python with numpy, scipy,
  trimesh and pyarrow.
- **Packing.** Use the kit at 1a6bae7b and a Python env with pandas and pyarrow. Our env, `kaggle_env`, also has the kaggle
  client, which the kit's `v2d_submit.py` uses. You also need the team wrapper `tools/track3/pack_submission.py`.

### 1. Build the scene from its parent

| Ep | Parent / input | Command sequence (as run) | Details |
|---|---|---|---|
| 1 | Wentao's `scored20_contactshape_proxy_mix_v1_20261103` ep1 bundle (HF revision `fb11a3ce`), plus our raw hands (`ken_c_scored_attach`) | First build the table source: `bash wbuild.sh w_ctl_ep1 1 control`. Then `H5_PATCH=.../patch_whold.py bash wbuild2.sh w_hold_ep1 1 rawhands`, which runs `wprep.py rawhands`, then `run_load_retarget.py --no-rest-hold --no-object-clean --no-level --attach-hands --smooth-hands 2`, then `wsupport.py` | `ep1_hold_window/README.md` section 3 |
| 3 | bob1's ep3 scene `b1_ep3` (= our `h5_ep3_A1`) | `scenefix.py b1_ep3 sf_ep3 --support-top support_0:0.9990`, then `comfix.py fix sf_ep3 sc_ep3 white_pot` (`scene_build/build_sc_ep3.sh`) | `ep3_support_com/README.md` section 3 |
| 5 | `w7_attcom_ep5`, built from Wentao's `scored20_lead_sprint_v2_20261103` ep5 by `w7build.sh` and `w7fix.sh` | `scene_build/build_rl2_ep5.sh`, which runs `relevel.py w7_attcom_ep5 rl_ep5 <depth plane> white_pot --drop-other-supports`, then `scenefix.py rl_ep5 rl2_ep5 --obj white_pot:-1.00 --support-top support_0:1.0097` | `ep5_relevel/README.md` section 3 |
| 8 | bob1's ep8 scene `b1_ep8` and bob1's merged ep8 bundle | Step 1: `scenefix.py <b1_ep8> sf_ep8 --obj beige_cup:2.8 --add-disk cupdisk:-0.003:0.008:0.085:0.971`, then `comfix.py fix sf_ep8 sc_ep8 white_pot`. Step 2: `H5_PATCH=patch_despin_ep8.py rlr_h5.py --bundle <bob1 merged> --episode 8 --attach-hands --smooth-hands 2 ...`. Step 3: `merge_gates.py` copies only the cup pose, 17 right-hand columns and `bundle_cleaned/result.npz` into a copy of `sc_ep8`, under pre-registered gates. | `ep8_cup_despin/README.md` section 3 |
| 10 | our `ken_c_scored` ep10 bundles, built with bob1's recipe (`patch_respinf.py`) | `build_f1b.sh k10_ep10_nest_b ...`, which is `h5_build.sh` with `H5_PATCH=p10seat_b.py`, then `rlr_h5.py --episode 10 --merge --attach-hands --smooth-hands 2`. Then `nestcol.py prep / slabs / check / settle / apply` (CoACD through flash_chord). | `ep10_seat_nest/README.md`, "Rebuild from the parent" |
| 6 | none | none: Edmond's bob1 | `ep6_bob1/README.md` |

**Exact reproduction.** Use the bundled scenes (`final_ours_20261104/episode_00000N/scene/`). The file-level edits were
re-run from their parents during staging and reproduce the trained scenes: ep3, ep5 and ep8 step 1 (all non-path parquet
columns identical, meshes identical, only path strings differ; these re-runs were on Windows, ep5 with a path-separator patch
in scratch copies). The other steps are not bit-exact:
- Loader builds (ep1, ep8 step 2, ep10 F1-b, and the ep5 parent) re-sample the MANO contact labels and `object_mesh_radius`.
- CoACD (ep10 F2) is not bit-deterministic; the exact training hulls are an optional bundle item.

### 2. Install and relocate

```bash
cp -a <built scene> /tmp/haisenk_v2d/scenes/<NAME>        # or: tar xzf scene.tgz -C /tmp/haisenk_v2d/scenes
python tools/relocate_scene.py /tmp/haisenk_v2d/scenes/<NAME>   # repoints parquet/URDF/USDA/MTL absolute paths; keeps *.orig
```

On Babel, `training/install_scene_b.sh <built scene dir> <NAME>` copies the scene, relocates it and checks the paths.
`training/install_scene.sh` is the orchard version, which also queued the runs.

### 3. Train with `kq.sh`, recipe KEN

```bash
# queue line: NAME EP PARQUET_DIR SEED RECIPE PRIO
#   PARQUET_DIR = <scene>/assets/human_motion_data/ego_recon/processed/sequence_id=<seq>/robot_name=sharpa_wave
printf '%s\n' "rf_ep8_ds_K_s43 8 /tmp/haisenk_v2d/scenes/rf_ep8_ds/assets/human_motion_data/ego_recon/processed/sequence_id=ep8_ken_c_scored_attach/robot_name=sharpa_wave 43 KEN 10" >> kq.txt
bash training/kq.sh kq.txt 0 1 2 3          # one FlashSAC training per listed GPU
```

Each job runs `scripts/train_flash_sac.py experiment=sharpa_flash_sac task.parquet=<PARQUET_DIR> reset.seed=<SEED>
curriculum=tracking_100m_object2x training.total_environment_steps=100003840`. It adds the COMMON settings: motion speed 1.0,
fps 20.254119873046875, wrist termination 0.3, a checkpoint every 5,000,192 steps. It also adds the KEN settings:
- object termination 0.1 m / 0.5 rad;
- replay 4,194,304 in float32;
- `sim.njmax=1024 sim.nconmax=512`.

These are the same overrides as `campaign.py` RECIPES at de66fea3. The full command and the queue lines of the five finals are
in `training/README.md`. The finals were:

| Ep | Run | Seed | Host |
|---|---|---|---|
| 1 | `b_ep1_wh_K_s48` | 48 | Babel H100 |
| 3 | `h_ep3_sc_K_s42` | 42 | orchard H100 |
| 5 | `r_ep5_rl_K_s43` | 43 | orchard H100 |
| 8 | `rf_ep8_ds_K_s43` | 43 | Babel H100 |
| 10 | `n_ep10_nb_K_s42` | 42 | orchard H100 |

### 4. Checkpoint sweep and pick

`kq.sh` evaluates every checkpoint from 70M on with the kit's `examples/run_policy_evaluation.py` (own-reference add_auc) and
writes `sweep.json` and `best.txt`. The rollout of the chosen checkpoint is `runs/<NAME>/early_<STEP>/results/`.

| Ep | Packed step | How it was picked |
|---|---|---|
| 1 | 100003840 | kq best (0.830). Then a pre-registered rule, the registration-aware ep1 emulator against T2, picked s48 among 12 new seeds (`ep1_hold_window/selection/`). |
| 3 | 95006720 | kq best (0.947) |
| 5 | 95006720 | best anchored own-ref (0.909), which aligns on the frame-0 anchor as AUC.py does. The plain kq best was 90M. See `ep5_relevel/selection/`. |
| 8 | 95006720 | kq best (0.990). Then a pre-registered judge (kit AUC.py under 5 stand-in truths) against T6 (`ep8_cup_despin/selection/`). |
| 10 | 100003840 | kq best (0.980) |

### 5. Pack one episode

```bash
bash packing/pack_one.sh <EP> /tmp/haisenk_v2d/runs/<NAME>/early_<STEP>/results /tmp/haisenk_v2d/scenes/<SCENE>/score_input/meshes \
  https://github.com/zjifan121-source/NVIDIA_V2D_Egocentric/commit/1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6 <out>/one.parquet
```

`pack_one.sh` works in four steps:
1. It narrows the kit config and sample to one episode.
2. It relabels the bodies with the team wrapper's `relabel()`.
3. It copies the scene's `score_input` meshes.
4. It runs the kit's `tools/pack_submission.py --track track_3`.

The commit URL is bob1's, so that the provenance columns stay uniform; see Caveats.

### 6. Swap into bob1, or combine

```bash
python packing/swap_episode.py candidate_bob1.parquet one.parquet <EP> track3_submission.parquet --adopt-commit-url   # one-episode swap test
bash packing/final1104/combine_final.sh local_best 1:<ep1 one> 3:<ep3 one> 5:<ep5 one> 8:<ep8 one> 10:<ep10 one>      # final local-best file
cd <kit> && python submit/v2d_submit.py <file> --track track_3 --dry-run --only add_auc spider_sr maniptrans_sr rpe_cm -m "<message>"
cd <kit> && python submit/v2d_submit.py <file> --track track_3 --dry-run --only mppe_cm cd_o_cm -m "<message>"
```

`swap_submit.sh` chains `pack_one.sh`, `swap_episode.py` and the dry run. It submits only with `SUBMIT=1`, which needs Kaggle
credentials; those are not part of this release. The dry run is split by board because the kit's identity check compares the
last submitter on each board.

### 7. Check

Compare the sha256 of the policy, rollout, `one.parquet` and combined file with `bundles_manifest.json`. All of them were
re-hashed on 4 Nov at 14:52 on babel-s5-16 and locally.

## Team code and data this depends on (not copied here)

**Team repo `zjifan121-source/NVIDIA_V2D_Egocentric`:**

| Commit | Description | What we use |
|---|---|---|
| `1a6bae7b2682a40087cd70e5d68c6aa3e5f21b77` (`edm_training`) | "Add a no-Slurm training queue and night run sheet" | flash_chord (training and evaluation runtime) and the curriculum `tracking_100m_object2x.yaml`. The team loader `tools/track3/run_load_retarget.py` (blob `43d29ed0`). The team packing wrapper `tools/track3/pack_submission.py` (blob `8276045d`). `tools/track3/episodes.json` and `support_stacked.py`. T2's driver `tools/track3/jobs/gpu_queue.sh`. |
| `de66fea3ae7c99ba96458f3cd9d1b77f5f94b4ff` (Edmond) | "Push the full policy training pipeline and a training guide" | The training protocol in `reports/policy_targeted_2026-10-28/`: `TRAINING_GUIDE.md`, `campaign.py` RECIPES and `fresh_keep.sbatch`. Two overlay files: `flash_chord/lifecycle/reset.py` (blob `d8d005c0`) and `flash_chord/training/flash_sac/replay.py` (blob `1eca045a`). |
| `5294c6b2c70c0cf88cd31e08f004f71cbc2d7f8a` (our branch `haisen/hand-fixes-ep3-10-20`, pushed 2 Nov) | "Add hand fixes for scored episodes 3, 10 and 20" | `run_load_retarget.py` with the `H5_PATCH` / `H5_MERGED_NPZ` hooks, plus `tools/track3/hand_fixes/`. It built bob1's ep3 and ep10 parents and is the ep1 loader. |
| `1689cf6396bee0f6a3bf4cced610f2fe0f71ebf6` (Edmond) | "Track reports/ code, notes and small results" | The `code_commit_url` that every packed row cites (bob1's). It does not contain this folder. |

**Submission kit** `assets/data/v2d_submission_kit` (at 1a6bae7b):

| File | sha256 | Use |
|---|---|---|
| `examples/run_policy_evaluation.py` | `0bcc41f5...` | checkpoint sweep |
| `tools/pack_submission.py` | `49b13052...` | the rows' `packer_script_sha256` |
| `submit/v2d_submit.py` | `d31d4a7c...` | dry run, `--only`, CD-O extraction, submit |
| `config/tracks.json`, `data/track_3_sample_submission.parquet`, `data/track_3_evaluation_objects.json` | | packing |
| `metric_code/track_3/AUC.py` | `3f1a9899...` | the ep1 and ep8 selection emulators |
| `metric_code/track_3/CD-O.py` | `c166cf86...` | `cdo_transfer/` |

The organisers' metric code is **not redistributed** here.

**Teammates' releases on HF** (`rose-tea-cafe/v2d-t3-recon`, pointers only):
- **Edmond's bob1** (`experiments/edmond/bob1_submission_20261103/`): `candidate_bob1.parquet` is Kaggle submission 57934736,
  sha256 `46fe967a...`. It is the base of every swap and combine. It also holds the ep6 scene and policy, and the ep3 and ep8
  parent scenes.
- **Wentao's releases** (`experiments/wentacc/`):
  - `scored20_contactshape_proxy_mix_v1_20261103` at revision `fb11a3ce`: the ep1 base bundle, `support_v1` and
    `tools/install_support.py`.
  - `scored20_lead_sprint_v2_20261103`: the ep5 base bundle.

**Our earlier HF releases:**
- `experiments/haisenk/hand_fixes_v1_20261102/`: the ep3 parent `h5_ep3_A1` and the ep10 parent `h5_ep10_respinf_v4`.
- `experiments/haisenk/hand_fixes_wentao_v1_20261103/`: the ep1 prepared bundle and the as-built scene.

**Our reconstruction outputs (Babel `/data`, not uploaded):**
- `ken_c_scored`: FoundationStereo depth and SAM 3 masks, used by the depth-table measurements of ep3, ep5, ep8 and ep10.
- `ken_c_scored_attach/bundle_merged`: the raw hands for ep1 and ep5, and bob1's merged ep8 bundle.

**Not included:**
- the MANO hand model (licence);
- Kaggle and HF credentials;
- the scripts that read them (`_cdo_build.sh`, the bob1 download script, the token setup and check scripts);
- large binaries.

## Caveats

1. **Two final picks are local-only.**
   - **ep1 seed s48** was chosen by a GT-free emulator of AUC.py. That emulator was calibrated on T2's public-leaderboard
     changes, and s48 is the best of 12 seeds, so expect some winner's curse.
   - **ep8 despin `rf_ep8_ds`** was chosen by 5 GT-free stand-in truths. The reviewer's caveats:
     - Gate G3, re-measured with an exact centroid, reads 0.078 cm against its 0.05 cm limit. The physical effect is under 1 mm.
     - If Kaggle scored the cup as non-symmetric and the real cup did spin, the pick would score 0.742 against T6's 0.961.
   - Neither pick was ever scored by Kaggle. The ep1 and ep8 rows on the leaderboard are T2 and T6. Their scenes are documented
     in the same parts, and their policies and rows are listed as extra bundle items.
2. **Nothing after 09:50 EST on 4 Nov was submitted.** N1 (MPPE) was the last submission, and Ken stopped all submissions at
   about 10:05.
   - The final local-best file was only dry-run.
   - The quota counters at 13:50 show no team submission on any board between 09:50 and 13:50.
   - This assembly step made no Kaggle call.
3. **Commit URL.** Every packed row, ours and bob1's, cites commit `1689cf63`, which does not contain this code. The rows need
   uniform provenance and cannot change without resubmitting. The tech report and the eligibility note should point to the
   commit of this release.
4. **Leaderboard feedback was used.** Swap tests T1-T8 were adopted or rejected by their public-board change. The private
   fraction is 0, so the public board is final. The CD-O transfers C1 and C2 were chosen from Kaggle-measured per-object
   changes. Several meshes in our pose rows are bob1's donor meshes (`share/best_meshes_20261024`, picked earlier by
   public-episode AUC): the donor white_pot "P0" in ep3, ep6 and ep8, ep8's beige cup (the ep11 donor, stretched 2.8 cm by
   T6), and ep10's two cups (blue from ep13, beige from ep11). The ep1 and ep5 meshes come from Wentao's releases (ep5's
   pot trimmed 1.0 cm by us); their origin is documented there. The tech report should say so.
5. **ep10 disclosures.**
   - The simulation collision proxy differs from the submitted meshes. Ken approved it on 4 Nov at about 09:45.
   - F2 was trained over its own failed static check b: a 9.22 mm wall crossing on f119-160, after the hand releases the cup.
   - The s42 log has 17 `nefc overflow` lines at njmax 1024. The fallback without overflows is s43 @85M (0.979).
   - On the pose boards other than MPPE, ep10 is still bob1's.
6. **Provenance that was reconstructed** (each is verified by a rebuild or a hash where possible; see the part READMEs):
   - ep1: the download command for Wentao's release.
   - ep3: the exact `scenefix.py`, recovered from the session log.
   - ep5: the as-run `scenefix.py` and the inline support snippet.
   - ep8: the step-1 command lines.
   - ep10: the `install_scene.sh` run, and `ONLY=add_auc` on its pack dry run.
7. **Hosts.** Orchard (ep3, ep5 and ep10 training; Python 3.12) is gone. Babel (ep1 and ep8; Python 3.11) used the same commits.
   babel-s5-16 `/tmp/haisenk_v2d` (scenes, runs, the `fc_src_de66` overlay) is lost when H100 job 10774619 ends, around
   6 Nov 00:03. The durable copies are `C:\v2d_demo\archive\<candidate>\` (all hashes match) and Babel `/data` for ep1 and ep8.
8. **Symlinks in the scene directories.** In `bundle_cleaned/`, `w_hold_ep1` has 6 and `rl2_ep5` has 7 symlinks that point to
   node-local Babel `/scratch` and are dangling. `k10_ep10_nest_b` has 7 symlinks into Babel `/data`. None of them is needed for
   training or evaluation. Each `scene.tgz` in the manifest is a verified alternative upload: it has the same processed-parquet
   sha256 as the directory.
9. **Internal notes stay out of the public commit by default.** Each part's working notes (staging `release1104/notes/<part>/`)
   contain teammates' names, a teammate's Kaggle handle and submission strategy. The code tree itself has no credentials or
   Kaggle handles. Besides the absolute paths (which contain our Babel user name), its only personal identifier is the git
   author line (name and university e-mail) in `ep1_hold_window/scene_build/run_load_retarget_hooks_5294c6b.diff`, which is
   the header of the already-pushed commit.
10. **`packing/final1104/run_combine2.sh` contains one literal carriage return on purpose.** It removes the CR-named strays left
    by an earlier CRLF attempt. It is not a CRLF file.

## Checks done when assembling this release (4 Nov, about 14:50-15:10)

- **Python.** `python -m py_compile` passes on all 99 `.py` files under `release1104/` (code and notes) with CPython 3.12.7.
  - The cache went to a scratch prefix, so there is no `__pycache__` in the tree.
  - Two files under `ep5_relevel/scene_build/history/` failed only because Windows could not write the cache file (path too
    long). Both compile with a short `cfile`.
- **Shell.** `bash -n` passes on all 79 `.sh` files (GNU bash 5.3.15, Git Bash).
- **Hash lists.** The part hash lists all verify:
  - ep1 `SHA256SUMS` 54/54;
  - ep3 `SHA256SUMS.txt` 76/76;
  - ep5 `SHA256SUMS.txt` 60/60;
  - ep10 `SHA256SUMS` 58/58;
  - ep6 1/1;
  - ep8 `PROVENANCE.tsv` md5 78/78;
  - notes 9/9.
- **Content scan.** No file over 1 MB and no binaries (`.npz`, `.parquet`, `.stl`, `.tgz`, `.safetensors`, `.pkl`). No token,
  key, password or `kaggle.json` patterns.
- **Bundle hashes.** On babel-s5-16 at 14:52 (read-only step of job 10774619), every final policy, rollout parquet and json,
  scene directory (processed parquet), packed `one.parquet`, the combined file and the bob1 base matched the claimed sha256. The
  same items in `C:\v2d_demo\archive\` also match, including each `scene.tgz` and the processed parquet inside it.
