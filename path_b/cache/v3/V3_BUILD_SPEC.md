# Path B v3 build spec (single-object chain; dev episodes ep40 + ep26)

Written 2026-10-21 for a coding session. Companion overview: artifact "V2D ep41 Pipeline"; background: Project doc `claude/v2d-e2e-research-2026-10-21.md`. Decisions from James (2026-10-21): rerun everything from the video (no bundle reuse); keep the Path B layout (`path_b/run.py`, `pathb/stages/*.py`, one `report.json` per stage, `loss_report.md` per episode); keep only the v2 code v3 uses, park the rest in `path_b/cache/`; one Verda L40S (48 GB). **Dev episodes: ep40** (public, `mini_dust_pan`, the median single-object episode) **and ep26** (public, `plastic_dish_rack` + `white_pot`, the median two-object episode); ep41 is not rerun. **Build phase 1 = the single-object chain only**, run on ep40 and, per object, on ep26 (each object of ep26 goes through S1–S6 independently with the shared camera; the scene is assembled for scoring with the existing merge in `run_load_retarget.py --merge`, mode A). Then all episodes. **Phase 2** (after all episodes have run): the two-object scope (§8) is built and the two-object episodes are rerun and compared. SAM 3.1 access is granted (`facebook/sam3.1`, user `JzJz7779`). Workflow (§9, 2026-10-22, GitHub rule changed 2026-10-24): local-first; cloud only for dev-episode GPU smoke tests and all-episode runs; every passing smoke test is pushed to `zjifan121_pathb_v3` at once, no merge to `main_buffer` without James.

Every interface below was read from the code on 2026-10-21 (`main_buffer` @ `e61fc18`; local `zjifan121`; baseline `8c778f9`; kit in `assets/data/v2d_submission_kit`). Where a parameter is a proposal, it says "default:" and the acceptance test decides.

**Revision 2026-10-24 (James, after the phase-A/A2/ep40 review; artifacts "Path B v3 Review", "Path B v4 Design", "Path B Task Queue").** (1) Items that did not improve the score are removed from this spec: MV-SAM3D and SAM 3D as the mesh source (S3a is now the A2 canonical mesh), the `tracking` candidate, the full-MPC S4c settings, the A2.4 mesh-free tracker, the §8 two-object scene code. (2) §9 GitHub rule changed: push to `zjifan121_pathb_v3` immediately after every passing smoke test, before any single- or all-episode run; no PR/merge to `main_buffer` without James. (3) §12 adds v4 (contact-segmented inversion + policy-side changes), built on the Mac while phase B runs. (4) Hands stay HaMeR; EgoFound3R is not used (§7). (5) §6 run plan rebuilt around the Nov 1 stop with the cuts: one S4c candidate, two policies, S4c off until it carries the object.

## 0. Rules

1. Never change anything under `baseline/` or `assets/data/v2d_submission_kit/`. Baseline modules are called as they are (host wrappers launch the GHCR containers; `pathb/paths.py` → `HOST_PY`, `RECON`).
2. Ground truth and public scans are read only inside `score_local` and inside stages that check `ctx.split == "public"`. With `--split evaluation` any such read raises (GT firewall, §7).
3. Every stage writes `report.json` through `pathb/report.py` (`Metric(name, value, unit, kind, ok=(lo,hi), warn=(lo,hi), note)` → status PASS/WARN/FAIL/INFO; `StageResult(metrics, outputs, skip)`); `pathb/runner.run_stage` adds stage, status, episode, split, commit, time, GPU peak. `loss_report.md` names the first stage whose status is FAIL or ERROR.
4. A stage is rerun only with `--force` or when its report is missing or ERROR (`runner.run_stage`).
5. One heavy model on the GPU at a time. Stages are subprocesses into their own environment (§2); `pathb` itself imports only numpy/scipy/cv2/trimesh.

## 1. Repository changes (first commit of the session)

Move to `path_b/cache/v2/` (git mv, keep history): `pathb/spider_scene.py`, `pathb/spider_settle.py`, `pathb/stages/s4_scene.py`, `tests/test_spider_scene.py`, `PATH_B_V2_PHYSICS_INVERSION.md`, `pathb/merge_runner.py` (two-object; future). Remove `"s4_scene"` from `runner.STAGES`.

Keep: `run.py`, `pathb/{runner,report,paths,calib,geometry,bundle,team,loss,waterfall,visualize}.py`, `visualize_page.html`, `stages/{p0_ingest,p1_calibrate,p2_depth,p3_scale_bridge,p4_export,s23_load_retarget}.py`, `tests/{test_calib,test_geometry,test_report,test_rows,test_scale}.py`, `vm/`, `README.md` (rewrite §"Stages").

New `runner.STAGES` (order is execution order; names are module names in `pathb/stages/`):

```
p0_ingest  p1_calibrate  p2_depth                       (exist)
p1b_masks  p1c_camera  p1d_hands                        (S1 new/changed)
p3_scale_bridge  p4_export                              (exist; P3 reads p1c camera, not the bundle)
s3a2_fuse_scan  s3a2_register  s3b_track  s3c_fuse      (S3; s3a_mesh removed 2026-10-24, see §4 S3a and §11)
s4a_phases  s4b_reference  s4c_retarget                 (S4; candidates keyframe + still)
s3b_segments  s3c_invert  s3d_settle                    (§12 v4; inserted after s3a2_register when built)
s5_task                                                 (S5)
s6_train  s6_eval  s6_select                            (S6)
s7_pack                                                 (S7, all episodes; runs from run.py pack)
s23_load_retarget                                       (kept for the still fallback; see S6)
```

`run.py` gains `pack` (S7) and `oracle` (S0) commands beside `run`, `waterfall`, `visualize`. `waterfall` gains columns `keyframe`, `realised`, `rollout_<candidate>`.

## 2. Environments

| Env | Python | Used by | Setup |
|---|---|---|---|
| `path_b/.venv` (uv) | 3.11 | all `pathb` code, P0–P4, S3c, S4a/b, S5, S6-select, S7 | `uv sync --project path_b` |
| baseline host venv `baseline/reconstruction/.venv` + GHCR images (`tools/verda/images.lock`) | 3.11 | FoundationStereo, DROID-SLAM, SAM2, WiLoR/HaMeR, SAM3D, FoundationPose (TensorRT), EKF | exists on the VM (`setup.sh`) |
| `sam3` venv (new) | 3.12 | S1b SAM 3.1 (`facebook/sam3.1`, gated) | fallback: SAM2 container with Grounding DINO box, as the baseline. **2026-10-23: the fallback fired silently once (API mismatch); P1b must FAIL, not fall back, when SAM 3.1 does not load.** |
| `daid` venv (new) | 3.12 | S4c Do as I Do retargeting; **§12 S3d MuJoCo settle** | github.com/malik-group/do-as-i-do `retargeting/` (`pip install -e .`; MuJoCo Warp, mink, viser, coacd); ≥32 GB GPU for S4c; S3d is CPU MuJoCo |
| `flash_chord` venv | 3.11 | S6 train/eval, S5 replay | exists (`/root/v2d/venvs/`); MuJoCo 3.8.1 / mujoco-warp 3.8.1 / warp 1.14 |
| `tools/.venv` | 3.11 | `score_local`, `run_load_retarget`, kit packer/uploader | exists |

Do as I Do and FlashCHORD pin different MuJoCo Warp versions: separate venvs, files only between them.

## 3. Conventions (all stages)

- Frames: colour camera `ego_cam_a`, 0-indexed, `%06d`. Stereo frames are offset by `alignment.json: offset` (P2); every stage that mixes stereo and colour applies it once, through `pathb.calib.stereo_frame(i)`.
- Camera frame: OpenCV (+x right, +y down, +z forward), metres. World: table top z = 0, +z up (`p4_export: table_z_world_m` becomes 0 by construction in v3), +y = mean camera forward projected on the table.
- Poses on disk: `Transform3d` JSON (`rotation` wxyz, `translation`, `scale`) per frame, as the baseline writes; arrays in memory and in `.npz` as (N, 4, 4) float64 `*_to_world`. Quaternions in Kaggle/evaluator parquets are **xyzw**.
- Depth PNG: uint16 inverse depth `px = 65535/(d+1)` (`v2d_common.datatypes.DepthImage`).
- Masks: 8-bit PNG, 255 = object, one folder per track.
- Hands: MANO parameters per frame in the baseline's `hamer_aligned_filled/<track>/%06d.json` layout; world joints computed with `tools/track3/read_mano.py` (applies `hand_*_scale`).
- `report.json` bars: PASS/WARN ranges live in the stage file next to the Metric; copy the existing P0–P4 style.

## 4. Stages

### S0 — Oracles

**S0.2 Regime split: done.** Edmond's `oracle_2026-10-17.json` (20 public episodes, 21 variants, Oct 17 bundles: cleaned, merged, no stereo scale) answers it. Means over the 20 episodes:

| Variant (what is replaced by truth) | AUC | MPPE cm | Reading |
|---|---|---|---|
| `place` (our scene as submitted) | 0.388 | 12.4 | base |
| `still_gt0` (true frame-0 pose, never moves) | 0.447 | 9.3 | the still ceiling |
| `place_scale` (true world scale) | 0.488 | 7.7 | scale alone +0.10 |
| `place_rot_gt` (true rotations, our positions) | 0.404 | 12.1 | rotation is secondary |
| **`place_pos_gt`** (true positions, our rotations) | **0.741** | **1.9** | position error is the loss |
| `place_gt_when_resting` (truth on rest frames, ours when moving) | 0.625 | 7.5 | rest regime worth +0.24 |
| `place_gt_when_moving` (truth on moving frames, ours at rest) | 0.762 | 5.0 | moving regime worth +0.37 |
| `recon_hold_gt` (our pose held at the true rest times) | 0.337 | 11.1 | holding a wrong pose gains little |

Per object (33): mean `gt_moving_frac` 0.445 (55 % of frames at rest); our position error 11.2 cm at rest vs 16.2 cm moving; camera/common-mode RMS while both objects rest (2–10 cm) is as large as the per-object RMS (0.3–10 cm). Policy rollouts on ep41 (the only trained episode): still_level 0.356, attach 0.351, scan_level 0.242, 500M-step run 0.142 (MPPE 79 cm: the object is knocked away). Dev-episode baselines from the same file: **ep40** recon 0.34, still 0.40, true-still 0.40, moving fraction 0.68, position error 2.2 cm at rest / 13.0 cm moving; **ep26** scene 0.45, still 0.50, true-still 0.67, pot moving fraction 0.17 (dish rack static), position error 11.2 / 16.2 cm, camera RMS at rest 4.6 cm. Conclusions used below: (1) anchors matter only if they are accurate (today's rest pose is 11 cm off, so P1c + S3c anchors from stereo are the prerequisite); (2) the moving regime holds the larger headroom, so `tracking` stays a candidate; (3) **S4 starts with the `keyframe` candidate** (James, 2026-10-21): rest anchors + interpolated carry attack the dominant position error and give the policy a path it can execute; (4) more RL steps do not fix a bad reference.

**S0.1 GT-reference policy: still open; runs on both ep40 and ep26 (James, 2026-10-21).** Build `task.parquet` for each with the *ground-truth* object trajectory in place of ours and the team bundle's hands, through `run_load_retarget.py` (bundle with `object_to_world_transform` = GT; the public scan as mesh for this test only). Train with the `still_submit.sh` settings (60M, compressed VOC), evaluate with the kit, score with `score_local`. **Decision rule:** if rollout(GT reference) < the episode's still score (ep40 0.40, ep26 0.50), the S6 recipe (VOC decay, hand-term weights) is fixed before any S3 work beyond anchors; if ≥ 0.7, the reference is the whole problem and S1–S4 get the time.

### S1 — Evidence

**P0, P1, P2** unchanged (`stages/p0_ingest.py`, `p1_calibrate.py`, `p2_depth.py`). Outputs used downstream: `p1_calibrate/{rect_left,rect_right,cam_a}/`, `calib.json`; `p2_depth/{depth_rect,intrinsics_rect,planes.json,alignment.json,depth_a_half/}`. Known bars: row offset < 0.5 px; stereo offset found (1 frame on ep41; measure on ep40/ep26).

**P1b masks** (`stages/p1b_masks.py`, new)
- Wraps: SAM 3.1 (text prompts) in the `sam3` venv; fallback the baseline SAM2 container with Grounding DINO (`v2d_grounding_dino`, `v2d_sam2`) exactly as `run_ego_wilor.py` steps 5–6, but seeded on the frame P1b picks.
- In: `cam_a/%06d.jpg`, roster prompt (`episodes.json` → `objects[].prompt`), hand prompts "left hand", "right hand".
- Out: `p1b_masks/obj_0/%06d.png`, `hand_L/`, `hand_R/`, `prompts.json` (prompt, seed frame, box, model), `masks_overlay.mp4`.
- Parameters: seed frame = first frame where the object box confidence ≥ 0.35 and the mask area ≥ 0.5 % of the image (default); SAM 3.1 propagation over the whole clip, both directions.
- Acceptance: object mask present in ≥ 95 % of frames (PASS), ≥ 80 % (WARN); hand masks present whenever WiLoR detects a hand (P1d) in ≥ 90 % of those frames; render check on 8 frames written to `overlay/`.

**P1c camera** (`stages/p1c_camera.py`, new)
- Wraps: baseline `v2d_droid_slam/lib/video_to_slam.py` with `--input_depth_folder` (depth prior into bundle adjustment) **and** `--align_to_depth_folder`, both pointing at a masked copy of `depth_a_half/` (full-resolution reprojection if memory allows). Masking: write `p1c_camera/depth_prior/%06d.png` = stereo depth with hand and object pixels set to 0 (DROID treats 0 as no prior). No code change inside the baseline.
- In: `cam_a/` video (re-encoded from frames by P1 or the undistorted mp4), `calib.json` K for cam_a, `depth_prior/`.
- Out: `p1c_camera/camera_to_world/%06d.json`, `trajectory.txt` (TUM), `camera_report.json`.
- Acceptance (all computable without GT): path length vs stereo-only expectation (median per-frame speed < 1.5 m/s, PASS); reprojection of the P2 table plane across frames: plane height spread over the rest window < 1 cm (PASS) / 2 cm (WARN); scale check: DROID keyframe depth vs stereo depth median ratio within 3 %. Public only: compare against the bundle's camera path in `numeric_qa`.

**P1d hands** (`stages/p1d_hands.py`, changed)
- Wraps: baseline WiLoR → HaMeR → `align_hands` → `tracks_interpolate` (steps 4, 12–14 of `run_ego_wilor.py`), called through `tools/track3/run_baseline_reconstruction.py --late-hands` machinery, with `depth/` replaced by `p2_depth/depth_a_half` upsampled to full resolution (the alignment step reads `depth/%06d.png`).
- Out: `p1d_hands/hamer_aligned_filled/{2,3}/%06d.json`, `hand_scale.json`, `handedness.json`, `overlay.mp4`.
- Acceptance: per hand, frames with a pose ≥ 90 % of frames with a mask; `hand_scale` within 0.8–1.2 (PASS) / 0.6–1.5 (WARN); wrist speed < 2 m/s on 99 % of frames (from `run_load_retarget.bad_hand_frames`).

### S2 — One metric world

**P3 scale bridge** (`stages/p3_scale_bridge.py`, changed): the bundle-side ruler is replaced by the S1 camera: factor = stereo camera height / P1c camera height over the rest window, **expected 1.00 ± 3 %** because P1c is already metric; the stage keeps the computation as a check and writes `scale.json` with `factor`, `rest_window`, `table_plane` (n, d in world), `gravity`. On ep41 the v2 factor was 0.831 against the *bundle*; against P1c the pass bar is 0.97–1.03 on ep40 and ep26 (WARN 0.9–1.1; FAIL otherwise → P1c is wrong).
**P4 export** (`stages/p4_export.py`, changed): writes the v3 bundle layout (BUNDLES.md fields plus) `result.npz` with `camera_to_world_transform` (P1c), `hand_*` (P1d, world via camera), `table_plane`, `gravity`; no object track yet (S3 fills it). `mesh.obj` absent until S3a.

### S3 — Object model and track

**S3a mesh = the A2 canonical mesh (§11 A2.1–A2.3). Decided 2026-10-24.** The SAM 3D / MV-SAM3D stage (`stages/s3a_mesh.py`) is removed: on the ep40 real run it FAILed (not watertight, 0.78× the scan, CD-O 0.36 cm, 17 min), MV-SAM3D was never installed, and teammates found both view- and seed-dependent. The A2 variant `a2_mesh_fp` won the §11 A2.5 rule on the dev episodes (AUC 0.540 → 0.726 on ep40; CD-O 0.53 vs 0.24 → tie on ranks → deterministic variant). Keep `s3a_mesh.py` under `path_b/cache/` for reference only.
- The per-episode output keeps the `s3a_mesh/` name downstream expects: `s3a_mesh/mesh.glb` is the canonical mesh of the object (metres, canonical frame), `coacd/`, `mass.json` (volume × density table in `pathb/density.py`), `sha256.txt`; `s3a2_register` writes the frame-0 pose in the canonical frame.
- Canonical meshes are pooled over **every clip of the object in both splits (videos only)** before phase B's evaluation episodes run; one clip gave coverage 0.58 and CD-O 0.53 cm on ep40.
- Acceptance: watertight; extents within 0.8–1.25× of the fused partial; public only: CD-O vs scan via `tools/track3/eval_mesh.py` < 0.3 cm (PASS) / 0.5 (WARN).

**S3b track** (`stages/s3b_track.py`, changed)
- Wraps: `v2d_foundation_pose/lib/foundation_pose_tracker.py` (`register`, `track_one`, `track_one_with_recovery`, `track_one_particles`, `reset_to_pose`) run inside the FoundationPose container through a small driver script `path_b/pathb/drivers/fp_track.py` mounted with `--dev`. Inputs: `s3a_mesh/mesh.glb` (the canonical mesh; no scale search: `--skip_object_scale_estimation` equivalent), `cam_a/`, stereo depth upsampled, `obj_0` masks, cam_a K.
- Registration frame: the rest-before frame with the largest unoccluded mask, initialised from `s3a2_register`'s frame-0 pose. Track forward then backward as the baseline. ep40 real run 2026-10-23: WARN (one rotation step), 80–90 s. Under §12, S3b becomes the seed of the first static ICP only.
- **Recovery rule (replaces blind re-registration):** on IoU < 0.3, (a) build the candidate set = previous pose × symmetry group (from `pathb/symmetry.py`: cups → 36 rotations about the axis; `white_pot_lid` → {I, 180°}; others → {I}) plus `register()`'s top-5 hypotheses; (b) refine each with 2 `track_one` iterations; (c) score = FoundationPose scorer logit + 1.0·log(mask IoU) − λ·(rotation distance to the last good pose, rad) − μ·(median |rendered − stereo depth| in the mask, m); defaults λ = 2.0, μ = 50; (d) accept the best; mark the frame `recovered`. `track_one_particles` (n = 20, mask IoU weight 1.0) is used on frames whose hand-mask overlap with the object mask > 40 % (default), i.e. the carry.
- Out: `s3b_track/poses/%06d.json` (object_to_camera), `track_meta.json` (per frame: iou, recovered, particles, depth residual).
- Acceptance: frames with IoU ≥ 0.5: ≥ 80 % (PASS) / 60 % (WARN); max rotation change between consecutive frames < 30° (PASS) / 90° (WARN) outside recovered frames; no frame with > 60° change in the camera frame at all (FAIL if any).

**S3c fuse** (`stages/s3c_fuse.py`, new)
- Pure numpy/scipy. Inputs: S3b poses, P1c camera, S4a phases (helper), stereo depth.
- Rest anchors: for rest-before and rest-after windows, object_to_world = median translation and chordal-mean rotation of the S3b world poses over the window, after rejecting frames whose stereo depth residual > 2 cm; written to `anchors.json`.
- Fusion: the baseline ESKF/RTS (`run_ekf_smoothing.py` parameters: process 0.01/0.01 m, 0.02 rad; measurement 0.01/0.04 m, 0.02 rad, scaled by 1/IoU floor 0.1) re-implemented on **world** poses with two additions: anchor frames use measurement noise 0.002 m / 0.005 rad; recovered frames use 3× noise. Optional Point2Pose poses enter as a second measurement stream with its own noise (0.02 m / 0.05 rad) if `s3b_track/poses_p2p/` exists.
- Out: `s3c_fuse/object_to_world.npz` (N,4,4), `anchors.json`, `fuse_report.json`; and the bundle `p4_export/.../result.npz` is completed (`object_to_world_transform`, `object_to_camera_transform`, `mesh.obj` = S3a mesh).
- Acceptance: rest-before spread < 5 mm / 1° (PASS); object speed < 1.5 m/s everywhere; public only: playback score via `score_local` written to `waterfall` as column `fused`.

### S4 — Reference construction

**S4a phases** (`stages/s4a_phases.py`, new; pure function in `pathb/phases.py`)
- Input: P1d wrists in world, S3b/S3c object poses and mesh. `near(t)` = any valid wrist within 10 cm of the object surface (cKDTree, as `run_load_retarget.reach_frame`), dilated ±5 frames. rest-before = [0, first near); carry = [first near, last near]; rest-after = (last near, N). Objects never touched: whole clip is rest.
- Out: `phases.json` {rest_before:[a,b], carry:[a,b], rest_after:[a,b], first_reach, last_release}.
- Acceptance: carry length ≥ 10 frames unless the object never moves (object displacement < 2 cm → `static: true`).

**S4b reference** (`stages/s4b_reference.py`, new)
- Candidates (2026-10-24): **`keyframe` and `still` only.** `tracking` is removed: it scored identically to `keyframe` on every run (0.540 / 0.726 / 0.381 on ep40) and cost a third of S4c and S6. Under §12 a third candidate `inverted` is added and `keyframe` becomes its fallback. Writes the kinematic references as v3 bundles (`result.npz` + mesh), each a complete input for S4c/S5:
  - `keyframe`: rest-before anchor held until `first_reach`; carry = S3c poses on frames whose `track_meta` shows IoU ≥ 0.5 and not recovered, else interpolation between the nearest trusted frames (position: cubic Hermite with zero end velocity; rotation: slerp); rest-after anchor held from `last_release`. Minimum: the whole carry may be one interpolation from anchor to anchor over the carry window.
  - `still`: rest-before anchor for all frames (the fallback; equals `--still-object`).
- Hands: P1d MANO in the world, cleaned with `run_load_retarget` hand cleaning (import its functions; do not copy).
- Out: `s4b_reference/<candidate>/` bundles; `reference_report.json` (per candidate: frames trusted, interpolated, held).
- Acceptance (public): `score_local` playback per candidate → waterfall columns `ref_keyframe`, `ref_still` (and `ref_inverted` under §12).

**S4c retarget** (`stages/s4c_retarget.py`, new; Do as I Do)
- Adapter `pathb/daid.py`: writes the `trajectory_keypoints.npz` layout that `retargeting/` stage 1 produces (read its writer once in-session: wrist pose convention, fingertip keypoints, object pose per frame, mesh path) from a v3 bundle; copies `coacd/` pieces; the table is the MuJoCo floor at z = 0.
- Run: DaiD stages 2–5 (`decompose` → scene MJCF with `--robot-type sharpa` → IK (mink) → sampling MPC in MuJoCo Warp) in the `daid` venv, **for one candidate only** (`keyframe`; `inverted` under §12); `--max-sim-steps 0` (full trajectory). Settings (2026-10-24): the **cut** settings are the only ones allowed — 256 samples, 1.6 s horizon, ≤ 16 iterations, ≈ 42 min per object; the full DaiD defaults (8.5 min per control step, ≈ 370 GPU-h for all episodes) are removed. Resample DaiD's 50 Hz output back to video frames (FlashCHORD 20.254 Hz).
- **Status 2026-10-23 (ep40 real run): FAIL** — the object never moved (max speed 6e-7 m/s, contacts on 7–10 frames) and the read-back trajectory differed from its own replay by 43 mm (bar 5 mm). **Until both are fixed on ep40, S4c is skipped in all-episode runs (`s4c.enabled: false` in the run config) and S5 uses mode A.** Fixing it is time-boxed to one day (Task Queue #13).
- Out: `s4c_retarget/<candidate>/trajectory_mjwp.npz` (robot joints, realised object pose per frame, contacts), `scene.xml`, `metrics.json` (DaiD's per-step tracking error), and a **realised bundle** `s4c_retarget/<candidate>/bundle/` (object_to_world = realised trajectory).
- Gate (public): `score_local` on the realised bundle ≥ the kinematic one − 0.01 AUC → `use_realised: true`; else the kinematic reference goes to S5 (mode A fallback below). Checks: penetration depth ≤ 5 mm; object displacement before `first_reach` ≤ 1 cm; max object speed ≤ 1.5 m/s; **read-back vs replay ≤ 5 mm; the object must move ≥ 5 cm when the kinematic reference moves it**.
- Day-1 task: diff DaiD's `robots/sharpa` MJCF against FlashCHORD `assets/urdfs/sharpawave/` (joint names, order, limits, 22 per hand); write `pathb/sharpa_map.json`; stop on any unmatched joint.

### S5 — Task file (`stages/s5_task.py`, new)

- **Mode A is the default while S4c is disabled (2026-10-24); Mode B becomes the default again only when S4c passes its gate on ep40.**
- **Mode B:** write FlashCHORD's ManoSharpa parquet from S4c output. Columns (from `flash_chord/data/loader.py`): `object_body_names`, `object_body_position`, `object_body_wxyz`, `robot_{left,right}_wrist_position`, `robot_{left,right}_wrist_wxyz`, `robot_{left,right}_finger_joints`, `{left,right}_robot_finger_joint_names`, `robot_{left,right}_frames`, `{left,right}_robot_frame_names`, `mano_{left,right}_object_contact_positions/_normals/_part_ids`, plus the fps and metadata the loader expects (read the loader's required set and the existing `tools/track3` parquet for the exact dtypes/shapes). `robot_*_frames` by forward kinematics with FlashCHORD's URDF (`pathb/fk.py`, pure numpy from the URDF chain). Contacts from DaiD's contact output; part ids by nearest MANO part of the matched fingertip.
- **Mode A (fallback, exists):** `s23_load_retarget` on the chosen kinematic bundle (`run_load_retarget.py` with `--no-rest-hold` since v3 holds already; `--still-object` for the still candidate).
- Scene files: URDF + support USD built from `s3a_mesh/mesh.glb` through the baseline `mesh_to_usd` (as `run_load_retarget` does).
- Acceptance: FlashCHORD kinematic replay (`scripts/debug/replay.py`, kinematic config) runs to the end; fingertip-to-object distance in the replay matches S4c contacts within 1 cm on 90 % of contact frames; the runner's checks (no link > 2 cm inside the object, seating gap < 1 cm).

### S6 — Policy

- **s6_train** (`stages/s6_train.py`): per candidate present in S5, FlashSAC `sharpa_flash_sac` with Edmond's overrides (`tools/track3/jobs/still_submit.sh` EXTRA: `task.motion_speed=1.0 sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 curriculum.thresholds=[…60006400]`), `sim.njmax=512 sim.nconmax=512` (L40S), 60,006,400 steps, run in `tmux`, checkpoint path recorded. **Measured 2026-10-23: 50,000 env-steps/s on the L40S → ≈ 20 min per candidate.** Two candidates per episode (`keyframe`, `still`; plus `inverted` under §12) — the third was cut with `tracking`. Policy-side changes are in §12 S6.
- **s6_eval**: kit `examples/run_policy_evaluation.py` semantics (`evaluation.world_count=1`, `reset_mode=explicit`, `object_trajectories_output`), one parquet + JSON per candidate.
- **s6_select**: public: `score_local` on each rollout (`tools/track3/score_rollout.py` reads the evaluator parquet) → best AUC, ties → lower MPPE. Evaluation: lowest mean rollout-vs-reference object error over the clip from the evaluator JSON, with termination-equivalent frames counted (error > 10 cm / 0.5 rad) as a penalty; never ground truth. Writes `selection.json` {candidate, reason, scores}.
- Acceptance: on each dev episode the selected rollout must beat that episode's still rollout (first measured in S0; oracle reference: ep40 0.40, ep26 0.50); otherwise S6 reports FAIL and S0.1's decision rule applies.

### S7 — Pack and submit (`run.py pack`, new)

- Preflight (`pathb/preflight.py`): 20 evaluation episodes have `s6_select/selection.json` and the rollout parquet; `s3a_mesh/sha256.txt` equals the hash of the mesh inside S5's URDF and of the file handed to the packer; no `report.json` under `--split evaluation` lists a GT/scan path in `files_read`; `git status` clean and HEAD pushed. Any failure stops.
- Pack: `tools/pack_submission.py --track track_3 --results <dir of episode_XXXXXX.parquet> --meshes <dir of episode_XXXXXX/<object>.glb> --code-commit <url> -o track3_submission.parquet`; review every `MESH WARNING`.
- Upload: `submit/v2d_submit.py track3_submission.parquet --track track_3 -m "<tag>" --wait`; receipt to `path_b/data/submissions/<tag>/`; tag the commit `submit/b-YYYY-MM-DD-N`; checkpoints, task parquets, meshes to HF via `path_b/vm/hf_outputs.py upload`.

## 5. Task list (in order; each task ends with its test)

CPU tests run on the Mac with `uv run --project path_b pytest path_b/tests`; GPU smoke tests on the VM on ep40 and ep26 (per object) only.

1. Repo move to `cache/v2`, new `STAGES`, `run.py oracle|pack` stubs. Test: existing 13 unit tests pass; `run.py run --episodes 41 --stages p0_ingest p1_calibrate p2_depth` reproduces run `4c2eac31` reports.
2. `pathb/phases.py` + `s4a_phases`. Test: synthetic wrist/object arrays → expected windows; ep41 bundle → first_reach 19 (fixture from P3's report); ep40 and ep26 values recorded in BUILD_LOG.
3. `pathb/symmetry.py` (roster → group) and `pathb/fuse.py` (ESKF/RTS on world poses with anchors). Tests: a flipped synthetic track is recovered; anchors pin a noisy static segment to < 1 mm.
4. `s4b_reference` (three candidates) + waterfall columns. Test: GT trajectory in → playback AUC 1.0 for `tracking`; `keyframe` from GT anchors + interpolation ≥ `still`.
5. S0.1: `run.py oracle --episodes 40 26` (GT-reference task + train + eval) — **first VM job**; record numbers in `path_b/oracles/` next to `oracle_2026-10-17.json` (S0.2, done; `path_b/data/` is git-ignored, so oracles live in `path_b/oracles/`).
6. `p1b_masks` (SAM 3.1 or SAM2 fallback). VM smoke: overlays for ep40 and ep26.
7. `p1c_camera` (masked depth prior into DROID). VM smoke: plane-height spread bar.
8. `p1d_hands` (stereo-aligned, late hands). VM smoke: coverage bars.
9. P3/P4 v3 changes. Test: factor within 0.97–1.03 on ep40 and ep26.
10. ~~`s3a_mesh` (SAM3D + MV-SAM3D)~~ **superseded 2026-10-24 by §11 A2.1–A2.3 (canonical mesh)**; the SAM 3D stage FAILed on ep40 and is removed.
11. `s3b_track` driver with the recovery rule. VM: no > 60° camera-frame jumps on ep40/ep26; ep26's pot (oracle: rotation error 75° at rest, 106° moving) is the flip test. (ep40 real run: WARN.)
12. `s3c_fuse`. VM: waterfall `fused` ≥ the episode's `stage2_stereo` column (from HF run `0fa525bc`) on ep40 and ep26.
13. `pathb/daid.py`, `sharpa_map.json`, `s4c_retarget`. VM: ep40 and ep26 (per object) **keyframe only** realised; gate result. (ep40 real run: FAIL — object never moves, 43 mm read-back gap; disabled in all-episode runs until fixed.)
14. `s5_task` mode A default, mode B + `pathb/fk.py` behind the S4c gate. VM: kinematic replay passes.
15. `s6_*` on ep40 and ep26 (mode A merge for ep26), **two candidates**; selection. VM: best rollout vs that episode's still.
16. `pack` preflight + dry pack on the public split (no upload). (Coded.)
17. Evaluation split: P0–P2 + the full chain on the 20 episodes; submission.
18. **§12 v4** (Mac, while 17 runs): tasks 18.1–18.6 in §12; dev-episode go/no-go before any v4 GPU hour.

## 6. Run plan (one L40S; rebuilt 2026-10-24 for the Nov 1 stop)

Phase-A build and the ep40 smoke test are done (BUILD_LOG 2026-10-23). The first VM was lost on 2026-10-23 with run `phaseb` (≈ 2.5 GPU-h; code intact). Budget below uses the measured timings: S6 20 min per candidate, S4c 42 min per object (disabled), ≈ 1.5 GPU-h per single-object episode without S4c.

| When | Job | GPU-h |
|---|---|---|
| Oct 24 | James: new L40S, rotate HF write token, delete old deploy key. Session: push the Mac-only commits (§9), `phase_b.sh` per-episode HF upload, then `new_instance.sh` + dev-cache restore | 2 |
| Oct 24–25 | ep40 real run (S4c off, two candidates, A2 mesh) + front-end comparison (team bundle vs v3 S1, same downstream) → choose S1 for phase B; S0.1 on ep40 and ep26 | 2 |
| Oct 25–27 | **Phase B**: v3 on 20 public then 20 evaluation episodes (≈ 60 h with the cuts: one S4c candidate → S4c off, two policies); waterfall; pack + preflight; **submission #3 needs James's go** | 60 |
| Oct 24–28 (Mac) | §12 v4 S3a pooling, S3b, S3c, S3d, tests; v4 on ep40/ep26 from the dev cache; go/no-go | 0 |
| Oct 28–30 | if v4 passes on both dev episodes: v4 references for 40 episodes (CPU), S6 two candidates × 60M on all 40, then 250M on the chosen reference for the 20 evaluation episodes | ≈ 55 |
| Oct 31 | pack + preflight → **submission #4 (James's go)**; review page; BUILD_LOG summary. If phase B overran, v4's S6 runs on the 20 evaluation episodes only (≈ 27 h) | |
| Nov 1 | stop | |

Removed: the "quick win" still submission #2 (16 GPU-h; optional, James decides), phase 2 run D (55 GPU-h; §8), S4c on three candidates. After each finished episode: `hf_outputs.py upload --run <run> --episode <n>`; delete nothing on the VM before the upload is listed.

## 7. Compliance

- GT firewall: `pathb/paths.py` gains `gt_allowed(ctx)`; every reader of `public/data/*.parquet` or `public/mesh/` goes through it; with `split == "evaluation"` it raises; each stage appends the files it read to `report.json: files_read`.
- Mesh hash S3a → S5 → S7 (`sha256.txt` compared in preflight).
- `manual_inputs.json` per episode lists any hand-chosen parameter (none expected; prompts come from the roster).
- `THIRD_PARTY.md`: Do as I Do (MIT), SAM 3.1 (SAM license), Open3D (MIT), FoundationStereo/FoundationPose/WiLoR/HaMeR/MANO (via the baseline, as the organiser ships them), MuJoCo (Apache-2.0). MV-SAM3D and Point2Pose removed (not used).
- **Hands (decided 2026-10-24): HaMeR via the baseline; EgoFound3R is not used in anything built, submitted or released.** Its README (github.com/BAp3r/EgoFound3R, read 2026-10-24) releases code and weights under CC BY-NC-SA 4.0, "academic research and educational use only, commercial use is not permitted". The challenge pays prizes and requires winners to release code (FAQ: Apache 2.0); the ShareAlike clause conflicts with that release and the NC clause is a risk for a prize-winning entry. The baseline's own WiLoR (CC BY-NC-ND) and MANO (non-commercial) are the organiser's choice and stay.

## 8. Phase 2: two-object scope — reduced 2026-10-24

Built and tested on ep26 with stand-in inputs (2026-10-23): support relation, per-object keyframes anchored on the roster-first object, Do as I Do two-body scene. Result 0.500 either way (RPE 28.3 vs 28.1 cm), because the stand-in called the moving pot "at rest". **The two-object scene code and run D (≈ 55 GPU-h) are dropped from the plan**: §12 v4 handles two objects per segment (a static segment "on another object" is registered in that object's frame), and the Do as I Do two-body scene only matters if S4c works. What stays: S4a phases per object, S5 parquet with two `object_body_*` entries, S6 selection with RPE, the mode-A merge. Rerun phase 2's existing code on ep26 only once v3's own perception has run on it; keep it only if it beats the merge.

## 9. Workflow (James, 2026-10-22; supersedes the 2026-10-21 version)

**Where things run.** The Mac has no CUDA GPU. CPU stages are built, debugged and run locally on the real ep40 and ep26 data: unit tests, P0, P1, P3/P4, S3c, S4a, S4b, S5 parquet writing, `score_local`, S6 selection, S7 pack/preflight, all adapters. GPU stages (P2, P1b, P1c, P1d, S3a, S3b, S4c, S6 train/eval) are smoke-tested on the L40S on ep40 and ep26 **only**, each as soon as it is written. The cloud is used for exactly two things: those dev-episode smoke tests and the all-episode runs (phases B and D).

**Dev cache.** First cloud job: P0–P2, P1b, P1c, P1d, S3a, S3b outputs for ep40 and ep26 (~15 GPU-min), uploaded with `hf_outputs.py` and restored into `path_b/data/` on the Mac, so every CPU stage develops against real intermediate data. S0.1 runs in the same job.

**Per task.** (1) Local: code, `uv run --project path_b pytest path_b/tests`, run the stage on ep40/ep26 if CPU, commit locally (message starts with the task number); **when the tests pass, `git push origin zjifan121_pathb_v3`**. (2) Cloud, GPU stages only: push to GitHub (the VM pulls `zjifan121_pathb_v3` from GitHub; the bare-repo route is the fallback when GitHub is unreachable), run on ep40 and ep26, record numbers and report statuses in `BUILD_LOG.md`, `hf_outputs.py upload`, restore to the Mac if a later CPU stage needs it. **When the smoke test passes: commit the BUILD_LOG line and push to `zjifan121_pathb_v3` immediately — before any single-episode or all-episode run starts.** Never edit code on the VM. A failing smoke test returns to (1). Task N+1 may be coded while task N's cloud job runs.

**Phases.** A: phase-1 build (tasks 1–15). **A2: S3 observation variant (§11), built and compared on the dev episodes before any all-episode run.** B: phase-1 run on all 20 public then 20 evaluation episodes (tasks 16–17 to the packed parquet). **C (2026-10-24): v4 build (§12) on the Mac during B; D: v4 run (references on CPU, S6 on the L40S) if the dev go/no-go passes.** The former phase-2 scene build and run are dropped (§8). E: preflight, final summary, **HOLD before `v2d_submit.py`**. Phases B and D start automatically once their inputs exist; they need a VM (below). During B and D the next phase is coded locally while the run proceeds; the run is checked between tasks and fixed-and-resumed on failure (a stage with a non-ERROR `report.json` is kept).

**VM lifecycle.** Verda bills a stopped instance at full price, so the VM is deleted after each cloud use. After the dev cache and a phase's last smoke test are on HF: tell James the VM can be deleted. Before B and D: write the run's needs and GPU-hour estimate to `BUILD_LOG.md`, tell James "need an L40S", wait for the IP, rebuild with `setup.sh` (`BRANCH=zjifan121_pathb_v3` once on GitHub, else push through the bare repo), restore from HF, start the run in tmux.

**GitHub (James, 2026-10-24; replaces the phase-boundary rule).** All future work is pushed to GitHub **immediately after its smoke test passes** (GPU stages) or its tests pass (CPU tasks), and always before a single-episode or all-episode run uses it. Push to the personal branch **`zjifan121_pathb_v3` only**. **No PR and no merge into `main_buffer`** until James says so in chat; do not rebase onto `main_buffer` on your own (sync is Task Queue #14, with his yes). Never force-push. First action under this rule: push the 29 commits that exist only on the Mac (`ba9b051f` and ancestors; GitHub is at `89f44903`).

**Episode runs.** `phase_b.sh` uploads each finished episode to HF before starting the next, and the full run is started in the same turn as the passing smoke test (a VM idled for a day once).

**Waits.** Exactly two: VM provisioning, Kaggle upload (HOLD). The GitHub push no longer waits. Everything else is decided by the rules in the session prompt and recorded under "Decisions" in `BUILD_LOG.md`.

## 10. Session facts (2026-10-21; updated 2026-10-24)

- Branch: `zjifan121_pathb_v3` (Mac at `ba9b051f`, GitHub at `89f44903` as of 2026-10-24 — push first). PRs to `main_buffer` only on James's word (§9).
- VM: the 2026-10-21 instance (`31.56.109.90`, later `86.38.238.153`) is gone; its SSH host key changed on 2026-10-24 — **do not accept the new key**. James provides the next L40S IP; update `~/.ssh/config` `Host l40s` with it. First VM step: `path_b/vm/new_instance.sh` (`BRANCH=zjifan121_pathb_v3`, now on GitHub; `RESTORE_RUN=4c2eac31 RESTORE_FRAMES=1`), ≈ 2 h. New venvs on the VM: `sam3` (SAM 3.1) and `daid` (Do as I Do, cloned under `/root/v2d/third_party/`).
- Secrets: `path_b/.env` on the Mac, written by James (git-ignored; copy to the VM with `scp`; never commit, never print): `HF_TOKEN=<read-only token of HF user JzJz7779>`, `KAGGLE_API_TOKEN=<James's Kaggle access token>`; HF uploads need James's write token via `hf auth login` on the VM. **The old VM's disk held `path_b/.env` and a read-only deploy key: James rotates the HF write token and deletes the deploy key (repo Settings → Deploy keys) before the next instance.** Source the file (`set -a; . path_b/.env; set +a`) in every shell that runs P1b, `hf_outputs.py` or `v2d_submit.py`.
- Kaggle uploads: none without James's explicit go-ahead in chat.

## 11. Phase A2 — S3 observation variant (after phase A, before phase B; local)

**Why.** Teammates ran SAM 3D Objects and MV-SAM3D on our objects (2026-10-22): run-to-run randomness and the choice of the input view dominate the result, and view ("keyframe") selection has no reliable rule. Both are conditional generative models (SAM 3D: two-stage latent flow matching, ~25 sampling steps; seed fixes the noise sample, not the view dependence), so the problem is structural. The mesh today also does three jobs at once — CD-O shape, FoundationPose template, collision body — so a random mesh randomises tracking and physics as well. A2 replaces generation with **observation** and **decouples pose from shape**. Facts it rests on: stereo depth is metric; the same 10 physical objects recur across the 40 clips (`episodes.json`: white_pot 15, blue_cup 10, plastic_dish_rack 10, wooden_spoon 10 evaluation-only, others 3–5); 7 of 10 objects are symmetric (revolution: white_pot body, white_pot_lid, blue_cup, beige_cup, water_pitcher body; mirror: mini_dust_pan, wooden pieces).

**Where it runs.** Everything in A2 except the two GPU items below runs on the Mac from the dev cache (P2 depth, P1b masks, P1c camera for ep40 and ep26). GPU items follow the §9 smoke-test rule (L40S, dev episodes only): (i) one FoundationPose run per dev object with the new mesh; (ii) the mesh-free tracker. If no VM is up when A2 reaches them, A2 is evaluated with the FoundationPose track from phase A re-registered to the new mesh, and (ii) is deferred to the next VM window.

### A2.1 Per-clip partial scan (`stages/s3a2_fuse_scan.py`, new; CPU)
- In: `p2_depth/depth_a_half/` (stereo depth in cam_a), `p1b_masks/obj_<k>/`, `p1c_camera/camera_to_world/`, `s4a_phases/phases.json` (rest windows).
- Do: Open3D `ScalableTSDFVolume` (voxel 4 mm, sdf_trunc 16 mm; defaults) over all rest-before and rest-after frames where the object mask area ≥ 1 % of the image and the hand masks do not overlap it; depth outside the object mask set to 0; integrate in the world frame; marching cubes → `partial_scan.ply` (metres, world frame, visible surfaces only). Write per-frame weights (mask area) to `fuse_report.json`.
- Acceptance: ≥ 30 frames integrated (PASS) / 10 (WARN); surface RMS vs stereo depth on 10 held-out rest frames ≤ 5 mm (PASS) / 10 mm (WARN). Public only: one-sided Chamfer partial → scan ≤ 0.3 cm (the partial scan should lie *on* the true surface).

### A2.2 Canonical mesh per object (`run.py canonical --object <name>`, new; CPU)
- In: every available `partial_scan.ply` of that object (dev episodes first; later all clips of both splits — videos only, never GT or scans).
- Do: pairwise ICP with scale fixed at 1 (all scans are metric) onto the clip with the largest surface; re-fuse into one TSDF; fit the symmetry (revolution axis for the five revolution objects: PCA axis + circle fits per height slice; mirror plane for dust pan and wooden pieces: best plane maximising overlap); complete by symmetry (revolve the median radial profile / mirror the cloud); screened Poisson → watertight; decimate to ≤ 20k faces → `canonical/<object>/mesh.glb` + `canonical.json` (axis, plane, source clips, coverage fraction before completion).
- Objects without a usable symmetry (plastic_dish_rack, mini_sweeper, wooden_spoon): fused TSDF + mirror if the fit is confident, else fused TSDF alone; flag `coverage < 0.6`.
- Fallbacks (deterministic, in this order): SymmCompletion (MIT, pretrained PCN checkpoint, deterministic flag) → AdaPoinTr (MIT) → ForeHOI (MIT, learned regressor, RGB in). Retrieval (Uni3D over Objaverse-XL) only for dish rack / sweeper if they fuse into a blob; record which fallback fired.
- Acceptance: watertight; extents within 0.8–1.25× of the fused partial; public only: CD-O vs scan via `tools/track3/eval_mesh.py` **< the phase-A mesh for the same object** (today's bundle meshes: 0.47–0.74 cm; fitted 0.48 cm) — this is the comparison A2 exists to make.

### A2.3 Object frame and registration (`stages/s3a2_register.py`, new; CPU)
- Per episode and object: ICP of the rest-before partial scan onto the canonical mesh (scale fixed) → `object_frame_T_world0` (frame-0 pose in the canonical frame). All poses the episode submits are expressed in the canonical mesh's frame, so mesh and poses stay consistent as the kit requires (`tracks/track_3/README.md`, "meshes in the same body coordinates as their rollout poses").

### A2.4 Mesh-free tracking — removed 2026-10-24
Coded as `stages/s3b2_track_points.py` (TAPIP3D backup); the tracker's CUDA extension did not build on the VM and it never ran. Point2Pose was not attempted. Per-frame tracking is what §12 replaces, so A2.4 is not pursued; the file moves to `path_b/cache/`.

### A2.5 Comparison and decision — done 2026-10-23
- Measured on the stand-in inputs: ep40 `ref_keyframe` 0.540 → `a2_mesh_fp` **0.726** (MPPE 3.3 cm); CD-O 0.24 (team mesh) vs 0.53 cm (canonical from one clip, coverage 0.58). ep26 0.349 → 0.500 (= still). Ranks: AUC 1 vs 2, CD-O 2 vs 1 → tie → the deterministic variant. **Decision: `a2_mesh_fp` is S3 for phase B** (recorded under "Decisions"). The gain is from mesh placement (the mesh sits on the tracked object), not shape; the shape improves with cross-clip pooling (§4 S3a).
- Owed: the same comparison on v3's own perception for ep40 (Task Queue #3).
- Collision body for S4/S6 comes from the canonical mesh (CoACD as before).

**Spec consequences applied 2026-10-24:** §2 `mvsam3d` dropped; §4 S3a = A2.1–A2.3; §7 `THIRD_PARTY.md` adds Open3D; §5 task 10 superseded.

## 12. Path B v4 — contact-segmented inversion + policy-side changes (design 2026-10-24; build on the Mac while phase B runs)

**Why.** The oracle (§S0.2) puts the loss in object *position* (true positions alone → 0.741; true positions on moving frames alone → 0.762), and v3's per-frame tracker has not shown centimetre accuracy on any real episode (ep40 real run: best reference 0.381–0.400 = still). v4 keeps S1–S2 and the canonical mesh and replaces per-frame tracking with per-segment solving: the clip is split into **static**, **stable-grasp** and **unstable** segments; a static segment has one pose, a grasp segment has one fixed hand→object offset (the stereo-aligned hand track supplies the motion), and unstable segments are short bridges. Unknowns fall from 6 per frame to ≈ 30 per clip, each constrained by hundreds of depth, mask and fingertip observations. Rest poses are settled in MuJoCo so they lie on the table. Evidence from the 2026-10-23 research round: rigid-in-hand inversion is the principle behind the egocentric hand-object methods that report centimetre-level object error (ROHIT arXiv 2512.07394, VICON arXiv 2610.05180, ComPose; mocap-level results in DexMimic / OATH). Companion artifact: "Path B v4 Design".

**Scope.** New code is CPU-only (`pathb/segments.py`, `pathb/invert.py`, `stages/s3b_segments.py`, `stages/s3c_invert.py`, `stages/s3d_settle.py`) plus a P1d export and S6 config changes. No new environment: Open3D (already in A2), scipy, MuJoCo from the `daid` venv (CPU). Two objects are native: segments are per object; a static segment resting on another object is registered in that object's frame.

### 12.1 P1d export (change)
`p1d_hands` additionally writes `fingertips_world.npz` (N, 2, 5, 3) and `hand_verts_world.npz` (N, 2, 778, 3) from the aligned MANO (through `tools/track3/read_mano.py`). Hands remain HaMeR (§7).

### 12.2 S3b segments (`stages/s3b_segments.py`, new; `pathb/segments.py` pure function)
- In: `p2_depth/depth_a_half`, `p1b_masks/obj_<k>`, `p1c_camera`, `fingertips_world.npz`, `s3a2_register` frame-0 pose, canonical mesh.
- Per frame, masked object points in world. **static**: median displacement of the object points over a 10-frame window < 5 mm and no fingertip within 2 cm of them. **grasp**: ≥ 2 fingertips within 1 cm of the object points and the same fingertip set (by hand × finger) for ≥ 5 frames. Else **unstable**. Dilate grasp by 2 frames; merge static runs shorter than 5 frames into their neighbours.
- Out: `segments.json` [{kind, start, end, fingertips}], `phases.json` as S4a writes it (S4a becomes a view of this; `pathb/phases.py` stays for the fallback).
- Acceptance: on ep40 the first grasp starts within ±5 frames of S4a's `first_reach` (19) and the last ends within ±5 of `last_release` (230); ep26 dish rack is one static segment; ep26 pot has ≥ 1 grasp segment (the stand-in never found one — that is the test).

### 12.3 S3c invert (`stages/s3c_invert.py`, new; `pathb/invert.py`)
- **Static segment**: TSDF of the segment's masked depth (A2.1 code), ICP of the canonical mesh onto it with scale fixed, initialised from `s3a2_register` (first segment) or the previous segment's end pose; symmetric objects: candidates = pose × symmetry group (`pathb/symmetry.py`), keep the lowest point-to-mesh residual → one pose for the segment. Bar: residual median ≤ 5 mm.
- **Grasp segment**: unknown `T_hand→object` (6 DoF, `wrist` of the grasping hand; two-handed grasps use the hand with more fingertips in contact). `object(t) = wrist(t) · T`. Residual over all frames of the segment, scipy `least_squares` with a soft-L1 loss: (a) rendered canonical-mesh depth vs stereo depth inside the object mask (trimesh ray cast or Open3D raycasting scene, every 2nd frame if slow); (b) contact fingertips to the mesh surface (signed distance → 0); (c) end poses equal the neighbouring static poses (weight 10). Initialise `T` from the preceding static pose and the wrist at the segment's first frame. **Slip guard**: if the per-frame residual (a) exceeds 1.5 cm for 10 consecutive frames, split the segment there and re-solve the parts (the split becomes an unstable bridge). Bar: residual (a) median ≤ 1 cm on ≥ 50 % of frames, else the segment is marked `failed` and S4b uses `keyframe` interpolation for it.
- **Unstable segment**: cubic Hermite (position, zero end velocity) + slerp between the neighbouring segment end poses, then ≤ 5 Gauss-Newton steps of per-frame refinement on residual (a) where the mask covers ≥ 1 % of the image; motion limited to 1.5 m/s.
- Out: `s3c_invert/object_to_world.npz` (N,4,4), `invert_report.json` (per segment: kind, residuals, iterations, failed), and the bundle `result.npz` completed as S3c did. S3b FoundationPose (§4) is only used to seed the first static ICP's orientation when `s3a2_register` fails.
- Acceptance (public): waterfall column `inverted` ≥ `ref_keyframe` on ep40 and ep26 — the v4 go/no-go (Task Queue #9).

### 12.4 S3d settle (`stages/s3d_settle.py`, new; `daid` venv, CPU MuJoCo)
For each static segment: load the CoACD pieces, place the object at the ICP pose on the table plane (z = 0) or on the supporting object (from the segment's registration frame), simulate 0.5 s with gravity, read the resting pose. Accept if it moved < 1.5 cm and rotated < 10°, else keep the ICP pose and flag `unsettled`. Replace the static pose; shift the adjoining grasp/unstable end constraints accordingly and re-run 12.3's end-pose terms (one pass). Bar: penetration of mesh into table ≤ 2 mm after settle.

### 12.5 S4–S5 under v4
S4b gains candidate **`inverted`** (12.3 + 12.4 poses, rests held exactly); `keyframe` becomes the fallback for episodes where any grasp segment `failed`; `still` stays. S4c (if enabled by its gate) runs on `inverted` only. S5 mode A unless S4c passes. Waterfall adds `ref_inverted`, `rollout_inverted`.

### 12.6 S6 policy-side changes
1. **Residual on prior**: the actor's output is a correction added to the reference joint targets (the kinematic replay) rather than an absolute target, so step 0 replays the reference and learning repairs contacts. Check FlashCHORD's config for an existing residual/`action_mode` option first; if a code change is needed it goes into our fork under `path_b/third_party/flash_chord_patch/` (never under `baseline/`), as a patch applied in the `flash_chord` venv.
2. **Object-error termination during training**: terminate when the object is > 5 cm from the reference (in addition to the wrist bar), so position tracking is what is rewarded; evaluation runs with the kit's own settings as before.
3. **Steps**: two candidates × 60M for selection; then **250M on the selected reference, 2 seeds when GPU time allows (1 otherwise)**; the oracle's 500M-on-a-bad-reference result (0.142) is why steps go only to the chosen one.
4. **Selection**: checkpoint and seed by **MPPE of the rollout against our own reference** (lower wins; ties → AUC against the reference), never GT; public episodes additionally report `score_local` for the waterfall.
Acceptance: on ep40 and ep26 the selected rollout beats the still rollout and loses ≤ 0.1 AUC to its own reference's playback; S0.1 (GT reference) bounds the policy tax first.

### 12.7 Tasks (§5 numbering continues)
18.1 P1d export + unit test (fixture hands → fingertips). 18.2 `segments.py` + `s3b_segments` (synthetic + ep40/ep26 dev cache). 18.3 `invert.py` static + grasp + unstable, `s3c_invert`; test: GT trajectory with synthetic depth → ≤ 3 mm; ep40 dev cache → `inverted` column. 18.4 `s3d_settle`. 18.5 S4b/S5 wiring, waterfall columns, go/no-go on both dev episodes. 18.6 S6 changes (config first, patch if needed), smoke on one GPU hour. Then the §6 Oct 28–31 run. Every task follows §9: tests pass → push `zjifan121_pathb_v3`.

### 12.8 Risks and guards
Slip inside a grasp (split rule); heavy hand occlusion (fingertip + end-pose terms carry the fit; 50 % bar; fallback keyframe per segment); symmetric objects (candidate rule; rotation is secondary in the oracle); unknown policy tax (S0.1 first); schedule (8 days; v4 code is CPU and ≈ 5 files).

Sources (2026-10-24): `path_b/oracles/oracle_2026-10-17.json`; `path_b/BUILD_LOG.md`; EgoFound3R README (CC BY-NC-SA 4.0); the 2026-10-23 research round (artifact "Path B v4 Design" lists the papers); Kaggle Track 3 boards 2026-10-23 (leader 0.922 AUC / 1.30 cm MPPE; team 0.425, rank 13/27).

Sources (checked 2026-10-22): SAM 3D Objects repo and paper (arXiv 2511.16624); MV-SAM3D (arXiv 2603.11633); Point2Pose (arXiv 2604.10415, github.com/TheWisp/point-to-pose); TAPIP3D (github.com/zbw001/TAPIP3D); Open3D RGBD integration docs; SymmCompletion (github.com/hongyuyann/symmcompletion); AdaPoinTr (github.com/yuxumin/PoinTr); ForeHOI (arXiv 2602.06226); Uni3D (github.com/baaivision/Uni3D); `tools/track3/episodes.json`.
