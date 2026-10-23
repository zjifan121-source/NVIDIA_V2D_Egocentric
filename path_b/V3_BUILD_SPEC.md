# Path B v3 build spec (single-object chain; dev episodes ep40 + ep26)

Written 2026-10-21 for a coding session. Companion overview: artifact "V2D ep41 Pipeline"; background: Project doc `claude/v2d-e2e-research-2026-10-21.md`. Decisions from James (2026-10-21): rerun everything from the video (no bundle reuse); keep the Path B layout (`path_b/run.py`, `pathb/stages/*.py`, one `report.json` per stage, `loss_report.md` per episode); keep only the v2 code v3 uses, park the rest in `path_b/cache/`; one Verda L40S (48 GB). **Dev episodes: ep40** (public, `mini_dust_pan`, the median single-object episode) **and ep26** (public, `plastic_dish_rack` + `white_pot`, the median two-object episode); ep41 is not rerun. **Build phase 1 = the single-object chain only**, run on ep40 and, per object, on ep26 (each object of ep26 goes through S1–S6 independently with the shared camera; the scene is assembled for scoring with the existing merge in `run_load_retarget.py --merge`, mode A). Then all episodes. **Phase 2** (after all episodes have run): the two-object scope (§8) is built and the two-object episodes are rerun and compared. SAM 3.1 access is granted (`facebook/sam3.1`, user `JzJz7779`). Workflow (§9, 2026-10-22): local-first; cloud only for dev-episode GPU smoke tests and all-episode runs; GitHub pushes batched per phase with James's confirmation.

Every interface below was read from the code on 2026-10-21 (`main_buffer` @ `e61fc18`; local `zjifan121`; baseline `8c778f9`; kit in `assets/data/v2d_submission_kit`). Where a parameter is a proposal, it says "default:" and the acceptance test decides.

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
s3a_mesh  s3b_track  s3c_fuse                           (S3)
s4a_phases  s4b_reference  s4c_retarget                 (S4)
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
| `sam3` venv (new) | 3.12 | S1b SAM 3.1 (`facebook/sam3.1`, gated) | fallback: SAM2 container with Grounding DINO box, as the baseline |
| `mvsam3d` (inside the SAM3D container or its own venv) | per repo | S3a MV-SAM3D | github.com/devinli123/MV-SAM3D; needs SAM3D weights + Depth Anything 3 (we pass our own stereo pointmaps instead) |
| `daid` venv (new) | 3.12 | S4c Do as I Do retargeting | github.com/malik-group/do-as-i-do `retargeting/` (`pip install -e .`; MuJoCo Warp, mink, viser, coacd); ≥32 GB GPU |
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

**S3a mesh** (`stages/s3a_mesh.py`, new)
- View selection: among frames with object mask area ≥ 2 % of the image and hand-mask overlap with the object mask ≤ 5 %, pick up to 5 frames maximising the angular spread of the camera-to-object direction (from P1c + mask centroid depth); always include one rest-before and, if present, one rest-after frame (phases from S4a run early as a helper: `pathb/phases.py` is a pure function used by both).
- SAM 3D Objects per view: call `v2d_sam3d/lib/image_to_mesh.py` with `pointmap_path` = stereo pointmap of that frame (unproject `depth_a_half` with its K, PyTorch3D axes as `_depth_to_pointmap` does), `seed=0`, `depth_mask_path` = object mask. Keep `mesh_transform.json` (rotation, translation, **scale**).
- MV-SAM3D: run its fusion on the per-view SAM3D outputs with our masks and poses (P1c camera, object pose from the view's `mesh_transform`); if the fusion fails or the repo cannot be installed in time, fall back to "best single view by stereo-depth fit" (lowest median |rendered depth − stereo depth| inside the mask, as `fit_shape.py` measures).
- Metric size: vertices × per-axis `scale` from SAM3D, then one uniform correction from the stereo-depth fit over the 5 views (`fit_shape.py` logic, 3 axes allowed only within 0.8–1.25).
- Out: `s3a_mesh/mesh.glb` (visual, metres, object frame = SAM3D canonical frame), `mesh_views.json`, `coacd/` (convex pieces via `coacd`, threshold 0.05), `mass.json` (volume × density table in `pathb/density.py`, fixed before any evaluation run), `sha256.txt`.
- Acceptance: watertight (trimesh) PASS; extents 0.05–0.6 m; stereo-fit residual < 1 cm median (PASS) / 2 cm (WARN); public only: CD-O vs scan via `tools/track3/eval_mesh.py` < 0.3 cm (PASS) / 0.5 (WARN), size ratio 0.8–1.25.

**S3b track** (`stages/s3b_track.py`, changed)
- Wraps: `v2d_foundation_pose/lib/foundation_pose_tracker.py` (`register`, `track_one`, `track_one_with_recovery`, `track_one_particles`, `reset_to_pose`) run inside the FoundationPose container through a small driver script `path_b/pathb/drivers/fp_track.py` mounted with `--dev`. Inputs: `s3a_mesh/mesh.glb` (no scale search: `--skip_object_scale_estimation` equivalent), `cam_a/`, stereo depth upsampled, `obj_0` masks, cam_a K.
- Registration frame: the S3a view with the largest unoccluded mask. Track forward then backward as the baseline.
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
- Order of work: `keyframe` first (S0.2), then `tracking`; `still` is free. Writes the kinematic references as v3 bundles (`result.npz` + mesh), each a complete input for S4c/S5:
  - `tracking`: S3c poses; rest windows replaced by the anchors with a 5-frame ease-in/out (`run_load_retarget.hold_until` logic, both ends).
  - `keyframe`: rest-before anchor held until `first_reach`; carry = S3c poses on frames whose `track_meta` shows IoU ≥ 0.5 and not recovered, else interpolation between the nearest trusted frames (position: cubic Hermite with zero end velocity; rotation: slerp); rest-after anchor held from `last_release`. Minimum: the whole carry may be one interpolation from anchor to anchor over the carry window.
  - `still`: rest-before anchor for all frames (the fallback; equals `--still-object`).
- Hands: P1d MANO in the world, cleaned with `run_load_retarget` hand cleaning (import its functions; do not copy).
- Out: `s4b_reference/<candidate>/` bundles; `reference_report.json` (per candidate: frames trusted, interpolated, held).
- Acceptance (public): `score_local` playback per candidate → waterfall columns `ref_tracking`, `ref_keyframe`, `ref_still`.

**S4c retarget** (`stages/s4c_retarget.py`, new; Do as I Do)
- Adapter `pathb/daid.py`: writes the `trajectory_keypoints.npz` layout that `retargeting/` stage 1 produces (read its writer once in-session: wrist pose convention, fingertip keypoints, object pose per frame, mesh path) from a v3 bundle; copies `coacd/` pieces; the table is the MuJoCo floor at z = 0.
- Run: DaiD stages 2–5 (`decompose` → scene MJCF with `--robot-type sharpa` → IK (mink) → sampling MPC in MuJoCo Warp) in the `daid` venv, per candidate (`tracking`, `keyframe`); `--max-sim-steps 0` (full trajectory). Config knobs to expose in `s4c.yaml` (read DaiD `config/default.yaml` for names): horizon, samples, warm-up weld frames (default = rest-before length), failure penalty weight, control rate (resample reference to 50 Hz? DaiD commands at 50 Hz; FlashCHORD runs at 20.254 Hz — resample DaiD output back to video frames).
- Out: `s4c_retarget/<candidate>/trajectory_mjwp.npz` (robot joints, realised object pose per frame, contacts), `scene.xml`, `metrics.json` (DaiD's per-step tracking error), and a **realised bundle** `s4c_retarget/<candidate>/bundle/` (object_to_world = realised trajectory).
- Gate (public): `score_local` on the realised bundle ≥ the kinematic one − 0.01 AUC → `use_realised: true`; else the kinematic reference goes to S5 (mode A fallback below). Checks: penetration depth ≤ 5 mm; object displacement before `first_reach` ≤ 1 cm; max object speed ≤ 1.5 m/s.
- Day-1 task: diff DaiD's `robots/sharpa` MJCF against FlashCHORD `assets/urdfs/sharpawave/` (joint names, order, limits, 22 per hand); write `pathb/sharpa_map.json`; stop on any unmatched joint.

### S5 — Task file (`stages/s5_task.py`, new)

- **Mode B (default):** write FlashCHORD's ManoSharpa parquet from S4c output. Columns (from `flash_chord/data/loader.py`): `object_body_names`, `object_body_position`, `object_body_wxyz`, `robot_{left,right}_wrist_position`, `robot_{left,right}_wrist_wxyz`, `robot_{left,right}_finger_joints`, `{left,right}_robot_finger_joint_names`, `robot_{left,right}_frames`, `{left,right}_robot_frame_names`, `mano_{left,right}_object_contact_positions/_normals/_part_ids`, plus the fps and metadata the loader expects (read the loader's required set and the existing `tools/track3` parquet for the exact dtypes/shapes). `robot_*_frames` by forward kinematics with FlashCHORD's URDF (`pathb/fk.py`, pure numpy from the URDF chain). Contacts from DaiD's contact output; part ids by nearest MANO part of the matched fingertip.
- **Mode A (fallback, exists):** `s23_load_retarget` on the chosen kinematic bundle (`run_load_retarget.py` with `--no-rest-hold` since v3 holds already; `--still-object` for the still candidate).
- Scene files: URDF + support USD built from `s3a_mesh/mesh.glb` through the baseline `mesh_to_usd` (as `run_load_retarget` does).
- Acceptance: FlashCHORD kinematic replay (`scripts/debug/replay.py`, kinematic config) runs to the end; fingertip-to-object distance in the replay matches S4c contacts within 1 cm on 90 % of contact frames; the runner's checks (no link > 2 cm inside the object, seating gap < 1 cm).

### S6 — Policy

- **s6_train** (`stages/s6_train.py`): per candidate present in S5, FlashSAC `sharpa_flash_sac` with Edmond's overrides (`tools/track3/jobs/still_submit.sh` EXTRA: `task.motion_speed=1.0 sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 curriculum.thresholds=[…60006400]`), `sim.njmax=512 sim.nconmax=512` (L40S), 60,006,400 steps, run in `tmux`, checkpoint path recorded. ~0.8 h each (A6000 figure; measure on the first run).
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
10. `s3a_mesh` (SAM3D + pointmap; MV-SAM3D if installed; fit). VM: CD-O vs scan on ep40's dust pan and ep26's pot and dish rack.
11. `s3b_track` driver with the recovery rule. VM: no > 60° camera-frame jumps on ep40/ep26; ep26's pot (oracle: rotation error 75° at rest, 106° moving) is the flip test.
12. `s3c_fuse`. VM: waterfall `fused` ≥ the episode's `stage2_stereo` column (from HF run `0fa525bc`) on ep40 and ep26.
13. `pathb/daid.py`, `sharpa_map.json`, `s4c_retarget`. VM: ep40 and ep26 (per object) keyframe and tracking realised; gate result.
14. `s5_task` mode B + `pathb/fk.py`. VM: kinematic replay passes.
15. `s6_*` on ep40 and ep26 (mode A merge for ep26), three candidates; selection. VM: best rollout vs that episode's still.
16. `pack` preflight + dry pack on the public split (no upload).
17. Evaluation split: P0–P2 + the full chain on the 20 episodes; submission.

## 6. Run plan (one L40S)

| When | Job | Hours |
|---|---|---|
| VM day 0 | `setup.sh` (BRANCH=zjifan121, RESTORE_RUN=4c2eac31 RESTORE_FRAMES=1); new venvs `sam3`, `daid`; weights | 2 |
| day 1 | S0.1 on ep40 and ep26 (2 × 0.8 h + eval); P0–P2 on the 20 evaluation episodes | 5 |
| day 1 | Quick win: still candidates on stereo-scaled bundles for 20 evaluation episodes → submission #2 (uses existing `s23_load_retarget` + S6 settings) | 16 (overnight) |
| days 2–4 | tasks 6–12 on ep40 and ep26, then on the other public episodes | 12 |
| days 4–5 | tasks 13–15 on ep40 and ep26; gate | 7 |
| day 6 (Oct 27) | references frozen; chain on the 20 evaluation episodes | 8 |
| days 7–9 | S6 three candidates × 20 episodes (≤ 48 h), selection, pack, submission #3 (Oct 30) | 48 |
| Oct 31 | resubmission for verified fixes only; Nov 1 stop | |

After each milestone: `hf_outputs.py upload --run <run>`; delete nothing on the VM before the upload is listed.

## 7. Compliance

- GT firewall: `pathb/paths.py` gains `gt_allowed(ctx)`; every reader of `public/data/*.parquet` or `public/mesh/` goes through it; with `split == "evaluation"` it raises; each stage appends the files it read to `report.json: files_read`.
- Mesh hash S3a → S5 → S7 (`sha256.txt` compared in preflight).
- `manual_inputs.json` per episode lists any hand-chosen parameter (none expected; prompts come from the roster).
- `THIRD_PARTY.md`: Do as I Do (MIT), MV-SAM3D (check license on clone), SAM 3.1 (SAM license), Point2Pose (BSD-3, optional), FoundationStereo/FoundationPose (NVIDIA licenses via the baseline).

## 8. Phase 2: two-object scope (built after all episodes have run through phase 1)

Trigger: every public and evaluation episode has a phase-1 result and submission #3's inputs are frozen. Then, on ep26 first: S4a phases per object + support relation (`support_stacked.py`); S4b keyframe per object with the roster-first object's frame-0 pose as the scene anchor; S4c Do as I Do scene extended to two free bodies + both hands (released code is one hand, one object; `cache/v2/spider_scene.py` slot logic is the reference; 1–2 days); S5 parquet with two `object_body_*` entries; S6 selection adds RPE. Rerun the two-object episodes, compare the waterfall and rollout scores against phase 1 (mode-A merge), keep the better per episode. Also later: EgoFound3R hands, Point2Pose, generalist FlashSAC.

## 9. Workflow (James, 2026-10-22; supersedes the 2026-10-21 version)

**Where things run.** The Mac has no CUDA GPU. CPU stages are built, debugged and run locally on the real ep40 and ep26 data: unit tests, P0, P1, P3/P4, S3c, S4a, S4b, S5 parquet writing, `score_local`, S6 selection, S7 pack/preflight, all adapters. GPU stages (P2, P1b, P1c, P1d, S3a, S3b, S4c, S6 train/eval) are smoke-tested on the L40S on ep40 and ep26 **only**, each as soon as it is written. The cloud is used for exactly two things: those dev-episode smoke tests and the all-episode runs (phases B and D).

**Dev cache.** First cloud job: P0–P2, P1b, P1c, P1d, S3a, S3b outputs for ep40 and ep26 (~15 GPU-min), uploaded with `hf_outputs.py` and restored into `path_b/data/` on the Mac, so every CPU stage develops against real intermediate data. S0.1 runs in the same job.

**Per task.** (1) Local: code, `uv run --project path_b pytest path_b/tests`, run the stage on ep40/ep26 if CPU, commit locally (message starts with the task number). (2) Cloud, GPU stages only: push through the bare repo (`git push --no-verify l40s:/root/v2d/repo.git <branch>`), pull on the VM, run on ep40 and ep26, record numbers and report statuses in `BUILD_LOG.md`, `hf_outputs.py upload`, restore to the Mac if a later CPU stage needs it. Never edit code on the VM. A failing smoke test returns to (1). Task N+1 may be coded while task N's cloud job runs.

**Phases.** A: phase-1 build (tasks 1–15). **A2: S3 observation variant (§11), built and compared on the dev episodes before any all-episode run.** B: phase-1 run on all 20 public then 20 evaluation episodes (tasks 16–17 to the packed parquet). C: phase-2 build (§8) on ep26. D: phase-2 run on every two-object episode; keep the better of phase 1/2 per episode by the S6 rule; repack. E: preflight, final summary, **HOLD before `v2d_submit.py`**. Phases B and D start automatically once their inputs exist; they need a VM (below). During B and D the next phase is coded locally while the run proceeds; the run is checked between tasks and fixed-and-resumed on failure (a stage with a non-ERROR `report.json` is kept).

**VM lifecycle.** Verda bills a stopped instance at full price, so the VM is deleted after each cloud use. After the dev cache and a phase's last smoke test are on HF: tell James the VM can be deleted. Before B and D: write the run's needs and GPU-hour estimate to `BUILD_LOG.md`, tell James "need an L40S", wait for the IP, rebuild with `setup.sh` (`BRANCH=zjifan121_pathb_v3` once on GitHub, else push through the bare repo), restore from HF, start the run in tmux.

**GitHub.** Pushes are batched at phase boundaries (end of A, B, C, D) only: write the phase summary to `BUILD_LOG.md`, ask James once ("push `zjifan121_pathb_v3` and open the PR to `main_buffer`?"), push nothing without that yes, keep working locally meanwhile. `git pull --rebase origin main_buffer` before pushing; PR into `main_buffer`; never force-push.

**Waits.** Exactly three: VM provisioning, GitHub push at a phase boundary, Kaggle upload (HOLD). Everything else is decided by the rules in the session prompt and recorded under "Decisions" in `BUILD_LOG.md`.

## 10. Session facts (2026-10-21)

- Branch: `zjifan121_pathb_v3`, created from local `zjifan121` (which holds the v2 files that task 1 moves to `cache/v2/`). PRs go to `main_buffer`.
- VM: Verda L40S, `ssh root@31.56.109.90`. Add to `~/.ssh/config`: `Host l40s` / `HostName 31.56.109.90` / `User root` / `IdentityFile ~/.ssh/id_ed25519`. First VM step: `path_b/vm/setup.sh` per `path_b/vm/README.md` with `BRANCH=zjifan121 RESTORE_RUN=4c2eac31 RESTORE_FRAMES=1` (the new branch reaches GitHub only at the first phase boundary), then push `zjifan121_pathb_v3` through the bare repo and check it out on the VM. This VM is deleted after the dev cache and phase A's smoke tests are on HF (§9). New venvs on the VM: `sam3` (SAM 3.1) and `daid` (Do as I Do, cloned under `/root/v2d/third_party/`).
- Secrets: `path_b/.env` on the Mac, written by James (git-ignored by the root `.gitignore` `.env` rule; copy to the VM with `scp`; never commit, never print): `HF_TOKEN=<read-only token of HF user JzJz7779, the account granted SAM 3D Objects and SAM 3.1; the same token is in CLAUDE.local.md §6>` and `KAGGLE_API_TOKEN=<James's Kaggle access token, given in chat on 2026-10-21; the kit's uploader reads this variable>`. Uploads to the HF bundle repo need James's write token via `hf auth login` on the VM, as before. Source the file (`set -a; . path_b/.env; set +a`) in every shell that runs P1b (SAM 3.1 download), S3a (SAM 3D), `hf_outputs.py` or `v2d_submit.py`.
- Kaggle uploads: none without James's explicit go-ahead in chat (submission #2 is on hold).

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

### A2.4 Mesh-free tracking (`stages/s3b2_track_points.py`, new; GPU smoke only)
- Primary: Point2Pose (BSD-3; RGB-D + SAM masks → per-frame SE(3) + online TSDF; multi-object; recovers after occlusion). Time-box one day; its code is young.
- Backup: TAPIP3D (Apache-2.0) with known depth/intrinsics/extrinsics → 3D tracks of masked object points → per-frame RANSAC-Kabsch rigid fit to the frame-0 point set; symmetric-equivalent fits resolved by depth residual and continuity (same rule as S3b).
- Output in the same `poses/` layout as S3b so S3c fuses either or both; `track_meta.json` records the source.

### A2.5 Comparison and decision (CPU)
- Rerun S3c → S4 → S5 → S6-select on ep40 and ep26 with: (1) phase-A mesh + FP track; (2) canonical mesh + FP track (FP re-registered to the new mesh); (3) canonical mesh + mesh-free track (if A2.4 ran). Score each with `score_local`; waterfall columns `a2_mesh_fp`, `a2_mesh_points`; CD-O per object via `eval_mesh.py`.
- Decision rule (record under "Decisions"): the variant with the higher mean of (AUC rank, CD-O rank) over the dev episodes becomes S3 for phase B; ties → the deterministic variant. If A2 loses on both AUC and CD-O, keep phase A's S3 and log why.
- Collision body for S4/S6 comes from the chosen mesh (CoACD as before).

**Spec consequences if A2 wins (apply then, not before):** §2 drop `mvsam3d`; §4 S3a/S3b replaced by A2.1–A2.4; §7 `THIRD_PARTY.md` adds Point2Pose / TAPIP3D / Open3D (and any fallback that fired); §5 tasks 10–11 marked superseded. Phase 2 (§8) is unaffected.

Sources (checked 2026-10-22): SAM 3D Objects repo and paper (arXiv 2511.16624); MV-SAM3D (arXiv 2603.11633); Point2Pose (arXiv 2604.10415, github.com/TheWisp/point-to-pose); TAPIP3D (github.com/zbw001/TAPIP3D); Open3D RGBD integration docs; SymmCompletion (github.com/hongyuyann/symmcompletion); AdaPoinTr (github.com/yuxumin/PoinTr); ForeHOI (arXiv 2602.06226); Uni3D (github.com/baaivision/Uni3D); `tools/track3/episodes.json`.
