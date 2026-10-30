# Path B v4 build spec (contact-segmented inversion; dev episodes ep41 + ep13)

Written 2026-10-24 for the coding session. Supersedes `V3_BUILD_SPEC.md`, which moves to `path_b/cache/v3/` together with the v3-only code (§1). Companion pages: artifacts "Path B v4 Design" (the idea, stage table, v3-vs-v4), "Path B v3 Review" (what v3 measured), "Path B Task Queue" (order and budget). Where v4 keeps a v3 stage unchanged, the stage's full interface text is reproduced here so this file is self-contained; the cached v3 spec is history, not a dependency.

**Decisions from James (2026-10-24):** v4 replaces v3 as the pipeline; v3-only code and docs go to `cache/v3/`. **Dev episodes: ep41** (public, `Pan_Solo`, `white_pot`, 299 frames; the episode v1/v2 measured) **and ep13** (public, `Cup_In_Dish_Rack`, `blue_cup` + `plastic_dish_rack`, 190 frames; the cup ends resting in the rack — the two-object support case). Hands stay HaMeR (§7). Nov 1 stop. **No v3 run on the 40 episodes**: the GPU goes to the dev cache, S1 on all episodes, and v4 (§6). Push to `zjifan121_pathb_v3` after every passing test or smoke test; no merge to `main_buffer` without James; no Kaggle upload without James.

**What v4 changes, in one paragraph.** v3 asked a per-frame tracker where the object is in every frame and smoothed afterwards; on the one real run it scored no better than a still object (ep40: 0.381–0.400). The oracle (§S0.2) says the lost points are object *position while it moves*. v4 splits each clip into **static**, **stable-grasp** and **unstable** segments and solves each with the rule that holds there: one pose per static segment (multi-frame ICP of the canonical mesh, then settled in MuJoCo so it rests on the table); over a grasp, the hand track carries the object — one hand→object offset fitted on the whole segment as the starting guess, then small per-frame corrections with the contact set fixed (ROHIT's relaxation: the object may shift inside the hand); unstable stretches are short bridges. Everything new is CPU. The first v4 task measures whether our wrist track is steady enough for any of this to pay (§S0.3).

Interfaces were read from the code on 2026-10-21/24 (`main_buffer` @ `e61fc18`; `zjifan121_pathb_v3` @ `ba9b051f`; baseline `8c778f9`; FlashCHORD in `robotic_grounding/flash_chord`; kit in `assets/data/v2d_submission_kit`). Where a parameter is a proposal, it says "default:" and the acceptance test decides.

## 0. Rules

1. Never change anything under `baseline/` or `assets/data/v2d_submission_kit/`. Baseline modules are called as they are (host wrappers launch the GHCR containers; `pathb/paths.py` → `HOST_PY`, `RECON`). FlashCHORD changes, if any, go into `path_b/third_party/flash_chord_patch/` and are applied in the `flash_chord` venv.
2. Ground truth and public scans are read only inside `score_local` and inside stages that check `ctx.split == "public"`. With `--split evaluation` any such read raises (GT firewall, §7).
3. Every stage writes `report.json` through `pathb/report.py` (`Metric(name, value, unit, kind, ok=(lo,hi), warn=(lo,hi), note)` → PASS/WARN/FAIL/INFO; `StageResult(metrics, outputs, skip)`); `pathb/runner.run_stage` adds stage, status, episode, split, commit, time, GPU peak. `loss_report.md` names the first stage whose status is FAIL or ERROR.
4. A stage is rerun only with `--force` or when its report is missing or ERROR.
5. One heavy model on the GPU at a time. Stages are subprocesses into their own environment (§2); `pathb` itself imports only numpy/scipy/cv2/trimesh/open3d.
6. A silent fallback is a bug: a stage whose primary model does not load FAILs (P1b fell back to SAM2 silently once).

## 1. Repository changes (first commit of the v4 session)

Move to `path_b/cache/v3/` (git mv, keep history; `cache/v3/README.md` lists each file and why):
- `V3_BUILD_SPEC.md`, `SPEC_AUDIT.md` (audits the v3 spec; a v4 audit starts fresh).
- `pathb/stages/s3a_mesh.py`, `tests/test_s3a.py` (SAM 3D mesh: FAILed its bars; replaced by the canonical mesh).
- `pathb/stages/s3b2_track_points.py` (A2.4 mesh-free tracking: never ran; per-frame tracking is what v4 removes).
- `pathb/phase2.py`, `tests/test_phase2.py` (the Do as I Do two-body scene; 0.500 either way on ep26 stand-ins; v4 handles two objects per segment). Keep `pathb/drivers/support_relation.py` (reused by S3c for "resting on another object").
- `pathb/standin.py` and `run.py standin` (stand-in GPU outputs; the real dev cache replaces them).
- `pathb/a2.py` and `run.py a2` (the A2.5 comparison; decided — see §11). `pathb/canonical.py`, `s3a2_fuse_scan.py`, `s3a2_register.py` stay: they are S3a.
- `reports/dev_analysis_2026-10-23*` stay in `reports/` (history); `pathb/analysis.py` stays and gains the ep41/ep13 defaults.

Code changes in kept files: `pathb/reference.py` loses the `tracking` candidate (`CANDIDATES = ("inverted", "keyframe", "still")`, build order); `pathb/runner.STAGES` becomes the list below; `run.py` loses `standin`/`a2`, gains `wristcheck` (S0.3).

New `runner.STAGES` (execution order; names are module names in `pathb/stages/`):

```
p0_ingest  p1_calibrate  p2_depth                       (exist; S1)
p1b_masks  p1c_camera  p1d_hands                        (S1; P1d gains fingertip/mesh export)
p3_scale_bridge  p4_export                              (exist; S2)
s3a2_fuse_scan  s3a2_register                           (S3a; canonical meshes via run.py canonical)
s3b_segments                                            (S3b, new)
s3b_track                                               (FoundationPose; seed only, runs when s3a2_register is WARN/FAIL)
s3c_invert  s3d_settle                                  (S3c/S3d, new; s3c_fuse removed)
s4a_phases  s4b_reference  s4c_retarget                 (S4; s4a is a view of s3b_segments; s4c gated, off by default)
s5_task                                                 (S5)
s6_train  s6_eval  s6_select                            (S6)
s7_pack                                                 (S7; run.py pack)
s23_load_retarget                                       (kept: mode A runner for the still fallback)
```

`waterfall` columns: `raw`, `ref_inverted`, `ref_keyframe`, `ref_still`, `realised_*`, `rollout_*`, `@scan`; the v3 columns `fused`, `ref_tracking`, `a2_*`, `p2_*` are dropped.

## 2. Environments

| Env | Python | Used by | Setup |
|---|---|---|---|
| `path_b/.venv` (uv) | 3.11 | all `pathb` code, P0–P4, S0.3, S3a2, S3b, S3c, S4a/b, S5, S6-select, S7 | `uv sync --project path_b`; open3d 0.19.0 (0.20's macOS wheel needs Homebrew libusb), scikit-image, scipy |
| baseline host venv + GHCR images (`tools/verda/images.lock`) | 3.11 | FoundationStereo, DROID-SLAM, WiLoR/HaMeR, FoundationPose (seed only) | exists on the VM (`setup.sh`) |
| `sam3` venv | 3.12 | P1b SAM 3.1 (`facebook/sam3.1`, gated; `use_fa3=False` on the L40S) | exists; no silent fallback (rule 6) |
| `daid` venv | 3.12 | S3d settle (CPU MuJoCo), S4c Do as I Do (gated) | github.com/malik-group/do-as-i-do `retargeting/`; MuJoCo Warp, mink, coacd |
| `flash_chord` venv | 3.11 | S6 train/eval, S5 replay | exists (`/root/v2d/venvs/`); MuJoCo 3.8.1 / mujoco-warp 3.8.1 / warp 1.14 |
| `tools/.venv` | 3.11 | `score_local`, `run_load_retarget`, kit packer/uploader, `support_stacked` | exists |

Removed from v3: `mvsam3d`, the SAM3D container's use in S3a.

## 3. Conventions (all stages)

- Frames: colour camera `ego_cam_a`, 0-indexed, `%06d`. Stereo frames are offset by `alignment.json: offset` (P2); every stage that mixes stereo and colour applies it once, through `pathb.calib.stereo_frame(i)`.
- Camera frame: OpenCV (+x right, +y down, +z forward), metres. World: table top z = 0, +z up, +y = mean camera forward projected on the table.
- Poses on disk: `Transform3d` JSON (`rotation` wxyz, `translation`, `scale`) per frame, as the baseline writes; arrays in memory and in `.npz` as (N, 4, 4) float64 `*_to_world`. Quaternions in Kaggle/evaluator parquets are **xyzw**.
- Depth PNG: uint16 inverse depth `px = 65535/(d+1)`; "no depth" = pixel 0, which decodes as infinity — every consumer must treat it as invalid (P1d does since 2026-10-23).
- Masks: 8-bit PNG, 255 = object, one folder per track.
- Hands: MANO parameters per frame in the baseline's `hamer_aligned_filled/<track>/%06d.json` layout (with `camera: {cam_t, intrinsics}` as P1d writes since 2026-10-23); world joints through `tools/track3/read_mano.py`.
- Segments: `segments.json` = list of `{kind: static|grasp|unstable, start, end, hand: L|R|null, fingertips: [[hand, finger], …], support: <object>|table|null}`, frame ranges inclusive.
- Object frame: every pose of an object is expressed in its **canonical mesh frame** (A2.3), so mesh and poses stay consistent as the kit requires.

## 4. Stages

### S0 — Oracles and the wrist check

**S0.2 Regime split (done; Edmond's `oracle_2026-10-17.json`, 20 public episodes).** Means: `place` 0.388 / 12.4 cm (our scene as submitted); `still_gt0` 0.447; `place_pos_gt` (true positions, our rotations) **0.741 / 1.9 cm**; `place_rot_gt` 0.404; `place_gt_when_moving` 0.762; `recon_hold_gt` 0.337. Position error is the loss and the moving regime holds the headroom. **Dev-episode floors:** ep41 still 0.356, true positions 0.909, moving-frame truth 0.882; ep13 (scene) still 0.416, true positions 0.877, moving-frame truth 0.781. **New variant to run (James, 2026-10-24, from the review):** `place_pos_rot_gt` (true positions *and* rotations) vs `place_pos_gt`, per episode — the size of the rotation loss once position is fixed. Implemented in `run.py oracle --variants`, public split only.

**S0.1 GT-reference policy (open; ep41 and ep13).** `task.parquet` with the ground-truth object trajectory and our P1d hands, through `run_load_retarget.py` (public scan as mesh, this test only). Train with the `still_submit.sh` settings (60M), evaluate with the kit, score with `score_local`. **Decision rule:** rollout(GT reference) < the episode's still (ep41 0.356, ep13 0.416) → the S6 recipe is the first problem; ≥ 0.7 → the reference is the whole problem. This run is also the only place the optional 5 cm object stop (§S6) is tried.

**S0.3 Wrist check (`run.py wristcheck`, new; CPU; the first v4 task).** For ep41 and ep13 from the dev cache: on frames where a hand rests on or next to a *static* object (S3b static segment with a fingertip within 5 cm of the object points — the hand is still, the object is still), compute the wrist-to-object-centroid distance per frame and its standard deviation σ over each such window; also the share of carry frames with a valid wrist (`coverage_carry`). Out: `wristcheck.json`. **Gate:** σ ≤ 2 cm on both episodes → build S3c as specified; 2–3 cm → build S3c but expect the carry bar (§S3c) to fail often, keyframe fallback carries the weight; > 3 cm → S3c is not built; v4 keeps only the static/settle gains and the time goes to S6. `coverage_carry` < 0.9 → the hand-gap rule (§S1 P1d) is mandatory. Record the decision in `BUILD_LOG.md`.

### S1 — Evidence (v3 stages, kept)

**P0, P1, P2** unchanged (`stages/p0_ingest.py`, `p1_calibrate.py`, `p2_depth.py`). Outputs: `p1_calibrate/{rect_left,rect_right,cam_a}/`, `calib.json`; `p2_depth/{depth_rect,intrinsics_rect,planes.json,alignment.json,depth_a_half/}`. Bars: row offset < 0.5 px; stereo offset found.

**P1b masks** (`stages/p1b_masks.py`): SAM 3.1 (text prompts from `episodes.json` → `objects[].prompt`; hands "left hand", "right hand") in the `sam3` venv, propagation over the whole clip both directions; seed frame = first frame with object box confidence ≥ 0.35 and mask area ≥ 0.5 %. Fixes of 2026-10-23 kept: `init_state` receives only declared arguments; the two hand seeds must differ (mask IoU ≤ 0.5); a hand mask that jumps to a second blob is rejected in favour of the blob nearest the previous frame. Out: `p1b_masks/obj_<k>/%06d.png`, `hand_L/`, `hand_R/`, `prompts.json`, `masks_overlay.mp4`. Acceptance: object mask in ≥ 95 % of frames (PASS) / 80 % (WARN); **"SAM 3.1 ran" is a FAIL bar, not WARN** (rule 6); left/right overlap ≤ 0.5.

**P1c camera** (`stages/p1c_camera.py`): baseline `v2d_droid_slam/lib/video_to_slam.py` with `--input_depth_folder` and `--align_to_depth_folder` on a masked copy of `depth_a_half/` (hand and object pixels = 0). Out: `p1c_camera/camera_to_world/%06d.json`, `trajectory.txt`, `camera_report.json`. Acceptance: median per-frame speed < 1.5 m/s; table-plane height spread over the rest window < 1 cm (PASS) / 2 cm (WARN); DROID vs stereo depth median ratio within 3 %. ep40 real run: PASS, ~60 s. **Owed:** same-episode comparison against the team bundle's camera (Task Queue #3, now on ep41).

**P1d hands** (`stages/p1d_hands.py`, changed): WiLoR → HaMeR → `align_hands` (stereo depth, full resolution) → `tracks_interpolate`, through the baseline's late-hands machinery; records carry `camera: {cam_t, intrinsics}`; non-finite alignment fits are rejected and refilled from neighbours (gap ≤ 15). **v4 additions:** `fingertips_world.npz` (N, 2, 5, 3), `hand_verts_world.npz` (N, 2, 778, 3) and `wrist_world.npz` (N, 2, 4, 4) from the aligned MANO via `tools/track3/read_mano.py`; `coverage.json` (share of frames with a wrist, overall and inside each S3b grasp segment, written after S3b runs — P1d exposes a `coverage(segments)` helper). **Hand-gap rule:** a wrist gap ≤ 15 frames inside a grasp is interpolated (Hermite/slerp) and flagged; a longer gap splits the grasp segment (S3b consumes the flag). Acceptance: per hand, pose on ≥ 90 % of masked frames; `hand_scale` 0.8–1.2 (PASS) / 0.6–1.5 (WARN); wrist speed < 2 m/s on 99 % of frames; `coverage_carry` reported (INFO; the gate is in S0.3). ep40 real run: PASS, 6 min, scales 0.84 / 0.81.

### S2 — One metric world (v3, kept)

**P3** (`stages/p3_scale_bridge.py`): factor = stereo camera height / P1c camera height over the rest window, expected 1.00 (PASS 0.97–1.03, WARN 0.9–1.1, FAIL otherwise → P1c is wrong); writes `scale.json` with `table_plane`, `gravity`. **P4** (`stages/p4_export.py`): v4 bundle layout `result.npz` with `camera_to_world_transform`, `hand_*`, `table_plane`, `gravity`; object track and mesh are filled by S3.

### S3a — Canonical mesh (A2.1–A2.3, kept; decided 2026-10-23)

- **Per-clip partial scan** (`stages/s3a2_fuse_scan.py`): Open3D `ScalableTSDFVolume` (voxel 4 mm, sdf_trunc 16 mm) over rest frames where the object mask area ≥ 1 % and no hand mask overlaps it; depth outside the mask = 0; world frame; marching cubes → `partial_scan.ply`. Acceptance: ≥ 30 frames integrated (PASS) / 10 (WARN); surface RMS vs held-out stereo depth ≤ 5 mm (PASS) / 10 mm (WARN); public: one-sided Chamfer partial → scan ≤ 0.3 cm.
- **Canonical mesh per object** (`run.py canonical --object <name>`, `pathb/canonical.py`): pairwise ICP (scale fixed) of **every clip of the object in both splits (videos only, never GT or scans)** onto the clip with the largest surface; re-fuse; fit the symmetry (revolution axis for white_pot body, white_pot_lid, blue_cup, beige_cup, water_pitcher body; mirror plane for mini_dust_pan and the wooden pieces); complete by symmetry; screened Poisson → watertight; ≤ 20k faces → `canonical/<object>/mesh.glb` + `canonical.json`. Objects without a usable symmetry (plastic_dish_rack, mini_sweeper, wooden_spoon): fused TSDF + mirror if confident, else fused alone, flag `coverage < 0.6`. Fallbacks in order: SymmCompletion → AdaPoinTr → ForeHOI; record which fired. **Pooling is mandatory before the evaluation episodes run**: one clip (ep40) gave coverage 0.58 and CD-O 0.53 cm.
- **Registration** (`stages/s3a2_register.py`): ICP of the rest-before partial scan onto the canonical mesh → frame-0 pose in the canonical frame; bar: the placed mesh explains ≥ 0.8 of the partial's points. WARN/FAIL → S3b_track (FoundationPose) runs to seed the orientation.
- Per-episode outputs keep the downstream names: `s3a_mesh/mesh.glb` (= the canonical mesh), `coacd/` (threshold 0.05), `mass.json` (`pathb/density.py`), `sha256.txt`.
- Acceptance: watertight; extents within 0.8–1.25× of the fused partial; public: CD-O vs scan via `tools/track3/eval_mesh.py` < 0.3 cm (PASS) / 0.5 (WARN). Dev objects: white_pot, blue_cup (revolution), plastic_dish_rack (fused only).

### S3b — Contact segmentation (`stages/s3b_segments.py`, new; `pathb/segments.py` pure function)

- In: `p2_depth/depth_a_half`, `p1b_masks/obj_<k>`, `p1c_camera`, P1d `fingertips_world.npz`, `wrist_world.npz`, hand-gap flags; `s3a2_register` frame-0 pose; canonical mesh.
- Per frame: masked object points in world (subsampled to ≤ 2 000). Per-frame fingertip-to-object distance `d_tip` = min over object points.
- **Thresholds from the data, not fixed** (review item 6): `near = 2 × median(d_tip)` over *certain-contact* frames = frames where the object centroid moves ≥ 1 cm/frame in the same direction as the wrist (the hand is carrying it); if fewer than 10 such frames exist, `near = 2.5 cm` (default). `far = 2 × near`.
- Rules (per frame, then hysteresis): **static** = median displacement of the object points over a 10-frame window < 5 mm and no fingertip within `far`; **grasp** = ≥ 2 fingertips of one hand within `near` with the same fingertip set (hand × finger) for ≥ 5 frames; **unstable** otherwise. Hysteresis: a frame changes kind only when 3 consecutive frames vote for the new kind. Static runs < 5 frames merge into their neighbours; grasp segments are split at hand gaps > 15 frames (P1d flag).
- **Support**: for each static segment, `support` = `table` if the object's lowest point is within 1.5 cm of z = 0, else the other object whose canonical mesh (at its pose over the window) is within 1.5 cm below (the team's `support_stacked` rule through `drivers/support_relation.py` is the check). ep13's cup in the rack is the test.
- Out: `segments.json`, `phases.json` (S4a's layout: rest-before/carry/rest-after derived from the first and last grasp; `pathb/phases.py` stays for the fallback), `segments_report.json` (thresholds used, segment table).
- Acceptance: ep41: first grasp starts within ±5 frames of the v1 `first_reach` (18–19) and the last grasp ends within ±5 of `last_release` (264); ep13: the rack is one static segment with `support: table`; the cup has ≥ 1 grasp segment and its final static segment has `support: plastic_dish_rack`. Objects never grasped: whole clip static (`static: true`).

### S3c — Segment solving (`stages/s3c_invert.py`, new; `pathb/invert.py`)

Tools: Open3D ICP; Open3D `RaycastingScene` for rendered mesh depth (CPU, no GL renderer); scipy `least_squares` (soft-L1). **Time one full episode before committing** (300 frames × ≤ 5 steps × ray-cast); if > 20 min on the Mac, subsample to every 2nd frame in the grasp fit.

- **Static segment**: TSDF of the segment's masked depth (A2.1 code), ICP of the canonical mesh onto it (scale fixed), initialised from `s3a2_register` (first segment), the previous segment's end pose, or S3b_track's seed; symmetric objects: candidates = pose × symmetry group (`pathb/symmetry.py`), keep the lowest point-to-mesh residual → one pose. Bar: residual median ≤ 5 mm.
- **Grasp segment, pass 1 — the prior**: unknown `T_hand→object` (6 DoF) for the grasping hand's wrist (two-handed: the hand with more fingertips in contact). `object(t) = wrist(t) · T`. Residual over the segment: (a) rendered canonical-mesh depth vs stereo depth inside the object mask; (b) contact fingertips to the mesh surface (signed distance → 0); (c) end poses equal the neighbouring static poses (weight 10). Initialise `T` from the preceding static pose and the wrist at the segment's first frame.
- **Grasp segment, pass 2 — per-frame refinement (ROHIT's relaxation)**: starting from `wrist(t) · T`, refine each frame's pose on residuals (a) + (b) with the *contact set fixed* and the change per frame limited to 5 mm / 3° relative to the previous refined frame (a motion prior, not a hard weld). Default limits; the acceptance test decides. **Slip guard:** if residual (a) exceeds 1.5 cm for 10 consecutive frames after pass 2, split the segment there, re-run S3b's grasp detection on the parts, re-solve; the split becomes an unstable bridge.
- **Unstable segment**: cubic Hermite (position, zero end velocity) + slerp between the neighbouring segment end poses, then ≤ 5 Gauss-Newton steps of per-frame refinement on residual (a) where the mask covers ≥ 1 % of the image; speed ≤ 1.5 m/s.
- Out: `s3c_invert/object_to_world.npz` (N,4,4), `invert_report.json` (per segment: kind, pass-1 residuals, pass-2 median residual and drift, iterations, `failed`), the bundle `result.npz` completed (`object_to_world_transform`, `object_to_camera_transform`, `mesh.obj` = canonical mesh).
- Bars: grasp residual (a) median ≤ 1 cm on ≥ 50 % of the segment's frames, else `failed` → S4b uses `keyframe` interpolation for that segment; static residual ≤ 5 mm; no frame-to-frame jump > 3 cm outside splits. Public: waterfall `ref_inverted` ≥ `ref_keyframe` on ep41 and ep13 — **the v4 go/no-go** before any v4 GPU hour.

### S3d — Settle (`stages/s3d_settle.py`, new; `daid` venv, CPU MuJoCo)

For each static segment: load the CoACD pieces, place the object at the ICP pose on its `support` (table plane z = 0, or the supporting object's canonical mesh at its own settled pose — supports are settled first), simulate 0.5 s with gravity, read the resting pose. Accept if it moved < 1.5 cm and rotated < 10°, else keep the ICP pose and flag `unsettled`. Replace the static pose; re-run S3c's end-pose terms once. Bar: penetration into the support ≤ 2 mm after settle; ep13's cup must settle *in* the rack, not through it.

### S3b_track — FoundationPose seed (v3 S3b, demoted)

Runs only when `s3a2_register` is WARN/FAIL. `pathb/drivers/fp_track.py` in the FoundationPose container with the canonical mesh (`--skip_object_scale_estimation` equivalent), the v3 recovery rule (symmetry candidates + top-5 hypotheses, scored by logit + log IoU − λ·rotation − μ·depth residual; λ 2.0, μ 50). Output is a seed orientation for the first static ICP only; not a track.

### S4 — Reference construction

**S4a phases** (`stages/s4a_phases.py`): now a view of `segments.json` (rest-before = frames before the first grasp, carry = first grasp start → last grasp end, rest-after = after); the v3 wrist-within-10-cm rule (`pathb/phases.py`) remains the fallback when S3b FAILs. Acceptance: carry ≥ 10 frames unless `static: true`.

**S4b reference** (`stages/s4b_reference.py`): candidates **`inverted`** (S3c + S3d poses; static segments held exactly), **`keyframe`** (v3 rule: rest poses held, carry = trusted frames else Hermite/slerp interpolation; the fallback for episodes where any grasp segment `failed`), **`still`** (rest-before pose on every frame). `tracking` removed (identical to keyframe on every v3 run). Hands: P1d MANO in the world, cleaned with `run_load_retarget` hand cleaning (import, do not copy). Out: `s4b_reference/<candidate>/` bundles; `reference_report.json`. Acceptance (public): `score_local` per candidate → `ref_inverted`, `ref_keyframe`, `ref_still`.

**S4c retarget** (`stages/s4c_retarget.py`; Do as I Do; **gated, `s4c.enabled: false` by default**): the v3 adapter (`pathb/daid.py`, `sharpa_map.json`), cut MPC settings only (256 samples, 1.6 s horizon, ≤ 16 iterations, ≈ 42 min per object), one candidate (`inverted`). Status 2026-10-23 on ep40: FAIL (object never moved; read-back vs replay 43 mm against a 5 mm bar). Enabled only after both are fixed on ep41 (time-box one day, Task Queue #13). Gate (public): realised ≥ kinematic − 0.01 AUC; penetration ≤ 5 mm; displacement before the first grasp ≤ 1 cm; read-back ≤ 5 mm; the object moves ≥ 5 cm when the reference moves it.

### S5 — Task file (`stages/s5_task.py`)

**Mode A (default):** `s23_load_retarget` on the chosen kinematic bundle (`run_load_retarget.py --no-rest-hold`; `--still-object` for `still`). **Mode B** (S4c passes its gate): FlashCHORD's ManoSharpa parquet from S4c output with `pathb/fk.py` (numpy FK; reproduces the 67 `robot_*_frames` to 1e-7 m). Scene files: URDF + support USD from the canonical mesh through the baseline `mesh_to_usd`. Acceptance: kinematic replay runs to the end; runner checks (no link > 2 cm inside the object, seating gap < 1 cm — v3 saw 1.6 cm on ep40; S3d should close it).

### S6 — Policy

What the baseline already does (read 2026-10-24 in `robotic_grounding/flash_chord`): `configs/experiment/sharpa_flash_sac.yaml` → `override /action: residual_hand_pose` (the actor outputs a correction added to the reference wrist pose, `runtime/actions/hand_pose.py` L275–285) and `override /termination: tracking` (`configs/termination/tracking.yaml`: wrist_position 0.15 → team override 0.3, object_position **0.10**, object_orientation **0.50**, all enabled; termination reward −100 in `curriculum/tracking_250m_mixed_reset.yaml`). **So "residual-on-prior" and "object-error termination" are not v4 changes.** `evaluation/harness.py` L74 disables early termination at evaluation but reuses the checkpoint's thresholds as the failure criteria in the metrics JSON.

- **s6_train**: FlashSAC `sharpa_flash_sac` with Edmond's overrides (`tools/track3/jobs/still_submit.sh`: `task.motion_speed=1.0 sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3 curriculum.thresholds=[…60006400]`), `sim.njmax=512 sim.nconmax=512`, in `tmux`. Measured: 50 000 env-steps/s on the L40S → **60M ≈ 20 min**. Per episode: `inverted` (or `keyframe` if it fell back) and `still` at 60M; then **250M on the selected candidate**, 2 seeds when GPU time allows, 1 otherwise. Verify at the first run that no override removed `residual_hand_pose`. **Optional (S0.1 first):** `termination.object_position.threshold=0.05`; adopted only if the S0.1 rollout improves and training does not destabilise.
- **s6_eval**: kit semantics (`evaluation.world_count=1`, `reset_mode=explicit`, `object_trajectories_output`); one parquet + JSON per candidate/seed.
- **s6_select**: public: `score_local` against ground truth → best AUC, ties → lower MPPE. Evaluation (no GT): lowest MPPE of the rollout against our own reference, with failure frames counted at **pinned** thresholds (10 cm / 0.5 rad, written in `s6_select`, never read from the checkpoint). Writes `selection.json`.
- Acceptance: on ep41 and ep13 the selected rollout beats the still rollout and loses ≤ 0.1 AUC to its own reference's playback; S0.1 bounds the policy tax first.

### S7 — Pack and submit (`run.py pack`)

Preflight (`pathb/preflight.py`): 20 evaluation episodes have `selection.json` and the rollout parquet; `s3a_mesh/sha256.txt` equals the hash of the mesh inside S5's URDF and of the packed mesh; no evaluation `report.json` lists a GT/scan path in `files_read`; `git status` clean and HEAD pushed. Pack: `tools/pack_submission.py --track track_3 …`; review every `MESH WARNING`. Upload: `submit/v2d_submit.py … --wait`, **only on James's go**; receipt to `path_b/data/submissions/<tag>/`; tag `submit/b-YYYY-MM-DD-N`; checkpoints, task parquets, meshes to HF via `vm/hf_outputs.py upload`.

## 5. Task list (in order; each task ends with its test; push after each)

CPU tests run on the Mac (`uv run --project path_b pytest path_b/tests`); GPU smoke tests on the L40S on ep41 and ep13 only.

1. §1 moves and code changes; `cache/v3/README.md`; tests green with `tracking` gone and `standin`/`a2` removed. Push.
2. P1d export (`fingertips_world`, `hand_verts_world`, `wrist_world`, `coverage`) + hand-gap rule. Test: fixture hands → fingertips; synthetic gap → interpolated/flagged. GPU smoke on ep41/ep13 (the dev cache run, Task Queue #2–3).
3. `run.py oracle --variants place_pos_rot_gt` on the public split (CPU, `score_local`). Record per-episode and mean in `BUILD_LOG.md`.
4. **S0.3 wrist check** on ep41/ep13 from the dev cache. Record σ and `coverage_carry`; apply the gate; decide S3c's scope in `BUILD_LOG.md` "Decisions".
5. `pathb/segments.py` + `s3b_segments`. Tests: synthetic fingertips/object → expected segments and thresholds; ep41 reach/release within ±5 of 18–19 / 264; ep13 cup `support: plastic_dish_rack`.
6. `pathb/invert.py` static + grasp pass 1 + pass 2 + unstable; `s3c_invert`. Tests: GT trajectory with synthetic depth → ≤ 3 mm; a synthetic in-hand slip is absorbed by pass 2 and a large one triggers the split; ep41 and ep13 → `ref_inverted` column. Time one episode.
7. `s3d_settle`. Test: a floating cylinder settles to the plane; ep13's cup settles in the rack.
8. S4b `inverted` + fallback wiring; waterfall columns. **Go/no-go:** `ref_inverted` ≥ `ref_keyframe` on both dev episodes (public GT). Record.
9. S6: verify `residual_hand_pose`; pinned thresholds in `s6_select`; 250M stage; optional 5 cm stop behind S0.1. One GPU smoke hour.
10. S0.1 on ep41 and ep13 (first GPU window, with the dev cache); then **S1 on all 40 episodes** (`run.py run --stages p0_ingest … s3a2_fuse_scan --episodes all`, per-episode HF upload) and `run.py canonical` for the 10 objects on the Mac.
11. v4 references for all 40 episodes (CPU); S6 as §4 (`inverted` + `still`; `keyframe` + `still` if task 8 was a no-go); pack + preflight; **HOLD** for James.

## 6. Run plan (one L40S; Nov 1 stop; **no v3 all-episode run** — James, 2026-10-24)

The v3 chain is not run on the 40 episodes. The GPU goes to: the dev cache, S0.1, the front-end check, **the evidence stages (S1) on all 40 episodes** — v4 needs them anyway, and the canonical-mesh pooling needs every clip's partial scan — and the v4 policy window. The only submission is v4's. If v4 fails its go/no-go, the fallback is the `keyframe` + `still` candidates (v3's reference logic, kept in S4b) through the same S6 on the same S1 outputs — ≈ 27 GPU-h, still inside the window.

| When | Job | GPU-h |
|---|---|---|
| Oct 24 | James: new L40S, rotate HF write token, delete old deploy key. Session: push the Mac-only commits, `phase_b.sh` per-episode HF upload, task 1 | 0 |
| Oct 24–25 | `new_instance.sh`; **dev cache for ep41 and ep13** (P1b, P1c, P1d with the new exports, partial scans; ~30 min) → HF → Mac; S0.1 on both; front-end comparison (team bundle vs v3 S1) on ep41 | 3 |
| Oct 25–26 | **S1 on all 40 episodes** (P0–P2 restored from `4c2eac31` where present, P1b, P1c, P1d, `s3a2_fuse_scan`; ≈ 12 min per episode), each episode to HF as it finishes → Mac pools the canonical meshes (`run.py canonical`, all 10 objects) | ≈ 10 |
| Oct 26–28 | VM deleted (Verda bills stopped instances) unless task 9's smoke hour is ready; Mac: tasks 3–8 (oracle variant, wrist check, S3b, S3c, S3d, go/no-go on ep41/ep13) | 1 |
| Oct 28–30 | new L40S. **Go:** v4 references for 40 episodes (CPU, from the S1 outputs), S6 `inverted` + `still` × 60M on all 40, then 250M on the chosen reference for the 20 evaluation episodes. **No-go:** `keyframe` + `still` × 60M on all 40 (≈ 27 h), 250M on the winner if time allows | ≈ 55 |
| Oct 31 | pack + preflight → **submission #3 on James's go**; review page; BUILD_LOG summary. If the policy run overran: evaluation episodes only (≈ 27 h) | |
| Nov 1 | stop | |

After each finished episode: `hf_outputs.py upload --run <run> --episode <n>`; delete nothing on the VM before the upload is listed.

## 7. Compliance

- GT firewall: `pathb/gt.py` / `paths.gt_allowed(ctx)`; every reader of `public/data/*.parquet` or `public/mesh/` goes through it; `--split evaluation` raises; each stage appends `files_read` to `report.json`. Canonical-mesh pooling reads videos of both splits, never GT or scans — the firewall test covers `run.py canonical`.
- Mesh hash S3a → S5 → S7 (`sha256.txt` in preflight).
- `manual_inputs.json` per episode lists any hand-chosen parameter (none expected; prompts come from the roster; S3b thresholds are data-derived and recorded).
- `THIRD_PARTY.md`: Do as I Do (MIT), SAM 3.1 (SAM licence), Open3D (MIT), MuJoCo (Apache-2.0), FoundationStereo / FoundationPose / WiLoR / HaMeR / MANO via the baseline as the organiser ships them.
- **Hands: HaMeR; EgoFound3R is not used** (James, 2026-10-24). EgoFound3R's README (github.com/BAp3r/EgoFound3R) releases code and weights under CC BY-NC-SA 4.0, "academic research and educational use only, commercial use is not permitted"; the challenge pays prizes and requires winners to release code (FAQ: Apache 2.0): the ShareAlike clause conflicts with that release and the NC clause is a risk for a prize-winning entry.

## 8. Two objects

Native in v4: segments, inversion and settle run per object; `support` (S3b) puts a static segment resting on another object into that object's frame, and S3d settles supports first. S5 writes two `object_body_*` entries; S6 selection adds RPE; the mode-A merge (`run_load_retarget.py --merge`) assembles the scene for scoring. The v3 phase-2 Do as I Do two-body scene is cached (§1). Dev case: ep13.

## 9. Workflow

**Where things run.** The Mac has no CUDA GPU. CPU stages (P0, P1, P3/P4, S0.3, S3a2, S3b, S3c, S3d, S4a/b, S5 parquet, `score_local`, S6-select, S7) are built and run locally on the real dev cache of ep41 and ep13. GPU stages (P2, P1b, P1c, P1d, S3b_track, S4c, S6) are smoke-tested on the L40S on ep41 and ep13 only, each as soon as written. The cloud is used for dev-episode smoke tests, the dev cache, and all-episode runs.

**Per task.** (1) Local: code, tests, run on ep41/ep13 if CPU, commit (message starts with the task number); **tests pass → `git push origin zjifan121_pathb_v3`**. (2) GPU stages: the VM pulls `zjifan121_pathb_v3` from GitHub (bare-repo push is the fallback), runs on ep41 and ep13, numbers and statuses to `BUILD_LOG.md`, `hf_outputs.py upload`, restore to the Mac if a CPU stage needs it; **smoke test passes → commit the BUILD_LOG line and push immediately, before any single- or all-episode run**. Never edit code on the VM.

**GitHub.** Personal branch `zjifan121_pathb_v3` only. No PR and no merge into `main_buffer` until James says so; no rebase onto `main_buffer` on your own (sync is Task Queue #14, with his yes). Never force-push.

**Episode runs.** `phase_b.sh` uploads each finished episode to HF before starting the next; the full run starts in the same turn as the passing smoke test.

**Waits.** Exactly two: VM provisioning (James gives the IP), Kaggle upload (HOLD). Everything else is decided by the rules here and recorded under "Decisions" in `BUILD_LOG.md`. If James is away, take the most reasonable reading, record it, continue.

## 10. Session facts (2026-10-24)

- Branch `zjifan121_pathb_v3`: Mac at `ba9b051f` + the uncommitted spec files; GitHub at `89f44903` — push first.
- VM: the 2026-10-21 instance is gone; its IP now answers with a different SSH host key — **do not accept it**. James provides the next L40S IP; `~/.ssh/config` `Host l40s`. First VM step: `path_b/vm/new_instance.sh` (`BRANCH=zjifan121_pathb_v3`, `RESTORE_RUN=4c2eac31 RESTORE_FRAMES=1`), ≈ 2 h.
- Secrets: `path_b/.env` on the Mac, written by James (git-ignored; `scp` to the VM; never commit, never print): `HF_TOKEN`, `KAGGLE_API_TOKEN`; HF uploads need James's write token via `hf auth login`. **The old VM's disk held `.env` and a read-only deploy key: James rotates the HF write token and deletes the deploy key before the next instance.** `set -a; . path_b/.env; set +a` in every shell that runs P1b, `hf_outputs.py` or `v2d_submit.py`.
- Kaggle uploads: none without James's explicit go-ahead in chat.
- Dev episodes changed from ep40/ep26 to **ep41/ep13** (James, 2026-10-24). Consequences: the dev cache is built for ep41/ep13 (the ep40/ep26 cache was lost with the VM, so no extra cost); S3b acceptance numbers are ep41's v1 phases (18–19 / 264) and ep13's support case; S0.1 and the front-end comparison move to ep41/ep13; ep41 is rerun despite the 2026-10-21 note.

## 11. What v3 measured (record; details in the cached spec and "Path B v3 Review")

Stand-in runs (team bundle fed into v3 CPU stages), ep40: raw 0.460, fused 0.519, `ref_keyframe = ref_tracking` 0.540, still 0.400, **A2 canonical mesh + team track 0.726** (MPPE 3.3 cm; CD-O 0.53 vs 0.24). ep26: 0.349 → 0.500 (= still); phase 2 no change. Real-input ep40 (2026-10-23, one run, lost with the VM): P1b/P1c/P1d PASS; S3a (SAM 3D) FAIL; S3b WARN; S4c FAIL; S5 FAIL; best reference 0.381 kinematic / 0.400 still; S6 60M ≈ 20 min. Decisions carried: canonical mesh is S3a; `tracking` cut; S4c off until it carries; two policies; HaMeR. Owed from v3: front-end comparison (team vs v3 S1) and the first scored Path B policy.

Sources (checked 2026-10-24): `path_b/oracles/oracle_2026-10-17.json`; `path_b/BUILD_LOG.md`; `tools/track3/episodes.json`; FlashCHORD in `nvidia-isaac/video_to_data` (`configs/experiment/sharpa_flash_sac.yaml`, `configs/termination/tracking.yaml`, `runtime/actions/hand_pose.py`, `evaluation/harness.py`, `configs/curriculum/tracking_250m_mixed_reset.yaml`); ROHIT (arXiv 2512.07394, §3.1, suppl. Table 13); EgoFound3R README; challenge FAQ; Kaggle Track 3 boards 2026-10-23 (leader 0.922 / 1.30 cm; team 0.425, rank 13/27); the 2026-10-24 adversarial review of "Path B v4 Design".

## 13. Plan after the first full runs (James, 2026-10-26; supersedes §6's Oct 28–31 rows and Task Queue items 6–13)

**Facts the plan rests on** (run reports 2026-10-25/26, both review pages): 19 of 20 public episodes scored; ours 0.363 vs Edmond's place 0.401 on the same 19, 0.623 vs 0.391 where our mesh is right and 0.154 vs 0.520 where it is wrong. Every loss is a mesh case: table-patch partials pooled (held-out RMS 0.77–1.25 m), revolution axes capped instead of rejected, pot clips pooled 120° apart, anchor meshes registering 146–179° off in the scorer. `wooden_spoon` appears only in evaluation episodes (10 of 20), so pooling must include evaluation clips. S5/S6 have never run in v4; two-object scenes (16 of 20 evaluation episodes) score 0.237 on public vs the team's still 0.301. MP-SR tests per-object *means* (scorer guide), so it is reachable on clean clips (ep40, ep45 already pass in playback).

**Rules.** Scans are never read on evaluation (James: no). Pooling over evaluation clips is allowed (James: yes). Every change ships with a GT-free signal measured on evaluation and calibrated on public: explain-share, registration margin (B1), frame-0 consistency (B3), P3 scale factor, segment agreement with hand motion, settle penetration, grasps failed, and — the primary gate for the submission — the policy's rollout error against its own reference. No rollout splicing: candidates are mixed per object in the *reference*, one policy per episode.

### 13.1 Meshes (CPU, Oct 27)
- **A1 pooling gates** (`pathb/canonical.py`): drop partials failing S1's held-out depth check; axis prior = table normal, reject fits > 20° off (never cap); align revolution objects by their asymmetric feature (pot handle azimuth: direction of the farthest-from-axis points) and reject clips disagreeing > 30°; reject clips whose partial is > 0.3 cm from the pool. Test: explain-share ≥ 0.8 on every clip; public CD-O; dimensions of the 9 scanned objects.
- **A2 neighbour-free scans** (`s3a2_fuse_scan`): drop pixels inside another object's mask or within 2 cm of its points; 2 mm voxels for rack, sweeper, boards.
- **A3 pool over all 40 clips** (`run.py canonical --split all`), after the evaluation S1 download (~30 GB). Keep the 40-clip pool where it explains more than the 20-clip one, per object.
- **A4 three mesh candidates per object**: pooled, this clip's own, the team's SAM 3D mesh (bundle repo; verify it exists for evaluation objects — 69 bundles ≈ 33 public + 36 evaluation). Pick by explain-share, ties by B1 margin. Record the choice in `s3a2_register/report.json`.
- **A5 completion keeps real asymmetric geometry** (closed cup bottom, modelled handle, sweeper handle side). No invented features (CD-O is bidirectional).
- Then rerun S3a2 → S4b + waterfall on all 19 public episodes on **one** mesh set.

### 13.2 Anchor-flip insurance (CPU, Oct 27)
- **B1 registration-ambiguity test**: reuse `score_local.py`'s registration (RMS-radius rescale → 3 000 points → 25 PCA starts → trimmed ICP 90 %) to register our mesh onto our own rest scan; margin = runner-up cost − best cost. Calibrate on public: flagged ⇔ the scorer flipped (13, 31, 0). GT-free.
- **B2** flagged anchor → the A4 candidate with the largest margin, else A5.
- **B3 frame-0 rule and bar**: the scorer aligns the whole episode from frame 0 of object 0 (translation) and the first non-symmetric object (rotation). Bar: ‖pose(frame 0) − static-segment pose after settle‖ < 5 mm / 1° for those objects. Where object 0 is a cup, the second object's frame-0 orientation is the most valuable number in the episode.

### 13.3 Carry solve (CPU, Oct 28)
- **C1** inside a grasp: contact set fixed, pose free per frame, strong temporal prior (ROHIT's relaxation). Not "absorb 3 cm". Keep the slip guard; test inverted ≥ keyframe and MPPE on 41, 18, 43.
- **C2** rotation with the wrist in the grasp fit (oracle: `place_pos_rot_gt − place_pos_gt` = +0.259 mean; ep23 +0.60, ep24 +0.81, ep43 +0.60).
- **C3** thin objects: fingertip term up, depth term down. **C4** turned-over rest pose (180° flip candidate when rest-after differs from rest-before). **C5** two-handed grasps (optional).

### 13.4 Stacked objects and candidates (CPU, Oct 28)
- **D1** supports bottom-up (overlapping footprints: lower on the table, upper on the lower); settle in that order.
- **D2** per-object candidate choice (inverted / keyframe / still) in the reference; one policy per episode. Replaces the ep41/ep13 go/no-go gate (removed).
- **D3** `edmond_place` as a per-episode candidate (his bundle through the same GT-free signals).

### 13.5 Hands and camera (GPU window 1, Oct 27)
- **E1** second-hand seeding when WiLoR first sees it; reject overlap > 0.5 and re-seed; re-run P1b/P1d on 2, 10, 21, 22, 28, 33, 44.
- **E2** camera: measure on the opening 30 frames when no rest plane; keep more table in the depth prior; re-run P1c on 2, 20, 22, 24, 32.
- **E3** bars: wrist speed → WARN < 0.95; hand-scale bar 0.7.

### 13.6 Policy stage
- **F1 S0.1 runs in GPU window 1 (Oct 27)**: policy on the true trajectory for ep41 and ep13 (2 × 20 min). It is also S5/S6's first run in v4. Below still → the S6 recipe is the problem; ≥ 0.7 → the reference is.
- **F1b two-object smoke**: S5/S6 on ep26 end to end in the same window (merge, RPE selection).
- **F1c** dry `run.py pack` on the public split in the same window.
- **F2** two candidates × 60M on all 40; 250M on the chosen reference for the 20 evaluation episodes (trim to evaluation only if time is short).
- **F3 selection on evaluation**: primary gate = rollout error against its own reference (kit termination 10 cm / 0.5 rad); then the GT-free signals; candidates include `edmond_place`. Experiment on public first: selection accuracy vs the GT-best candidate.
- **F4** MP-SR: per-object means; aim it at 40, 45, 26, 18, 41.

### 13.7 Evaluation run and submission
- **G1** CPU chain on the 20 evaluation episodes; collect per object: explain-share, B1 margin, B3 frame-0 consistency, P3 scale factor, segment agreement, settle penetration, grasps failed.
- **G2** fit signals → score on public; apply to evaluation.
- **G3** reference with D2/D3, policies, F3 selection, pack, preflight; **HOLD** for James.
- **HF**: uploads resume (space freed, read token refreshed 2026-10-26); first upload = the Mac-only results (ep38 S1, ep2 rerun, the 19 public CPU chains, canonicals); S4b outputs also copied Mac → VM directly for the policy window.

### 13.8 Schedule
Oct 27: 13.1, 13.2, 13.5 + GPU window 1 (F1, F1b, F1c, E). Oct 28: 13.3, 13.4, G1. Oct 29–31: F2, G2, G3 → safe submission Oct 31 (James's go). Freeze Nov 4 5 pm EST, unlimited submissions Nov 2–4: Nov 1–3 for 250M reruns and a resubmission on Nov 3 if James lifts the Nov 1 stop.

**Dropped:** scans as meshes; "frame 0 is free" (the opposite holds); slip absorption to 3 cm; invented mesh markers; rollout splicing; the go/no-go gate; S4c; full HF re-upload of frame archives (results only).

Sources: review pages "All Episodes: v4 vs Refined Baseline" and "Seven Episodes Deep Dive" (2026-10-26); `track3_scorer_guide.md` (registration recipe, frame-0 alignment, MP-SR on means, scale never corrected); `BUILD_LOG.md` 2026-10-24–26; challenge FAQ (own reconstructed meshes) and page (Nov 4 freeze); FlashCHORD `termination/tracking.yaml`; ROHIT arXiv 2512.07394; `tools/track3/episodes.json` (wooden_spoon evaluation-only).

## 14. Plan from run `v4_i1` (James, 2026-10-29; supersedes §13's schedule and Task Queue items 8–13)

**State.** Inverted 0.598 on the 20 public episodes (playback), MP-SR 0.40, SP-SR 0.75; best of three 0.606; GT-free pick 0.580. S0.1: a policy on the true trajectory scores 0.935 (ep41) / 0.899 (ep13), and F2's first rollout kept 0.842 of a 0.939 reference, so the policy tax is 0.07–0.10 and the reference is the problem. **Every number is stated in two columns from now on: playback and submitted (rollout ≈ playback − 0.07…0.10).** Today: 0.598 / ≈ 0.50. This plan: 0.70–0.74 / 0.60–0.66. The 0.8 playback goal is not reached by it. Freeze Nov 4, 5 pm EST; submissions open Nov 2–4. Public first (James, 2026-10-29). **Evaluation runs stay paused (James, 2026-10-28) until he lifts the pause**: L10, G4 and G5 below are written out so they can start the hour he does; the review's point stands that they are the largest schedule risk (≈ 48 GPU-h after a 3–4 h Mac chain).

**Where the 0.2 is.** Lost (< 0.5): 21 (0.007), 24 (0.029), 23 (0.160), 0 (0.353), 43 (0.435), 11 (0.454) — five different failures. Mid (0.6–0.75): 12, 2, 42, 45, 13, 22, 27, 31 (MPPE 3–5 cm, SP-SR 1, MP-SR 0). Good (≥ 0.8): 18, 26, 39, 40, 41, 44.

**Measured facts behind the tasks.** (1) The organiser scans' own 180° symmetry margins under a trimmed ICP (3 000 points, 90 %): wooden pieces 0.8–1.3 mm, sweeper 0.9, lid 1.8 (forgiven by the metric's half-turn rule), cups up-down 6.8–8.9, pot / dust pan / rack / pitcher 6–12 mm; our board and sweeper CD-O is 3–4 mm, and the scorer's own 4 096-face re-mesh adds ≈ 1 mm, so the board flip is at noise level even with a perfect mesh. (2) ep21's pitcher rests fit 0.3–1.8 m away and its partials failed the held-out depth check at 0.77–1.25 m: the stereo-through-a-transparent-object signature (ClearGrasp), to be confirmed by L2 before L3 is built. (3) `white_pot`'s symmetry group is `{I}`: spin is scored, and the handle hides in the hand, so an in-hand turn can only be partly recovered. (4) ep13's S0.1 overflowed MuJoCo contact buffers (needed njmax ≈ 708 at 512); overflow drops contacts silently. (5) Rank = sum of ranks over AUC, SP-SR, MP-SR, MPPE, RPE, CD-O; MP-SR (0.40 vs the leader's 0.95) is our weakest rank.

### 14.1 Local (Mac), in order
| # | Task | Effort | Check | Expected (playback) |
|---|---|---|---|---|
| L1 | `run.py analysis` on the 8 mid episodes: error at rests vs carries | 0.5 h | per-frame timelines | scopes L5 |
| L2 | ep21 depth check: histogram of masked depth vs table depth on 20 rest frames (pitcher, cup) | 0.5 h | behind-table ⇒ build L3; else find the cause | gate |
| L3 | Silhouette term: render the canonical mesh, distance transform against the SAM mask at full colour resolution. (a) The static fit's residual when the held-out depth check fails, with L4's upright prior so a table rest is (x, y, yaw); (b) in pass 2 everywhere at low weight | 1 day | ep21 up; cups 22/12/13 not down; GT-free: silhouette IoU per rest | ep21 → ≈ 0.5 |
| L4 | Upright prior for revolution objects at rest + concentric stacking (height, spin) when a rest's footprint lies inside another revolution object's; override only when evidence beats the prior by a margin; log overrides | 0.5 day | ep11 up, none down | ep11 → ≈ 0.6 |
| L5 | Refinement scoped by L1: carry-only joint refinement (depth + silhouette + hand non-penetration + smoothness, rests pinned) if the mid loss is in carries; dropped if at rests | 0–1.5 days | mid MPPE; MP-SR | +0.03–0.05 |
| L6 | Hand-mesh non-penetration in pass 2 / L5, only where hand scale ≥ 0.85 (else inflate by 1/scale) | 0.5 day | no regression on 41/44 | part of L5 |
| L7 | Tracked bridge for in-hand turns: pass-2 chain forward and backward with the 5 mm / 3° clip and end-pose terms, no wrist prior; accept only if ep43 rises and ep41/45 do not fall. ep0's lid: hand-mesh proximity to find its grasp | 0.5 day | gate as stated | ep43, ep0 → ≈ 0.5 |
| L8 | Boards / sweeper at 1–2 mm voxels (background); then the scorer's registration 10× on our mesh with different point samples; stop if the flip rate is not ≈ 0 | 0.5 day | flip rate | +0.00–0.02 |
| L9 | Six metrics in every checkpoint compare; tasks weighed by rank gain | 1 h | — | — |
| L10 | **Paused until James lifts it.** Evaluation CPU chain: method frozen at the lift (proposal: after L3–L4); S3b → S5 on the 20 evaluation episodes; GT-free signals per object; later per-episode reference swaps only where the bars pass | 3–4 h Mac | firewall: no GT read | a submission exists |
| L11 | Kit packer: S7 switches to `pack_submission.py`; dry-pack the public split today | 2 h | preflight | — |

### 14.2 GPU (L40S), in order
| # | Task | GPU-h | Check |
|---|---|---|---|
| G1 | F2 continues on public. Before any two-object episode: `sim.njmax` / `nconmax` ≥ 1024; check L40S memory | — | no `nefc overflow` lines |
| G2 | After the first four public rollouts: selector decision (always-inverted vs rollout-vs-reference gate) | 0 | selection accuracy on the four |
| G3 | Swap improved public references per episode as L3–L7 land; retrain only those | ≈ 1 h / episode | rollout vs playback |
| G4 | **Paused until James lifts it.** Evaluation: 60M × 2 candidates × 20 (≈ 20 h), then 250M on the chosen reference (≈ 28 h) | ≈ 48 h | GT-free gate |
| G5 | Kit packer, preflight, **first evaluation submission Nov 1–2** (James's go); resubmit Nov 3 | 0.5 h | preflight |

### 14.3 Schedule
Oct 29: L1, L2, G1, L11, L9, L3 started. Oct 30: L3 done, L4, L5/L6 as scoped. Oct 31: L7, L8. L10 → G4 → G5 need ≈ 3 days end to end once the pause is lifted; lifted on Oct 30 they end Nov 2 with a submission Nov 2 and a resubmission Nov 3; lifted on Nov 1 the 250M stage no longer fits and G4 runs 60M only.

### 14.4 Research additions (2026-10-29, after the §14 plan; verified as stated)
- **L2 answered — ep21 is not a depth problem.** Frame 0 of ep21: stereo depth inside the pitcher mask is coherent (median 0.44 m, p10–p90 0.38–0.50 m; 1.5 k of 43 k pixels bleed to 1–2 m at the mask edge), the cup 0.49 m, the table 0.62 m. The fuse report shows the real causes: the pitcher has **no rest-before window** (`rest before 0 (window None)`; its rest-after partial is fine, 65 frames, RMS 3.4 mm) because the window rule stops at the first hand touch and ep21's hand track is the same-hand failure (overlap 0.98), and the cup anchor's registration is the ambiguous one B2 made worse (−0.056). So ep21 = **E1 (second-hand seeding) + anchor registration**, not L3. L3 (silhouette term) stays as a low-weight pass-2 term and as the static residual where the held-out depth check fails; it is no longer the ep21 task. **R0**: re-seed the hands on ep21/22/44 (public, ≈ 15 GPU-min each), rerun ep21; expected ep21 → ≈ 0.5.
- **R1 — FoundationPose refinement from the inverted trajectory (GPU, no new licence: it ships in the baseline).** Per frame, run the FoundationPose refiner (`track_one` iterations) starting from the inverted pose with the canonical mesh, RGB-D, and the mask; keep the refined pose only where its depth residual and silhouette IoU beat the inverted pose's. This is the standard render-and-compare polish that takes a roughly-right track to centimetre accuracy. Target: the mid group (MPPE 3–5 cm, MP-SR 0 on 2, 12, 13, 22, 27, 45): −1 cm MPPE flips MP-SR episodes (per-object mean < 3 cm), each worth 5 % of MP-SR. Check on public per episode against the checkpoint, both columns. Effort: 0.5 day (driver exists: `drivers/fp_track.py`), ≈ 3 GPU-min per episode.
- **R2 — GoTrack (Meta, CV4MR 2025) as the alternative refiner**: RGB-only flow-based render-and-compare with a CAD model, ADD-S AUC ≈ 90 on HOT3D (Aria, egocentric hand-object); refiner 40 ms per iteration; needs a textured mesh and a GPU. **Licence CC BY-NC 4.0** (code and checkpoint): NC only, no SA; James decides. Only if R1 underdelivers; needs R3 first.
- **R3 — textured canonical mesh**: integrate colour in the TSDF (Open3D does it) so refiners that render RGB (GoTrack; FoundationPose uses RGB too) see the real appearance. Half a day; also improves CD-O not at all, so low priority unless R2 is taken.
- **Not taken**: learned transparent-object pose methods (EBFA-6D, GDR-Net + edges; need training data we do not have) — and the pitcher is opaque in any case.

### 14.5 Refinement round from the 2026-10-30 research (James; highest priority; push rule below)
**Push rule (James, 2026-10-30).** This is a refinement attempt: work on a local branch `zjifan121_pathb_v4ref`; after each proposal run all 20 public episodes against the latest checkpoint on all six metrics; **push to `zjifan121_pathb_v3` only when the overall public score is higher** (inverted mean AUC up, no episode ≥ 0.7 down, rank-sum proxy not worse) — otherwise the commit stays local and BUILD_LOG records "tried, not adopted". Every number in two columns (playback / submitted).

Research verdicts (report 2026-10-30): **take** P1 board half-turn resolver, P2 distinct hand tracks, P3 lid stable-pose snap, P4 ROHIT-style 1-DoF stable-grasp carry (MIT code, idea reimplemented); **test** P5 FoundationPose refiner only with James's licence yes (`nvidia/foundationpose` ONNX under the NVIDIA Open Model License + the Apache-2.0 `foundation-pose-inference-library`, nvdiffrast plugin off; the NVlabs GitHub weights are not acceptable); **drop** GoTrack (CC BY-NC 4.0), 4D-HOF (no code), ComPose code (idea only).

| # | Task | Where | GPU-h | Check (public GT) | GT-free signal | Expected (playback) |
|---|---|---|---|---|---|---|
| G-A | **P2 hands**: two prompts must resolve to distinct tracks (mask IoU < 0.1, opposite handedness) else re-seed the second at its first-appearance frame (existing late-hand path); rerun P1b/P1d on ep21, 22, 44, then phases → S4b on those episodes | L40S between F2 episodes + Mac | 0.75 | ep21 up (pitcher gets a rest-before window ≥ 10 frames); 22/44 not down | hand-track overlap ≈ 0; rest-before window length | ep21 → ≈ 0.5 (+0.025) |
| G-B | **P5 FoundationPose refiner from the inverted trajectory** — only after James's licence yes: ONNX weights + Apache runner, nvdiffrast off; per frame, refine from the inverted pose with the canonical mesh, RGB-D, mask; keep only where score-net output and render-vs-mask IoU improve | L40S | 2 (20 public) | mid group MPPE −1 cm; MP-SR flips on 2 12 13 22 27 45; ep31 rack / ep12 cup 2–4 cm → < 2 | score-net output; render-vs-SAM IoU; closure error (G-D) | +0.03–0.05; MP-SR 0.40 → 0.55 |
| G-C | **P1 board half-turn resolver**: textured canonical mesh (colour TSDF, from all clips); for each rest, score θ vs θ+180° by (a) colour/edge NCC of the rendered textured mesh against the image inside the SAM mask and (b) board-yaw change across the carry vs HaMeR wrist-yaw change; flip only when both agree; eligible only for objects with half-turn margin < 2 mm (boards, sweeper, lid) | Mac (CPU) + L40S render 0.5 | 0.5 | ep24, ep23 up; no ≥ 0.7 episode down | NCC margin > 0 and yaw residual < 30° | +2.5 to +5.3 pts (0.13–0.27 on the mean) if both resolve |
| G-D | **P4 stable-grasp carry (ROHIT)**: inside a carry, object = wrist ∘ fixed offset (fitted on the first confident frames) ∘ R(φ) about the revolution axis, φ the only free parameter fitted to masks and depth, residual term for in-hand drift; grasp from object motion (> 2 cm) + nearest wrist within 10 cm when fingertips are hidden (ep22 pour); closure check = propagated pose at release vs the registered rest-after pose; fall back to the current result when closure is worse | Mac (CPU); L40S only if the renderer is used | 0–2 | 43 42 45 27 up; 22, 0, 2 partly; 41/44/18 not down | closure error (cm/°); stable contact area; mask IoU | +1.5 to +3.6 pts (0.08–0.18) |
| G-E | **P3 lid stable-pose snap**: precompute the lid's stable poses on the table and on the pot rim (drop simulation); at each rest replace the rotation by the nearest stable pose (keep yaw), translation unchanged; apply only when the snap is < 50° and mask IoU does not drop | Mac (CPU) | 0 | ep0, ep2 up (MP-SR flip likely) | tilt vs support normal < 5°; gap/penetration < 5 mm; mask IoU | ≤ +0.75 pt; MP-SR +2 episodes |
| G-F | After G-A…G-E land: retrain F2 candidates only for the changed episodes; re-report rollout vs playback | L40S | ≈ 1 per episode | rollout keeps ≥ 0.9 of the playback gain | rollout-vs-reference error | — |
| G-G | **Paused until James lifts the evaluation pause**: the same-hand sweep (G-A) on the 20 evaluation episodes | L40S | 1 | — | hand-track overlap | — |

| G-0 | **GPU access**: the L40S has been unreachable since 2026-10-30 (host key changed; never replaced without James). James confirms the key or provides a new instance; `new_instance.sh`; restore frames; recover or restart the F2 / `g3d` tmux jobs | James + L40S | 2 | ssh works | — | — |
| G-H | **Rollout loss** (outside the research scope, now the largest gap to the submitted score): F2 rollouts vs playback — ep18 0.885 → still 0.387, ep41 0.831 → 0.689, ep39 0.816 → 0.594, ep27 0.746 → 0.526, ep22 0.719 → 0.315, ep42 0.620 → 0.219, ep45 0.624 → 0.258, ep12 0.718 → 0.449; the 0.07–0.10 tax held only on 40, 26, 44. Diagnose 18, 42, 45 first (termination cause fractions and the object-error timeline from the evaluator JSON; S5 seating/support of the inverted vs still scene; hand reference in the carry: wrist speed, finger closure). Then two controlled tests on ep18: 250M steps; a reference with smoothed hands | L40S + Mac | 3–4 | ep18 rollout ≥ 0.7 | rollout-vs-reference error; termination fractions | submitted +0.05–0.10 if the cause is shared |

**State at the time of writing** (checkpoint `public_2026-10-29_k7`, coding session 2026-10-30): inverted **0.690** playback (L2's floor-fit fix 0.598 → 0.622; L4 upright prior 0.644; concentric stacking 0.657; HELD and TABLE_AT_REST rules 0.690), ≈ 0.59–0.62 submitted. Already moved since the research's issue table: ep11 0.883, ep0 0.654, ep23 0.494, ep13 0.780, ep44 0.891, ep21 0.387. L7 (tracked bridge) and L8 (2 mm board meshes) stopped on their gates; L3 not built (L2's gate). Re-baseline every expectation above against k7 before starting.

Order: G-0 → G-A → G-E → G-C → G-D → G-H → G-B (when licensed) → G-F. The policy retrain (G-F) waits until G-A…G-E land, since they change the trajectories the policy trains on.

**Dropped or demoted:** +0.05 for board meshes (≤ +0.02); 0.65 for ep43 (≈ 0.5); S3b threshold tuning; thin-object weights; a new tracker.

Sources: run `v4_i1` checkpoint; "Path B v4 Design" (rows 1–16, S0.1, F2); `BUILD_LOG.md` 2026-10-29; the symmetry-margin measurement (organiser scans, 2026-10-29); `track3_scorer_guide.md`; challenge page and FAQ; ClearGrasp (ICRA 2020); ROHIT (arXiv 2512.07394); the adversarial review of 2026-10-29.
