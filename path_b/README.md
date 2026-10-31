# path_b/ — Path B: a completely new approach

Path B turns a Track 3 episode into kit-ready reconstruction bundles and records, for every stage,
how well it did, so a bad score points at the stage that caused it. The design and plan are in the
team's Claude doc "Path B: End-to-End VLM Framework for Track 3"
(https://claude.ai/code/artifact/0873d10c-acc3-455c-936a-12baf370fabf).

**Path B v4** (2026-10-24): contact-segmented inversion. Each clip is split into static,
stable-grasp and unstable segments, and each is solved with the rule that holds there (one pose per
static segment; over a grasp, the hand track carries the object). The build spec, `V4_BUILD_SPEC.md`,
is the source of truth for stages, parameters, acceptance bars and the task order; `BUILD_LOG.md`
records every task and decision (v3's rows first, then the v4 section). Dev episodes: ep41 and ep13.
Earlier documents and parked code (v1's framework export and Stage 1 overview; v2's design, SPIDER
scene and merge runner; v3's spec, audit and v3-only stages), v2's `run.py visualize` page, the v3 dev
analysis, the 2026-10-26 review pages and the checkpoints before `public_2026-10-30_l5i` were removed on
2026-10-30; they are in git at `bb6bc46f` (`git show bb6bc46f:path_b/<path>`). `oracles/` holds the two
baselines the waterfall reads; `reports/checkpoints/` the public checkpoints from `public_2026-10-30_gd1` on
(tracked; `data/` is not; `public_2026-10-30_l5i` was removed on 2026-10-31, in git at `2ae4796a`, and
`public_2026-10-31_ga2` the same day, in git at `2894b075`).

Rules: all Path B code lives here. It uses `assets/` (inputs, docs, the official evaluator), calls
`baseline/` modules only read-only (agreed 2026-10-18), and reuses the shared `tools/track3/` code
in-process or as subprocesses. It writes only under `path_b/data/` (gitignored).

## Stages

`pathb/runner.py: STAGES` in execution order, one module per stage in `pathb/stages/`. A stage
whose module is not built yet reports SKIP and writes no report.

| Stage | Spec | Does | Main outputs | State |
|---|---|---|---|---|
| `p0_ingest` | S1 | Decodes `ego_cam_a/b/c`; frame counts and resolutions must match the manifest and calibration | `report.json` | built |
| `p1_calibrate` | S1 | Rectifies the head stereo pair (left `ego_cam_c`, right `ego_cam_b`) and undistorts `ego_cam_a` with the 14-coefficient factory model. The calibration does not say whether its intrinsics are native or upright, so all four readings are tried on the episode's frames and the best kept | `rect_left/`, `rect_right/`, `cam_a/`, `calib.json` | built |
| `p2_depth` | S1 | FoundationStereo (the baseline's `v2d_foundation_stereo`, read-only) on the rectified pairs; the table plane in every frame; depth reprojected into `ego_cam_a`, with its extrinsics reading chosen by feature reprojection | `depth_rect/`, `planes.json`, `alignment.json`, `depth_a_half/` | built |
| `p1b_masks` | S1 | SAM 3.1 text-prompted masks for each roster object and both hands (SAM2 + Grounding DINO fallback, flagged); two hand tracks that are not two distinct hands are re-seeded from WiLoR's boxes of their side (`HAND_RESEED`, §14.5 G-A′). On an episode whose S1 exists, the fix reruns P1b, P1d, P3 and P4 and keeps P1c's camera (DROID-SLAM is not repeatable) | `obj_0/`, `hand_L/`, `hand_R/`, `prompts.json` | built; ran on ep40 (L40S, 2026-10-23) |
| `p1c_camera` | S1 | DROID-SLAM with the stereo depth (hands and objects masked out) as its depth prior: a metric camera path | `camera_to_world/`, `trajectory.txt` | built; ran on ep40 |
| `p1d_hands` | S1 | WiLoR → HaMeR → depth alignment on the stereo depth, late hands on | `hamer_aligned_filled/`, `hand_scale.json` | built; ran on ep40. V4 adds fingertip / vertex / wrist exports and the hand-gap rule (task 2) |
| `p3_scale_bridge` | S2 | Stereo camera height over P1c's on the rest window: the scale check (expected 1.00), the table plane and gravity | `scale.json` | built |
| `p4_export` | S2 | The world (table top z = 0, +z up) and per object the bundle of P1c's camera and P1d's hands, no object yet | `episode_XXXXXX/<object>/...` | built |
| `s3a2_fuse_scan`, `s3a2_register` | S3a | Partial scans from stereo depth at rest; the canonical mesh per object, pooled over every clip (`run.py canonical`); frame-0 registration | `partial_scan.ply`, `<run>/canonical/`, `registration.json` | built (v3 A2.1–A2.3) |
| `s3b_segments` | S3b | Contact segmentation: static, grasp, unstable; thresholds from the data; support per static segment | `segments.json`, `phases.json` | V4 task 5 |
| `s3b_track` | S3b_track | FoundationPose, seed orientation only, when `s3a2_register` is WARN/FAIL | `poses/`, `track_meta.json` | built (v3 S3b; demoted) |
| `s3c_invert`, `s3d_settle` | S3c/S3d | Segment solving (static ICP; grasp: hand→object offset, then per-frame refinement; unstable bridges: interpolated, carried by S3b's hand, or, where the object is seen moving, by the nearest wrist, whichever explains the frames best); MuJoCo settle of static poses | `object_to_world.npz`, `invert_report.json` | V4 tasks 6–7 |
| `s4a_phases` | S4 | Rest-before, carry and rest-after (V4: a view of S3b's segments; the wrist rule is the fallback) | `phases.json` | built |
| `s4b_reference` | S4 | Kinematic references: `inverted`, `keyframe`, `still` | `<candidate>/` bundles | built (keyframe, still); `inverted` with task 8 |
| `s4c_retarget` | S4 | Do as I Do retargeting of the Sharpa hands in MuJoCo Warp; gate realised vs kinematic. Off by default (`s4c.yaml enabled: false`), one candidate | `<candidate>/trajectory_mjwp.npz`, `bundle/` | built; FAILed on ep40 |
| `s5_task` | S5 | FlashCHORD ManoSharpa parquet; `run_load_retarget.py` (mode A) by default, from S4c in mode B | task parquet, URDF, support USD | built |
| `s6_train`, `s6_eval`, `s6_select` | S6 | FlashSAC per candidate, kit evaluation, selection | checkpoints, rollouts, `selection.json` | built; 60M steps ≈ 20 min on the L40S |
| `s7_pack` | S7 | Preflight and the kit packer, over all evaluation episodes (`run.py pack`) | `track3_submission.parquet` | built (v3 task 16) |
| `s23_load_retarget` | S5/S6 | Stages 2–3 with the team's `run_load_retarget.py` (default repairs), then the reconstruction-only score on public episodes | `run/` (the runner's own output) | built (v2) |

Depth PNGs use the baseline's inverse-depth encoding (`pixel = 65535 / (depth_m + 1)`); 0 means no depth.

## Measuring loss

Every measurement is stored with its unit, its bar and a verdict (`PASS`, `WARN`, `FAIL`, or `INFO`
without a bar). Its `kind` says whether it needs ground truth: `health` checks work on all 40
episodes; `gt` ones use public ground truth or scans. A stage's status is its worst verdict;
`ERROR` (an exception) stops the episode.

- **Per stage:** `<stage>/report.json` (measurements, runtime, peak GPU memory, commit).
- **Per episode:** `loss_report.md` and `.json` name the first failing stage and list every
  measurement outside its bar. `run.json` records the command, commit, GPU and stereo engine.
- **Per bundle:** `numeric_qa.json` and `status.json` travel with the exported bundle.
- **Score waterfall** (public episodes, official metric code, `pathb/waterfall.py`): `raw`,
  `ref_inverted`, `ref_keyframe`, `ref_still`, `realised_*`, `rollout_*`, each also `@scan` (the
  public scan as mesh: a ceiling, not submittable), next to the 0fa525bc and S0.2 baselines. The
  scans are used for scoring only.

## Run (Linux GPU machine, from the repo root)

Needs: the dataset in `assets/data/track_3/`, the team's bundles in
`baseline/reconstruction/data/outputs/hub/` (`tools/track3/bundles.py download`), the baseline
host venv `baseline/reconstruction/.venv` with `-e modules/v2d_foundation_stereo/docker`, the
`v2d_foundation_stereo` image and its engine (`run_download_weights.py`, `run_export_engine.py`),
and for stages 2–3 the environment of README §5.12 in `RETARGET_VENV`. `vm/setup.sh` sets all of
this up on a new instance, and `vm/hf_outputs.py` saves and restores outputs on Hugging Face
(`vm/README.md`).

```bash
uv sync --project path_b
RETARGET_VENV=/path/to/retarget uv run --project path_b python path_b/run.py run --episodes 41
uv run --project path_b python path_b/run.py waterfall --episodes 41
uv run --project path_b pytest path_b/tests          # unit tests, CPU only
```

Outputs go to `path_b/data/outputs/<commit>/episode_XXXXXX/`, so changed code never reuses stale
results; `--out` picks another folder, `--force` reruns stages, `--stages` runs a subset.
