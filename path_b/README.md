# path_b/ — Path B: a completely new approach

Path B turns a Track 3 episode into kit-ready reconstruction bundles and records, for every stage,
how well it did, so a bad score points at the stage that caused it. The design and plan are in the
team's Claude doc "Path B: End-to-End VLM Framework for Track 3"
(https://claude.ai/code/artifact/0873d10c-acc3-455c-936a-12baf370fabf).

**Path B v3** (2026-10-21): everything is rerun from the three head-camera videos, with no team
bundle reused. The build spec, `V3_BUILD_SPEC.md`, is the source of truth for stages, parameters,
acceptance bars and the task order; `BUILD_LOG.md` records each task's local, VM and PR steps.
`cache/` keeps earlier documents and parked code: v1's framework export and Stage 1 overview, and in
`cache/v2/` the v2 design (`PATH_B_V2_PHYSICS_INVERSION.md`) with its SPIDER scene and merge runner,
which v3 does not run. `oracles/` holds the S0 oracle results (tracked; `data/` is not).

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
| `p1b_masks` | S1 | SAM 3.1 text-prompted masks for each roster object and both hands (SAM2 + Grounding DINO fallback) | `obj_0/`, `hand_L/`, `hand_R/`, `prompts.json` | built (task 6); GPU smoke test pending |
| `p1c_camera` | S1 | DROID-SLAM with the stereo depth (hands and objects masked out) as its depth prior: a metric camera path | `camera_to_world/`, `trajectory.txt` | built (task 7); GPU smoke test pending |
| `p1d_hands` | S1 | WiLoR → HaMeR → depth alignment on the stereo depth, late hands on | `hamer_aligned_filled/`, `hand_scale.json` | built (task 8); GPU smoke test pending |
| `p3_scale_bridge` | S2 | Stereo camera height over P1c's on the rest window: the scale check (expected 1.00), the table plane and gravity | `scale.json` | v3 built (task 9); ran on the Mac |
| `p4_export` | S2 | The v3 world (table top z = 0, +z up) and per object the bundle of P1c's camera and P1d's hands, no object yet | `episode_XXXXXX/<object>/...` | v3 built (task 9); ran on the Mac |
| `s3a_mesh` | S3 | SAM 3D Objects per view on the stereo pointmap, MV-SAM3D fusion, metric size fit, CoACD | `mesh.glb`, `coacd/`, `mass.json` | built (task 10); GPU smoke test pending |
| `s3b_track` | S3 | FoundationPose with the symmetry-aware recovery rule | `poses/`, `track_meta.json` | built (task 11); GPU smoke test pending |
| `s3c_fuse` | S3 | ESKF/RTS on world poses with rest anchors | `object_to_world.npz`, `anchors.json` | built (task 12); ran on the Mac |
| `s4a_phases` | S4 | Rest-before, carry and rest-after windows from wrist-to-object distance | `phases.json` | built (task 2); ran on the Mac |
| `s4b_reference` | S4 | Kinematic references: `keyframe`, `tracking`, `still` | `<candidate>/` bundles | built (task 4); ran on the Mac |
| `s4c_retarget` | S4 | Do as I Do retargeting of the Sharpa hands in MuJoCo Warp; gate realised vs kinematic | `<candidate>/trajectory_mjwp.npz`, `bundle/` | built (task 13); GPU smoke test pending |
| `s5_task` | S5 | FlashCHORD ManoSharpa parquet from S4c (mode B); `run_load_retarget.py` fallback (mode A) | task parquet, URDF, support USD | built (task 14); ran on the Mac (CPU venv, mode A) |
| `s6_train`, `s6_eval`, `s6_select` | S6 | FlashSAC per candidate, kit evaluation, selection | checkpoints, rollouts, `selection.json` | built (task 15); GPU smoke test pending |
| `s7_pack` | S7 | Preflight and the kit packer, over all evaluation episodes (`run.py pack`) | `track3_submission.parquet` | task 16 (phase B) |
| `s3a2_fuse_scan`, `s3a2_register`, `s3b2_track_points` | §11 A2 | Phase A2: partial scans from stereo depth at rest, canonical mesh per object (`run.py canonical`), registration and the re-expressed track, mesh-free point tracks (TAPIP3D); comparison and decision by `run.py a2` | `partial_scan.ply`, `<run>/canonical/`, `registration.json`, `poses/` | built; run only when named; A2.4 GPU smoke pending |
| phase 2 (`run.py p2`) | §8 | Two-object scope with `PATHB_PHASE2=1`: S4a support relation, S4b anchored on the roster-first object, S4c two-object Do as I Do scene, S5 two bodies (`--seat`), S6 RPE | `<run>_p2/` | built; ran on the Mac (ep26) |
| `s23_load_retarget` | S5/S6 | Stages 2–3 with the team's `run_load_retarget.py` (default repairs), then the reconstruction-only score on public episodes. One object only; v3 merges two-object scenes in S5 mode A | `run/` (the runner's own output) | built (v2) |

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
- **Score waterfall** (public episodes, official metric code): `raw` bundle → `stage2` repairs →
  `stage2_stereo` (Path B's scale) → `stage2_scan` (oracle scale from the public scans). The AUC
  differences say how much stage 2's repairs, Path B's scale, the scale still missing, and
  everything else (tracking, mesh, placement) cost. The scans are used for scoring only.

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
