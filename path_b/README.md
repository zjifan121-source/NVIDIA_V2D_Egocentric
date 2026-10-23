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
| `p1b_masks` | S1 | SAM 3.1 text-prompted masks for each roster object and both hands (SAM2 + Grounding DINO fallback) | `obj_0/`, `hand_L/`, `hand_R/`, `prompts.json` | task 6 |
| `p1c_camera` | S1 | DROID-SLAM with the stereo depth (hands and objects masked out) as its depth prior: a metric camera path | `camera_to_world/`, `trajectory.txt` | task 7 |
| `p1d_hands` | S1 | WiLoR → HaMeR → depth alignment on the stereo depth, late hands on | `hamer_aligned_filled/`, `hand_scale.json` | task 8 |
| `p3_scale_bridge` | S2 | v2 (today): the team bundle's scale from the stereo support plane. v3: the same ruler against the P1c camera, as a check (expected 1.00) | `scale.json` | v3 in task 9 |
| `p4_export` | S2 | v2 (today): the team bundle rescaled. v3: the bundle of P1c camera and P1d hands in the world (table z = 0) | `episode_XXXXXX/<object>/...` | v3 in task 9 |
| `s3a_mesh` | S3 | SAM 3D Objects per view on the stereo pointmap, MV-SAM3D fusion, metric size fit, CoACD | `mesh.glb`, `coacd/`, `mass.json` | task 10 |
| `s3b_track` | S3 | FoundationPose with the symmetry-aware recovery rule | `poses/`, `track_meta.json` | task 11 |
| `s3c_fuse` | S3 | ESKF/RTS on world poses with rest anchors | `object_to_world.npz`, `anchors.json` | task 12 |
| `s4a_phases` | S4 | Rest-before, carry and rest-after windows from wrist-to-object distance | `phases.json` | task 2 |
| `s4b_reference` | S4 | Kinematic references: `keyframe`, `tracking`, `still` | `<candidate>/` bundles | task 4 |
| `s4c_retarget` | S4 | Do as I Do retargeting of the Sharpa hands in MuJoCo Warp; gate realised vs kinematic | `<candidate>/trajectory_mjwp.npz`, `bundle/` | task 13 |
| `s5_task` | S5 | FlashCHORD ManoSharpa parquet from S4c (mode B); `run_load_retarget.py` fallback (mode A) | task parquet, URDF, support USD | task 14 |
| `s6_train`, `s6_eval`, `s6_select` | S6 | FlashSAC per candidate, kit evaluation, selection | checkpoints, rollouts, `selection.json` | task 15 |
| `s7_pack` | S7 | Preflight and the kit packer, over all evaluation episodes (`run.py pack`) | `track3_submission.parquet` | task 16 |
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
