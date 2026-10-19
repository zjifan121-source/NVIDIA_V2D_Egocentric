# path_b/ — Path B: a completely new approach

Path B turns a Track 3 episode into kit-ready reconstruction bundles and records, for every stage,
how well it did, so a bad score points at the stage that caused it. The design and plan are in the
team's Claude doc "Path B: End-to-End VLM Framework for Track 3"
(https://claude.ai/code/artifact/0873d10c-acc3-455c-936a-12baf370fabf).

**Stage 1 (now):** factory calibration and metric stereo depth, used to fix the scale of the team's
existing reconstruction bundles (the bridge). Later stages replace the bridged parts: object list
and masks, symmetry-aware tracking (Stage 2), mesh and world frame (Stage 3).

Rules: all Path B code lives here. It uses `assets/` (inputs, docs, the official evaluator), calls
`baseline/` modules only read-only (agreed 2026-10-18), and reuses the shared `tools/track3/` code
in-process. It writes only under `path_b/data/` (gitignored).

## Stages

| Stage | Does | Main outputs |
|---|---|---|
| `p0_ingest` | Decodes `ego_cam_a/b/c`; frame counts and resolutions must match the manifest and calibration | `report.json` |
| `p1_calibrate` | Rectifies the head stereo pair (left `ego_cam_c`, right `ego_cam_b`) and undistorts `ego_cam_a` with the 14-coefficient factory model. The calibration does not say whether its intrinsics are native or upright, so all four readings are tried on the episode's frames and the best kept | `rect_left/`, `rect_right/`, `cam_a/`, `calib.json` |
| `p2_depth` | FoundationStereo (the baseline's `v2d_foundation_stereo`, read-only) on the rectified pairs; the table plane in every frame; depth reprojected into `ego_cam_a`, with its extrinsics reading chosen by feature reprojection | `depth_rect/`, `planes.json`, `alignment.json`, `depth_a_half/` |
| `p3_scale_bridge` | One metric scale factor per episode: the camera's height above the table from stereo (the largest horizontal plane at most 35 cm under the object), divided by the same height in the bundle (its lowest mesh point), over the frames before a hand reaches an object | `scale.json` |
| `p4_export` | Each object's bundle rescaled by that factor, in the `tools/track3/BUNDLES.md` layout, with `run.json`, `numeric_qa.json` and `status.json` beside it | `episode_XXXXXX/<object>/...` |
| `s23_load_retarget` | Stages 2–3 with the team's `run_load_retarget.py` (default repairs), then the reconstruction-only score on public episodes. Two-object episodes are skipped for now (the runner merges only from the shared hub) | `run/` (the runner's own output) |

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
and for stages 2–3 the environment of README §5.12 in `RETARGET_VENV`.

```bash
uv sync --project path_b
RETARGET_VENV=/path/to/retarget uv run --project path_b python path_b/run.py run --episodes 41
uv run --project path_b python path_b/run.py waterfall --episodes 41
uv run --project path_b pytest path_b/tests          # unit tests, CPU only
```

Outputs go to `path_b/data/outputs/<commit>/episode_XXXXXX/`, so changed code never reuses stale
results; `--out` picks another folder, `--force` reruns stages, `--stages` runs a subset.
