# tools/ — shared team tooling (Path A and Path B)

Its own pinned Python environment (Python 3.11, `uv.lock`), separate from NVIDIA's code:

```bash
uv sync --project tools                                   # once
uv run --project tools python tools/track3/<script>.py    # from the repo root
```

| Script | What it does |
|---|---|
| `track3/build_manifest.py` | Writes `track3/episodes.json`: every public and evaluation episode with its frame count, video path, and objects in scorer order, each with a detection prompt. Re-run after editing `PROMPTS`. |
| `track3/score_local.py` | Scores predictions on the 20 public episodes with the kit's official metric code (README §5.11). `--self-test` checks it on known cases. |
| `track3/suggest_reference_frames.py` | Writes `track3/reference_frames.json`: a suggested `--reference_frame` per episode (recordings open with both hands raised; MediaPipe finds the best early frame). Runs in its own throwaway env, see its docstring. |
| `track3/run_baseline_reconstruction.py` | Runs NVIDIA's reconstruction on one episode, once per object, with a `run.json` record per run. Reference frame defaults to `reference_frames.json`. Uses the reconstruction host environment, not this one. |
| [`track3/r78/`](track3/r78/README.md) | Opt-in structure-gated SAM3D candidates, shared uniform object size with per-frame SE(3), heldout selection, and provided-mesh FoundationPose tracking. CPU fit/selection; generation/tracking use existing GPU runtimes. |
| [`track3/xkf/`](track3/xkf/README.md) | Opt-in multi-key-frame mesh pick: SAM 3D meshes from 6 frames x 2 seeds, each scored by FoundationPose fit on 10 held-out frames it was not made from; the best is tracked (+0.135 per episode on 11 public episodes). Plan/pick on CPU (this env); draws in the reconstruction host env, scoring in the FoundationPose runtime. |

The scripts read the dataset from `assets/data/track_3/` and the kit from `assets/data/v2d_submission_kit/`.
