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

The scripts read the dataset from `assets/data/track_3/` and the kit from `assets/data/v2d_submission_kit/`.
