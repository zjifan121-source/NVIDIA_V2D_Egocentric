# path_a/ — Path A: improve the baseline

Path A fixes and extends NVIDIA's reconstruction pipeline. Work happens on `path-a/main` and `a/<topic>` branches only. On `main` this folder holds just this README.

## What goes where

| Change | Location |
|---|---|
| Path A's own code: entrypoints/wrappers, configs, run and batch scripts, analysis notebooks | `path_a/` |
| Fixes to NVIDIA files | In place under `baseline/`, with the modification header (below) |
| New Docker module that must run with `--dev` | `baseline/reconstruction/modules/v2d_t3_<name>/` |

Why new modules can't live in `path_a/`: `--dev` mounts only `baseline/reconstruction/modules/` into the containers (`v2d_docker/container.py`), so module code outside that folder is invisible inside them. Host-side Python (anything that launches containers rather than running in one) works fine from `path_a/`.

## Rules

- Every edited NVIDIA file gets a header comment, as Apache-2.0 requires:
  `# Modified for V2D Track 3 by <name>, <YYYY-MM>: <what changed>`
- One concern per `a/<topic>` branch, small diffs.
- New modules follow NVIDIA's module contract: own Docker image, `run_download_weights` entry point if it needs weights, `run_shell` entry point, types from `v2d_common`.
- Compare against the unmodified baseline on the same episodes.
- List every NVIDIA file Path A changed (pure moves filtered out):
  `git diff -M --name-status baseline-0.2.0 -- baseline reconstruction robotic_grounding video_ingestion_agent | grep -v '^R100'`
