# path_b/ — Path B: a new pipeline

Path B builds a different method on the same inputs. Work happens on `path-b/main` and `b/<topic>` branches only. On `main` this folder holds just this README.

## Rules

- All Path B code lives in `path_b/`: its own entrypoint, stages, configs and Docker images.
- Path B may **call** NVIDIA code in `baseline/` (e.g. `baseline/robotic_grounding/`) but **never edits** it. This keeps Path B conflict-free and easy to extract later.
- Inputs come from `assets/data/track_3/` (see `assets/README.md`).
- Write outputs to `baseline/reconstruction/data/outputs/<branch>/<episode>/` or a `path_b/`-local gitignored folder, never into another branch's output folder.
