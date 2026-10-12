# path_a/ — Path A: build on the baseline

Path A's new code lives here.

- **New code** (entrypoints, configs, scripts, analysis) goes in `path_a/`.
- **Fixes to NVIDIA's files** are made in place under `baseline/`. Every edited NVIDIA file gets a header comment, as Apache-2.0 requires:
  `# Modified for V2D Track 3 by <name>, <YYYY-MM>: <what changed>`
- **New Docker modules** that must run with `--dev` go in `baseline/reconstruction/modules/v2d_t3_<name>/`, because `--dev` mounts only that folder into the containers.
- Inputs, docs and the official evaluator come from `assets/` on `main`.

List every NVIDIA code file Path A changed:

```bash
git diff --stat baseline-0.2.0 origin/main:baseline -- . ':!docs'
```
