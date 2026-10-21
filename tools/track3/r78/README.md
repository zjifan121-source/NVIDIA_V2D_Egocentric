# R7/R8: structure-gated meshes and shared metric size

Opt-in implementation of the ep27 experiment. Existing reconstruction defaults are unchanged.
This tool consumes frozen RGB, pinhole K, metric stereo depth, validity and object masks.
It does not compute depth, segment objects, read GT/scans, refine hands, or train policies.

- **R7:** preselect at most two clear views from RGB/masks; generate each once with SAM3D seed 0; review visible critical structure and neighboring-object contamination.
- **R8:** fit one uniform size multiplier across **3 support frames**, with independent object-to-camera SE(3) per frame. On **2 heldout frames**, keep size fixed and fit only SE(3). Report consistency on untouched spatial pixel blocks, silhouette IoU and valid-mask coverage.
- **Select and track:** structural PASS first; compare both heldout views by Pareto dominance over IoU, depth residual and coverage. If neither dominates, use the predeclared candidate order. Register the selected metric mesh afresh, then use normal FoundationPose tracking without another size search.

Structure review is manual and input-driven; this is not an automatic semantic mesh judge.
A low depth residual alone is not a selection criterion or evidence of accurate pose.

## Runtime

Run from the repository root, with Python 3.11. CPU stages require NumPy, SciPy,
Pillow and trimesh (already declared in `tools/pyproject.toml`). For example:

```bash
uv sync --project tools
uv run --project tools python -m tools.track3.r78 --help
```

`generate` runs in the existing **SAM3D GPU runtime**, with `v2d.sam3d` and weights available.
`seed` and `track` run in the existing **FoundationPose GPU runtime**, with prebuilt
TensorRT engines. Mount this repository and the input/output paths into those runtimes.
The commands below use `python` for the appropriate environment. No command downloads
weights, builds TensorRT engines, schedules cloud jobs or starts training implicitly.

## Inputs and quick start

Copy `examples/generation.json`, `examples/seed.json` and `examples/pool.json` into
your run directory. Paths in a config are relative to that config, or explicit paths
visible to the current runtime. Replace source revisions and paths with actual values.
The generation example deliberately has review flags **false**; set them only after
checking visibility, identity, mask extent and contamination. Freeze the candidate
views, fit/heldout split and pool order **before generation and GT evaluation**.

```bash
# CPU: snapshot config file references into an immutable SHA-bound manifest.
python -m tools.track3.r78 freeze --config run/generation.json --output run/generation.frozen.json
python -m tools.track3.r78 prepare --manifest run/generation.frozen.json --candidate 1 --output run/view1

# GPU, SAM3D runtime: one inference, fixed seed=0, supplied stereo pointmap.
python -m tools.track3.r78 generate --prepared run/view1 --weights weights/sam3d --output run/candidate1
```

Repeat `prepare`/`generate` with candidate 2 only if a second view was predeclared.
Keep the current mesh unchanged as the baseline. Generation preserves the raw model
and transform, and exports `mesh.obj` with the predicted rotation and uniform size
baked into vertices (translation is not baked). This matches the experiment's seed geometry.

Copy `candidate1/structure_review.template.json` to `run/review1.json`. Record the
visible critical features and contamination findings; set `state: PASS` only if there
are **zero critical missing/extraneous structures**. Keep the supplied baked-mesh SHA.
Do not mark invisible geometry as observed. Reference this review in `seed.json`.
The seed config uses the baked mesh and five frames with roles `support`/`heldout`.

```bash
python -m tools.track3.r78 freeze --config run/seed.json --output run/seed.frozen.json

# GPU, FP runtime: exactly five normal registrations, not a scale/grid search.
python -m tools.track3.r78 seed --manifest run/seed.frozen.json --weights weights/foundation_pose --output run/seed1

# CPU: consumes seed1's hash-bound mesh/pose pairs; no FP calls.
python -m tools.track3.r78 fit --manifest run/seed1/fit_manifest.json --output run/fit1

# CPU: pool.json lists fit1/fit2 in the original predeclared order.
python -m tools.track3.r78 select --pool run/pool.json --output run/selected
```

For a second candidate, use its own review, seed manifest, seed output and fit output;
do not reuse the first candidate's poses. The source frames, observations and split
must be identical across candidates. A single feasible candidate can be delivered,
but does not test selection quality. Missing candidates require an explicit failure
receipt; absent/unfinished output is not silently accepted as a completed comparison.

For tracking, create `track.json` with the same provenance and per-frame fields as the
examples, listing the entire **ordered consecutive** sequence. `mesh`, review and poses
are taken from the selection, not from this config. The sequence must include the
same five frozen support/heldout observations. Then:

```bash
python -m tools.track3.r78 freeze --config run/track.json --output run/track.frozen.json
# GPU, FP runtime: one register then track_one; fresh output only.
python -m tools.track3.r78 track --manifest run/track.frozen.json --selection run/selected/artifact.json --weights weights/foundation_pose --output run/tracked
```

The tracking adapter calls the existing `FoundationPoseTracker` with the provided
mesh: TensorRT, register 10 / track 5 iterations, seed 0, mask depth, one particle,
no re-registration and **skip_object_scale_estimation=true**. It does not reuse the
five R8 poses as a tracking cache. All output paths must be new; failed partial
outputs are preserved and cannot be resumed merely because files exist.

The metric mesh is also compatible with the existing pipeline's explicit
`--object_mesh_path <selected-fit>/mesh.obj --skip_object_scale_estimation` entry.
Use a **fresh** pipeline output and the same frozen inputs. That full pipeline has
additional stages/defaults; the narrow `track` command above reproduces this method's
FP-only boundary and does not automatically run refinement or rendering.

## Data contract

Each frame lists `id`, `rgb`, `K`, `depth`, `validity`, `mask`; seed frames additionally
have `role`. Fit manifests produced by `seed` also bind every `pose` to a completed
seed artifact and the exact mesh/input hashes. RGB, masks, validity and depth must
share one pixel grid; K must be a finite zero-skew OpenCV pinhole matrix on that grid.
Masks/validity are single-channel arrays or images, with positive values valid.

Depth encoding is explicit:

- `float_m_npy`: floating-point camera-Z metres. No inferred units.
- `v2d_inverse_uint16`: `Z = 65535 / code - 1`; codes 0 and 65535 are invalid.

At least 500 valid object depth pixels are required. A binary validity map is not
claimed to be stereo confidence. No dynamic-object point clouds are stacked through SLAM.
`freeze` generates required hashes; it refuses to overwrite a manifest or refresh a
previously hashed input that changed. Moving data between hosts requires preserving
bytes and updating path references explicitly. Generated seed/selection manifests
contain runtime paths; those paths must remain accessible or be deliberately relocated.

Default solver settings reproduce ep27: 15 iterations; bidirectional robust surface
correspondences; 0.15 m maximum correspondence distance; 90% retained; 0.01 m Huber
scale; uniform multiplier [0.5, 2.0]; pose updates bounded to 0.15 m / 30 degrees
relative to each initialization. These limits are not relaxed on failure.
`ITERATION_LIMIT` means a bounded candidate, **not convergence**. Support or heldout
boundary violations produce `SCALE_INCOMPLETE` failure and no accepted artifact.
Heldout SE(3) uses 80% spatial blocks; the remaining 20% measure input consistency.
Thus heldout frames are not independent pose ground truth or zero-adaptation validation.

## Output pairing and coordinates

Every completed stage writes `artifact.json` with input/code provenance and output
SHA-256 hashes. A fit writes `mesh.obj`, appearances, five `pose_*.npy`, support/heldout
traces and metrics. Selection references the exact bounded fit artifact. Tracking
writes the exact selected mesh plus matching `poses/*.npy` and native Transform3d
`poses/*.json`, suitable for the normal result writer, and a completion artifact.

Let `c` be the input mesh bounding-box center. R8 exports
`v_metric = c + s * (v_input - c)` and `T_public = T_center @ Translation(-c)`.
The exported poses act on the **exported original-coordinate mesh**, in OpenCV camera
coordinates, metres, with rigid rotations and unit pose scale. Scale is already baked;
do not multiply it again. Camera/world translations and hand scale are untouched.
Changing mesh or observations invalidates seed-pose compatibility; use the new
asset's registrations. Never pair a new mesh with an old pose cache.

## Validation and measured scope

```bash
python -m unittest discover -s tools/track3/r78/tests -v
```

CPU tests cover the shared similarity solve, independent SE(3), fixed heldout size,
nonzero-center mesh/pose equivalence, input/hash/review rejection, explicit failure
states and a tiny end-to-end flow with the GPU backend replaced by a test double.
The published CPU extraction reproduced both ep27 C1/C2 fits from saved inputs:
scales **1.009085684200183 / 1.0043439089416184**, identical five poses and byte-identical
metric meshes; selection remained C1. This publication did not rerun GPU inference,
tracking, reconstruction, training or scoring.

Historical same-front-end **ep27, all 246 frames**, A vs input-selected C1:
scene reconstruction AUC **0.266514 → 0.912729**; inter-object RPE
**41.408610 → 1.599919 cm**. These are Public local reconstruction results.
They support this asset-generation/selection/size/tracking combination on that case;
cross-case generalization remains under validation. They do not establish better
overall shape accuracy or improved final policy/hidden-test performance.
