# Input-constrained metric scene refinement (experimental)

Opt-in CPU extraction of the shared-shape and full-sequence pose solvers used in
Public ep21/22. It consumes existing measured inputs and matching predictions;
it does not generate depth, masks, hands, models, GT registrations or policy data.
Existing reconstruction defaults are unchanged.

The reusable reconstruction stages are:

1. [R7/R8](../../../../tools/track3/r78/README.md): input-reviewed view/structure
   selection, SAM3D candidates, shared uniform size with independent support-frame
   SE(3), heldout checks, and provided-mesh tracking.
2. `part_shape`: nine shared body/part parameters plus independent support poses.
   Two input-annotated part vertices and a fixed center define a geodesic basis.
3. `contact_shape`: 16-node smooth shared deformation plus support poses, jointly
   constrained by silhouettes, measured surfaces and an observed body/table contact.
4. [Metric tracking](../metric_mesh_tracking/README.md): normal FP with a frozen
   metric mesh; optionally use a matching provided pose and observed table-contact
   point. No new scale search or depth modification.
5. `temporal_pose`: fixed shape/size, calibrated stereo landmarks, all-frame
   silhouettes, temporal correction priors and observed rest-contact intervals.
   It exports a full native bundle, preserving the other objects, hands and camera.

These are bounded research candidates, not an automatic production replacement.
The two historical shape experiments **failed their original input acceptance
checks**, despite later local Public gains in downstream branches. The release
preserves those failures. Do not use GT scores to bypass them, silently reuse their
auxiliary poses as a full trajectory, or assume this chain has been validated on Scored20.

## CPU entry

From the repository root, Python 3.11:

```bash
python -m venv .venv-refinement
.venv-refinement/bin/pip install -r baseline/reconstruction/scripts/metric_scene_refinement/requirements.txt
PY=.venv-refinement/bin/python
ENTRY=baseline/reconstruction/scripts/metric_scene_refinement/run.py

# Copy one example, fill its paths, frame roles, hashes and input-side review.
$PY "$ENTRY" freeze --spec run/spec.json --output run/frozen.json
$PY "$ENTRY" validate --manifest run/frozen.json
$PY "$ENTRY" run --manifest run/frozen.json --output run/candidate
```

Paths in `spec.json` resolve relative to that file. Freeze records file and code
SHA-256 hashes; the frozen manifest is specific to those runtime paths. Do not put
runtime manifests or assets in Git. A changed input requires a separately reviewed
new run, not overwriting an existing manifest. Output must be new and separate from
all inputs; there is no implicit resume. `validate` and these three solvers are CPU-only.

Examples deliberately use `structure_review.accepted: false`. Set it true only
after reviewing target identity, visible critical structures and contamination.
`mesh_pose_pair` requires the reviewed source `mesh[_N].obj` and `result.npz` SHA-256;
`python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' FILE`
prints either digest. No paths are embedded in implementation code.

## Inputs and conventions

- `bundle`: native `result.npz`, target `mesh.obj` for slot 0 or `mesh_N.obj` for
  slot N, all partner meshes and OBJ-referenced materials/textures. Target poses
  and camera must be complete finite SE(3), with `world = camera_to_world @ camera`.
  Target scale must be 1: physical size is already baked into vertices.
- `observations`: `K_fp.json` (`fx,fy,cx,cy,width,height`), binary
  `masks/000000.png`, RGB `rgb/000000.png`, native uint16 `depth/000000.png`.
  Depth is camera-Z metres decoded as `65535/code - 1`; 0 and 65535 are invalid.
  All images share the same pinhole pixel grid. No implicit resize/undistortion.
  Temporal fitting needs all masks and K; the shape fits need declared support and
  heldout RGB/depth/masks. RGB is used for review overlays, not as a feature loss.
- `part_shape`: `semantic_vertices` JSON has `parts: [vertex_a, vertex_b]` and
  `center: vertex_c`, annotated in this exact source mesh without GT/scan input.
- `contact_shape`: frozen per-frame binary `validity` PNGs; normalized world plane
  `[nx,ny,nz,d]` as NPY; `contact_annotation` JSON with `frame`,
  `mask_boundary_pixel: [u,v]`, and an explanation in `source`. The visible
  body/table interface is used, **not the global lowest vertex**. Annotation frame
  must be in support. The plane and camera transform determine the contact ray intersection.
- `temporal_pose`: `landmarks` NPZ with `frame_ids [F]`, `points_camera [F,J,3]`,
  boolean `qualified [F,J]`, and `sigma_z [F,J]` in metres; matching measurement
  receipt with `state: OBSERVABILITY_PASS`; `calibration` JSON containing
  `contact_point_local_m [3]`; normalized world plane. Landmark correspondence,
  stereo triangulation/observability and rest-phase identification are upstream
  inputs, **not implemented by this release**. They must be inspected/frozen before fitting.

Frame roles are explicit. Calibration is a subset of fit frames; fit and validation
landmark frames must be disjoint. Each landmark needs two qualified calibration
observations. Landmarks in the same `landmark_groups` entry share per-frame weight
mass, avoiding counting correlated markers as independent evidence. All-frame
silhouettes still consume validation frames: this is heldout **landmark** consistency,
not wholly unseen-frame validation. Historical Public GT exposure must be declared.

## Fixed numerical rules

No weights, thresholds or solver variants are selected from GT. Unrecognized
parameters fail. The only exposed choices are declared data/roles and wall-clock
budget (and fixed-size settings in the examples).

- Part shape: 15 updates; 3 body-axis fractions bounded by 0.25 and two 3D part
  shifts bounded by 0.25 of their geodesic radii; support pose bounds 15 cm/30°.
  Trusted depth is eroded object mask with finite 5x5 depth range <=20 mm.
  Depth Tukey cutoff 40 mm, normalizer 10 mm; silhouette normalizer 5 px.
- Contact shape: 16 deterministic controls, Gaussian partition minus center weights,
  15 updates; vertex movement <=80 mm, support/heldout pose bounds 15 cm/30°;
  same topology and edge lengths within [0.5,2] of original; no normal flips.
  Depth Huber/normalizer 10 mm, contour 5 px, contact 3 mm. Fixed-shape heldout
  poses are fitted separately; spatial depth blocks are withheld from fitting.
- Temporal pose: ten correspondence passes, at most eight least-squares evaluations
  each; soft-L1 loss; correction prior 40 mm/20°, second difference 3 mm/2°,
  contact 2 mm, contour 5 px. Final correction **norms** bounded by 60 mm/30°.
  The temporal prior regularizes corrections, not the observed trajectory to zero.

These are historical engineering bounds, not geometric truth. Reaching the fixed
iteration limit is not convergence. Missing common render/depth support produces a
failed run, never a low residual obtained by silently discarding the frame.

## Outputs and failure handling

`EXECUTION.json` records state, time, hashes and errors. Failed attempts remain on
disk and return nonzero. Nothing automatically starts FP, rewrites source data, or
promotes a candidate based on a reduced loss.

Shape fits write `output/mesh.obj`, appearances, fit traces, per-frame metrics,
comparison image and `MESH_POSE_PAIR.json`. Pose arrays contain **only their explicit
support/heldout frame IDs**, in `T_camera_from_original_mesh_metres` convention.
They are auxiliary poses, not a completed trajectory. A failed input gate is an
experimental artifact only. Successful numerical gates still require visual review.
For an accepted shape, use fresh normal FP with `--object_mesh_path` and
`--skip_object_scale_estimation`, or the metric tracking runner; keep the new mesh
and new matching poses together. Never attach the old full pose cache to a new mesh.

Temporal fitting writes full `POSES.npz`, landmark errors and a candidate `bundle/`.
It copies original mesh/materials and changes only the selected object's camera/world
poses. Norm-bound failures retain diagnostics **without** exporting a bundle.
`COMPLETE_BOUNDED_CANDIDATE` means computation completed within bounds, not that RGB,
depth, motion, hand/object relations or policy performance passed final acceptance.
`run.json` retains `input_gate_pending: true`, `production_adopted: false`.

## Validation and measured scope

```bash
$PY -m unittest discover -s baseline/reconstruction/scripts/metric_scene_refinement/tests -v
$PY -m unittest discover -s baseline/reconstruction/scripts/metric_mesh_tracking -p 'test_*.py' -v
$PY -m unittest discover -s tools/track3/r78/tests -v
```

Release extraction was reproduced on saved CPU inputs, without new GPU inference
or scoring: ep22 body/part basis, parameters, mesh and auxiliary poses were exactly
equal; ep21 contact graph, fitted mesh and auxiliary poses were exactly equal;
ep21 full 186-frame temporal output and every native bundle field were exactly equal
(maximum array difference 0). The original input-gate failures were reproduced too.
The optional FP contact hook is CPU contract-tested; its refactored GPU entry was
not GPU-rerun as part of publication.

Historical ep27 R7/R8: full 246 frames, same-front-end A/C1 scene AUC
**0.266514 → 0.912729**, RPE **41.408610 → 1.599919 cm**.
These are Public local reconstruction results; cross-case generalization remains
under validation. They do not establish better overall shape accuracy or final policy scores.
Later ep21/22 experiments are documented in [EVIDENCE.md](EVIDENCE.md). The current
Public20 development combination is not a fresh blind run of this pipeline and is
not a Scored20 result. No HF bundle is published with this code release.
