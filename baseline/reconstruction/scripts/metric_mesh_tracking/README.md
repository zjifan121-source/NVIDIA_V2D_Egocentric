# Fixed metric mesh → clear anchor → bidirectional FoundationPose

Independent opt-in entry point for pipeline C. The default reconstruction workflow is unchanged.
It consumes an existing metric mesh; it does not generate meshes, estimate scale, or reconstruct hands/cameras.
The matching eight-sequence assets and poses are in `assets/reconstruction/scored_meshes` at repository root.

## Method

1. Freeze RGB, K, object masks and metric depth on the same pixel grid and frame timeline. Keep the provided mesh in metres, with scale already baked once.
2. Rank frames by existing hand-excluded valid object-depth support, descending, earlier frame on ties. Visually choose the first eligible frame: identifiable rigid structures, no gross blur, decisive occlusion or neighbouring-object contamination. Record rejected higher-ranked frames and the reason before tracking or reading scores. This is an input-driven **manual review**, not a new learned/automatic keyframe selector. Supply that selected `anchor_frame`; this entry point does not guess hand exclusion or silently choose a frame.
3. Register once with seed 0, 10 iterations, TensorRT backend, normal rank-0 selection. Inspect `anchor_overlay.png` for identity and gross visible-structure alignment. A failure stops the run; do not select another scored hypothesis.
4. Reset two fresh trackers to the same accepted anchor. Track forward and backward with 5 iterations, masked depth, one particle. No recovery, referee, EKF, scale search or joint refinement. Restore chronological order without duplicating the anchor.
5. Export original provided-mesh-to-camera SE(3), matching mesh digest and frame IDs. Internal centered-mesh transforms are checked; do not recenter the exported mesh or attach another mesh's poses.

The historical anchors were ep3=110, ep6=144, ep15=66, ep16=52, ep19=208, ep30=66, ep35=312, ep36=255. These reproduce those inputs, not a rule for unrelated sequences.

## Inputs and environment

CPU: Python >=3.10, NumPy, OpenCV. GPU stages use the existing native reconstruction environment with `v2d` modules, CUDA, FoundationPose weights and matching prebuilt TensorRT engines. No engine building or environment installation is performed by this entry point.

Copy `input.example.json` and fill in the actual sequence, all frames and relative paths. Review flags default to false: set them only after checking metric geometry and the input-only anchor choice. `mesh_dependencies` lists every OBJ material and texture. K uses `{fx, fy, cx, cy, width, height}`. RGB is decoded as RGB; masks must contain only 0/1/255. Depth **must** be the native inverse-depth uint16 PNG encoding `metres = 65535 / stored - 1`; 0/65535 are invalid. Do not pass millimetre PNGs. RGB/depth/mask must already share the K grid; this runner never resizes or reprojects them.

Supply `runtime.json` containing the exact `FoundationPoseTracker.runtime_info` from the intended native runtime (backend, model_registry, model_version, tensorrt_version, gpu_compute_capability, models). Existing anchor receipts contain this under `runtime_info`; extract that object, not the whole receipt. `--source-root` is the native modules directory containing `v2d_foundation_pose`. The freezer hashes the source and the runner verifies it when imported. Different engines/source fail explicitly instead of silently changing the experiment.

## Run

From this directory, with caller-chosen paths:

```bash
python prepare.py freeze --spec input.json --payload "$PAYLOAD" \
  --runtime-info runtime.json --source-root "$MODULES" --output "$PACKAGE"
python run.py --manifest "$PACKAGE/manifest.json" --protocol "$PACKAGE/protocol.json" \
  --payload "$PAYLOAD" --validate-only
```

`freeze` writes manifest/protocol digests and validates every input on CPU. Keep package and payload together or always pass `--payload` after remounting. It refuses an existing package directory.

In the native GPU environment:

```bash
python run.py --manifest "$PACKAGE/manifest.json" --protocol "$PACKAGE/protocol.json" \
  --payload "$PAYLOAD" --weights-dir "$WEIGHTS" --stage anchor --output "$OUTPUT"
# Inspect OUTPUT/anchor/anchor_overlay.png and registration evidence, then explicitly accept:
python prepare.py review --package "$PACKAGE" --output "$OUTPUT" \
  --reviewer reviewer_id --evidence anchor/anchor_overlay.png \
  --confirm-identity-structure-alignment
python run.py --manifest "$PACKAGE/manifest.json" --protocol "$PACKAGE/protocol.json" \
  --payload "$PAYLOAD" --weights-dir "$WEIGHTS" --stage track --output "$OUTPUT"
```

The shared 900-second window starts at anchor execution and includes review. Timeout checks occur between native calls; this is not a CUDA-kernel preemption mechanism. Use a process-level timeout externally if hard wall-clock termination is required. A failed/expired attempt is not resumed or overwritten. Do not reuse partial poses as a completed cache.

## Outputs and downstream handoff

`track/matched_mesh_poses.npz` contains ordered `frame_ids` and `public_poses[N,4,4]` (`T_camera_from_original_provided_mesh`, column vectors). `MESH_POSE_PAIR.json` pins mesh and pose digests, units and additional scale=1. `poses/*.json` are native Transform3d records. Each stage writes `RECEIPT.json`; only `COMPLETE_FULL_TRACK` with all N frames is a completed trajectory. Anchor candidates and raw failed outputs remain available for diagnosis.

This is an object tracking entry point, **not a full scene bundle exporter**. Downstream integration must preserve the matching mesh, combine with the intended common camera using the native coordinate convention, and preserve original frame IDs and hand validity. Never substitute GT-aligned arrays. Existing exported assets can be consumed without rerunning tracking.

## Validation and measured scope

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

CPU contracts cover direction schedules, coordinate pairing, depth decoding, input/provenance rejection and review gates. The extracted runner keeps the historical FP call order and numerical settings; this publication does not run new GPU inference and does not claim bitwise GPU parity after packaging.

Across the eight Scored target-object sequences (2,221 frames), same-observation alignment proxy increased from 0.630954 to 0.757114. This is not GT reconstruction AUC or policy performance. ep15 retains a second-half regression; ep19 retains missing baseline depth evidence; raw motion continuity is not uniformly better. No new R7/R8 fitting or automatic mesh generation is claimed for this eight-sequence run.
