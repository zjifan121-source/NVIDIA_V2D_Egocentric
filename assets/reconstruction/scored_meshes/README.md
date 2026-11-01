# Scored reconstruction meshes

Metric meshes used by pipeline C for ep3, 6, 15, 16, 19, 30, 35 and 36.

- `wooden_spoon/`: shared by ep3, 6, 15, 16, 35 and 36.
- `mini_sweeper/`: shared by ep19 and 30.
- `ep*_poses.npz`: matching full-sequence raw FoundationPose outputs.
- `manifest.json`: episode mapping, frame counts and mesh/pose hashes.

Meshes are reused assets, not newly generated per episode. OBJ, MTL and texture must stay together. Units are metres; scale is already baked. Use provided mesh / skip scale with additional scale 1.0. Do not recenter the mesh or attach poses from another asset. Poses map original mesh coordinates to camera coordinates using column vectors; they are not world-space or GT-aligned poses.

These are target-object assets, not complete hand/camera bundles. Alignment improved in the eight-case comparison; ep15 retains a second-half regression and ep19 retains an input-evidence warning. No policy improvement is claimed.

## Reproduce tracking

The opt-in [metric mesh tracking entry point](../../../baseline/reconstruction/scripts/metric_mesh_tracking/README.md) freezes caller-supplied inputs, registers one reviewed anchor, and runs independent forward/backward FoundationPose tracking with the matching metric mesh. It includes a portable input example and CPU tests.
