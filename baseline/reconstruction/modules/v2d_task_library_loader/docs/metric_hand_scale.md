# Metric hand scale in ego reconstruction bundles

The ego loader consumes `hand_left_scale` and `hand_right_scale` from
`result.npz`. Each is a positive finite scalar or an array of shape `(N,)`.
Missing values default to 1, preserving older bundles. Object meshes are already
metric in a result bundle; do not apply `object_scale` a second time.

Reconstruction applies hand scale about the pre-translation mesh centroid:

```text
X_camera = h * (V - c) + c + t
```

Here `V` is MANO geometry, `c` its centroid, `h` the saved hand scale, and `t`
the MANO translation parameter. `t` is not the anatomical wrist position.
The loader uses reconstruction's right-MANO/mirror-X convention to obtain `c`
for the left hand before converting to native-left MANO parameters.

For a camera-to-world rotation `R`, translation `T`, and MANO template wrist `J0`,
the equivalent world-frame parameters are:

```text
t_world = R @ (J0 + t) + T - J0
c_world = R @ (c - J0) + J0
```

The existing world-repose check includes scale and its center: world FK must
match the camera FK transformed by `camera_to_world`.

The loader writes four additional time-series columns to `ManoSharpaData`:

- `mano_left_scale`, `mano_right_scale`
- `mano_left_scale_center`, `mano_right_scale_center`

Older Parquet files remain readable; missing columns are null. Parameter-based
consumers must treat absent scale as 1. Consumers of the saved metric joint
positions must **not** scale those joints again. The hand-to-robot mapping
(`mano_to_robot_scale`) is a separate retargeting parameter.

If a separate application converts an entire scene by a factor `s`, simply
multiplying MANO translation by `s` is insufficient. The equivalent hand
parameters are `h_new = s*h`, `t_new = s*t + (s-1)*c`. Object mesh vertices,
object translations, and camera translations must use the same scene units.
This loader does not estimate or apply a new scene scale.

## CPU regression checks

In the task-library-loader environment, from the repository root:

```bash
python -m unittest discover \
  -s baseline/reconstruction/modules/v2d_task_library_loader/tests \
  -p test_mano_scale.py -v
```

The synthetic checks do not require MANO weights. To also exercise real MANO
kinematics, set `MANO_ASSETS_ROOT` to an authorized model directory containing
`models/MANO_RIGHT.pkl` and `models/MANO_LEFT.pkl`. Do not commit the model files.
Use the updated loader and schema together when rebuilding the loader image.
