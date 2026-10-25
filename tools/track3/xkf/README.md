# Multi-key-frame mesh pick (`xkf`)

Opt-in mesh selection for stage 1. Existing reconstruction defaults are unchanged.

The pipeline builds the object mesh with SAM 3D from one masked frame, once. That mesh is a lottery: the frame may
see the object edge-on or blurred, and SAM 3D invents every side it does not see. This tool builds 12 candidates
instead (6 frames x 2 seeds), judges each one on 10 held-out frames it was **not** made from, and tracks the best.
No ground truth is used anywhere.

1. **plan** (CPU): the draw frames are the run's reference (key) frame plus 5 frames spread over the video where the
   object mask is at least 0.6 x its median area (and 2,000 px), more than 40 frames from the key frame. The key frame
   is left out if its mask is under 0.3 x the median. The held-out frames are 10 frames spread over the frames where
   the object is equally well visible.
2. **draw** (GPU, reconstruction host environment): one SAM 3D draw per frame and seed, made exactly as the pipeline
   makes its mesh (`image_to_mesh` with texture baking, mesh post-process and the stereo depth; the SAM 3D transform;
   FoundationPose scale estimation).
3. **score** (GPU, FoundationPose runtime): for every draw and every held-out frame more than 15 frames from the draw's
   own frame, FoundationPose registers the mesh from scratch and the fit is scored like the pipeline's size search:
   mask IoU + exp(-median |observed / rendered depth - 1|).
   The score is taken at the **placed** pose, `pose @ inv(tf_to_centered)`: `register()` returns the pose of the
   uncentred mesh while `_score_scale` renders the centred one, and scoring the raw pose makes the ranking close to
   noise.
4. **pick** (CPU): each frame's score relative to that frame's median over the candidates, averaged per candidate
   (candidates skip different frames, so plain means are not comparable). It prints the launcher command that tracks
   the pick with the run's own settings (`--object-mesh ... --keep-mesh-size`).

## Evidence (11 public episodes, Babel, Oct 2026)

Local official metric, unrefined bundles, scan-free. "Key frame" is the default mesh (one SAM 3D draw at the key
frame), averaged over the 6 key-frame draws of three seed replicates; "pick" is this method, averaged over the three
replicates (seeds 1/2, 3/4, 5/6). Everything else is identical between the two columns.

| Episode | Objects | Key frame | Pick (rep 1 / 2 / 3) | Gain | Team pipeline |
| --- | --- | --- | --- | --- | --- |
| 0 | white pot, lid | 0.238 | 0.546 (0.775 / 0.298 / 0.565) | +0.308 | 0.237 |
| 11 | blue cup, beige cup | 0.822 | 0.770 (0.779 / 0.700 / 0.831) | -0.052 | 0.667 |
| 13 | blue cup, dish rack | 0.914 | 0.879 (0.904 / 0.888 / 0.844) | -0.036 | 0.958 |
| 18 | sweeper, dust pan | 0.329 | 0.406 (0.910 / 0.159 / 0.148) | +0.077 | 0.145 |
| 22 | water pitcher, blue cup | 0.623 | 0.524 (0.657 / 0.488 / 0.428) | -0.099 | 0.597 |
| 23 | two wooden boards | 0.413 | 0.716 (0.577 / 0.972 / 0.599) | +0.303 | 0.043 |
| 24 | two wooden boards | 0.547 | 0.287 (0.178 / 0.204 / 0.480) | -0.259 | 0.029 |
| 27 | white pot, dish rack | 0.940 | 0.951 (0.953 / 0.976 / 0.925) | +0.011 | 0.963 |
| 31 | sweeper, dish rack | 0.077 | 0.577 (0.169 / 0.881 / 0.681) | +0.500 | 0.050 |
| 39 | dust pan | 0.489 | 0.843 (0.836 / 0.928 / 0.764) | +0.353 | 0.847 |
| 43 | white pot | 0.258 | 0.641 (0.849 / 0.953 / 0.121) | +0.383 | 0.555 |
| **Mean** | | **0.514** | **0.649** | **+0.135** | **0.463** |

The pick helps in 7 of 11 episodes. It avoids bad source views (ep31: the key frame sees the sweeper edge-on and
SAM 3D draws a stick; the pick takes a broadside frame). It does not fix 180-degree twins of near-symmetric objects
(boards, brush, dust pan, the pot's spin): both twins fit the held-out frames equally, so the big swings between
replicates (ep18: 0.91 / 0.16 / 0.15) remain. On ep24 the pick is worse than the key frame.

How those runs were made, so that a reproduction matches:

- **Masks.** SAM 3 object masks prompted at frame 0 with the episode's prompt, keeping the instance that best overlaps
  SAM 2's mask from the default pipeline, with SAM 2's mask on any frame SAM 3 lost; placed into each run's `masks/1/`
  before the run (Grounding DINO and SAM 2 skipped for the object). On the two boards (ep23, ep24) both objects share
  the prompt "wooden board", so each board's mask was taken at a point clicked on it at frame 0, and frames SAM 3 lost
  took the nearest frame's mask (SAM 2's mask there is the table under them). The repo's
  `--object-segmenter sam3` keeps the most probable instance instead and has no point option, so on two identical
  objects it gives both the same mask; runs made with it are a similar but not identical mask set.
- **Key frame and tracking.** Frame 0 (frame 32 on ep0, where no hand is in view at frame 0), for the draws and for the
  tracking; the launcher's default reference frame comes from `reference_frames.json` instead, so pass
  `--reference-frame`.
- **Depth and camera.** FoundationStereo stereo seed (`--seed`), DROID-SLAM, HaMeR hands, `--dev`; the picked mesh was
  tracked through the provided-mesh path (`--object_mesh ... --skip_object_scale_estimation`, as `--object-mesh
  --keep-mesh-size` passes) and scored on the unrefined bundle.
- **Held-out frames.** Every second mask for most objects; every mask for the ep24 boards and the ep27 / ep31 dish racks
  (`--heldout-step 1` reproduces those).

Re-run with this code from the same run folders and draws, the plan (draw and held-out frames) and the pick match the
original runs on ep43 replicate 1 (108 of 108 per-frame scores equal to 0.001) and ep23 replicate 2 (97 of 108 equal
to 0.001, the rest within 0.04: FoundationPose's registration is not bit-for-bit repeatable on the GPU). Those checks
reuse the original draws: a seeded SAM 3D draw is not bit-for-bit repeatable either (ep43 frame 0 seed 1 redrawn: a
slightly different mesh, scale 1.325 vs 1.297), so fresh draws give a similar but not identical candidate set. The
tracking command was not re-run.

## Running it

From the repository root, with absolute paths. `RUN` is a finished run of the object (it provides `frames/`, `depth/`,
`masks/1/`, `intrinsics_stable.json` and `run.json`). Make it with the stereo seed and the reference frame you will
track from.

```bash
# 1. plan (CPU; numpy + Pillow, e.g. the tools environment); the key frame defaults to RUN's reference frame
uv run --project tools python -m tools.track3.xkf plan --run "$RUN" --output "$OUT/plan.json" \
    --episode 43 --object white_pot --seeds 1 2

# 2. draws (GPU; the reconstruction host environment that runs run_ego_reconstruction.py, with the same container
#    settings, e.g. V2D_CONTAINER_RUNTIME=apptainer and V2D_APPTAINER_IMAGE_DIR on a cluster without docker).
#    Uses --dev by default, as the evidence did; --candidate f0s1 draws one candidate (to spread them over GPUs).
python -m tools.track3.xkf draw --plan "$OUT/plan.json"

# 3. scores (GPU; inside the FoundationPose image with this checkout's modules over /workspace, as the pipeline's
#    --dev runs mount them, plus the compiled FoundationPose build those runs preserve, a private /tmp, and RUN / OUT /
#    the checkout at the same paths). --cleanenv drops CUDA_VISIBLE_DEVICES: pin the GPU with
#    APPTAINERENV_CUDA_VISIBLE_DEVICES=<gpu> on the host, or concurrent runs share GPU 0.
FP=$V2D_APPTAINER_IMAGE_DIR/v2d_foundation_pose.preserve/workspace/v2d_foundation_pose/lib/FoundationPose
W=/workspace/v2d_foundation_pose/lib/FoundationPose
TMP=$(mktemp -d)
apptainer exec --cleanenv --nv --pwd /app --no-mount /opt --no-mount tmp --bind "$TMP:/tmp" --home "$TMP:/tmp" \
    --env FOUNDATIONPOSE_WEIGHTS_DIR=/data/weights_dir/foundation_pose \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --bind "$PWD/baseline/reconstruction/data/weights:/data/weights_dir" \
    --bind "$PWD/baseline/reconstruction/modules:/workspace" \
    --bind "$FP/mycpp/build:$W/mycpp/build" --bind "$FP/bundlesdf/mycuda:$W/bundlesdf/mycuda" \
    --bind "$RUN:$RUN" --bind "$OUT:$OUT" --bind "$PWD:$PWD" \
    "$V2D_APPTAINER_IMAGE_DIR/v2d_foundation_pose.sif" \
    python "$PWD/tools/track3/xkf/score_fp.py" --plan "$OUT/plan.json" --output "$OUT/scores.json"

# 4. pick (CPU); prints the launcher command that tracks the pick with RUN's seed, prompt, segmenter and frame
uv run --project tools python -m tools.track3.xkf pick --plan "$OUT/plan.json" --scores "$OUT/scores.json" \
    --output "$OUT/pick.json"
```

Cost per object: 12 draws, each a SAM 3D generation plus a 36-registration size search; about 108 FoundationPose
registrations for the scores (12 candidates x ~9 held-out frames); one pipeline run to track the pick.
