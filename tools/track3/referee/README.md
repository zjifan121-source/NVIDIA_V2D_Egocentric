# CoTracker3 twin referee (`referee`)

Opt-in clean-up of FoundationPose object tracks. Existing reconstruction defaults are unchanged.

FoundationPose sometimes puts an object on a 180-degree twin of its own mesh partway through a video: after a
re-registration when the mask overlap drops, or by sliding under the hand. On near-symmetric objects the twin renders
almost like the right pose, so FoundationPose does not notice, and the track flips back and forth. The metric's ADD
sees every flipped frame as a large error.

The referee decides, frame by frame, which of the mesh's four twins the track is on, using CoTracker3 2D point tracks as
independent image evidence, and turns flipped frames back to frame 0's branch. No ground truth is used.

1. **tracks** (GPU, CoTracker3): up to 64 points inside the eroded object mask at every 30th frame, tracked forward and
   backward over the whole episode with the offline model at 1/4 resolution. The tracks depend only on the video and the
   mask, so they are computed once per episode and object.
2. **relabel** (CPU): each query point is lifted with the stereo depth and kept if it lies within 1.5 cm of the posed
   mesh. At every other frame, the four twins (the identity and the 180-degree turns about the mesh's PCA axes) are
   compared by the median reprojection error of those points against their 2D tracks; a twin is accepted when its error
   is below 0.6 times the runner-up's. The twins form a Klein group, so per-frame labels relative to frame 0 follow by
   vote propagation; frames that get no label take the nearest labelled frame's. The output track is each frame's pose
   turned by its label. By default frame 0's branch is kept, because the metric aligns the whole scene on frame 0; all
   the evidence below uses that setting. When no frame can be labelled from frame 0, `relabel` writes nothing.
3. **apply** (CPU): runs `relabel` on every object of a namespace and writes a new namespace. Refereed objects get the
   relabelled raw poses as `poses_smoothed/`, which `tools/track3/write_unrefined_bundle.py` reads; cups and the pot lid
   are left unchanged (see the evidence). The run's old bundles are not carried over: rebuild them before scoring.

## Evidence (public episodes, Babel, Oct 2026)

Local official metric (`score_playback.py`: ADD-AUC of the reconstruction itself, no policy), unrefined bundles. "Raw"
is the FoundationPose track as the pipeline smooths it; "refereed" replaces the refereed objects' tracks with the
relabelled raw poses. Scores are two-decimal or three-decimal as the scorer printed them.

| Test | Raw | Refereed | Notes |
| --- | --- | --- | --- |
| ep31 sweeper, 14 tracks whose frame-0 anchor is right | 0.776 | 0.845 | 13 of 14 up, worst -0.012 |
| ep31 sweeper, 8 tracks whose frame-0 anchor is wrong | 0.06-0.09 | 0.06-0.09 | not fixed by the frame-0 referee |
| ep18, sweeper refereed (SAM 3 pipeline run; dust pan unchanged) | 0.755 | 0.916 | five other ep18 runs: -0.008 to +0.164 |
| 10 public episodes x 2 meshes, cups and lid not refereed | 0.493 | 0.514 | 20 runs, worst -0.009; ep22 pitcher +0.05 |
| ep13, cup and rack both refereed (2 runs) | 0.912 / 0.826 | 0.586 / 0.564 | rack only: 0.911 / 0.826 |
| this package, `apply` on a public namespace with donor meshes, 8 episodes | 0.473 | 0.479 | ep18 0.90 -> 0.91, ep39 0.77 -> 0.80, other 6 unchanged at two decimals |

In the last row `apply` refereed 13 objects, skipped the cups and the lid, and left the two ep23 boards unchanged because
no frame could be labelled from frame 0 (no query frame had 6 points within 1.5 cm of the posed donor meshes). Three
public episodes (0, 11, 22) were not scored because one object of each has no FoundationPose run in that namespace.
Relabelling also changed 212 frames of an ep24 board and 244 frames of the ep43 pot, but both episodes have a wrong
frame-0 anchor there (0.03 and 0.10) and stayed unchanged.

Variants that were tried and are not recommended: re-basing the labels on the most common branch instead of frame 0
(`relabel --anchor majority`, kept for diagnostics) collapsed 6 of the 14 good ep31 tracks to 0.06-0.09 and rescued 3
of the 8 bad ones (0.74-0.92); over all 22 it averaged 0.459 against 0.567 for the frame-0 anchor. Smoothness-only
unwrapping of the track (no image evidence) lost 0.02 on average on the 14 good ep31 tracks, with 4 of them dropping by
0.09-0.25.

The packaged `relabel` reproduces the original script's output on an ep31 track to floating-point rounding (largest
quaternion difference 1.3e-15 over 290 frames). The packaged `tracks` reproduces the stored ep31 tracks with identical
queries and visibility and a median position difference of 0.001 px (largest 0.39 px; GPU arithmetic is not
bit-for-bit repeatable).

What it does not do:

- It cannot repair a wrong frame-0 anchor, or the metric registering the mesh to the scan's twin; such episodes stay
  where they are.
- It only knows 180-degree twins about the mesh's principal axes. Other rotation errors (for example a thin spoon
  spinning 60-110 degrees under the hand) are not repaired, and such frames may still be turned by a twin label.
- Refereed objects lose the R10 EKF smoothing (the relabelled raw poses replace `poses_smoothed/`). The gains above are
  not separated from that change. Dropping the cups' smoothing alone costs about 0.23, as much as refereeing them
  (ep11: 0.857 / 0.871 -> 0.628 / 0.639), and relabelling a cup can hurt on its own (the ep13 row): cups are skipped.
- The measured gains are mostly on the sweeper, plus the ep22 pitcher. The boards and the pot have no test with a
  right anchor, and the ep39 dust-pan gain may come from the lost smoothing rather than the labels.
- The effect on a trained policy (the leaderboard metric) has not been measured.

**Licence.** CoTracker3's code (mostly CC-BY-NC 4.0, some parts MIT or Apache 2.0) and the `facebook/cotracker3`
checkpoint (CC-BY-NC 4.0) are non-commercial. This package imports CoTracker and does not include it. Check that this
fits your use before shipping the `tracks` step, and name it in the technical report.

## Running it

From the repository root. `RUN` is a finished object run (`frames/`, `poses/`, `depth/`, `intrinsics_stable.json`,
`masks/1/`, `run.json`, and its mesh: `mesh_scaled.obj` or the provided mesh named in `run.json`). `CT`, `CKPT_DIR`,
`TRACKS` and `OUT` are absolute paths you choose. Pass the frames the masks were made on (the pipeline's undistorted
frames).

```bash
# 0. CoTracker3 (once): code and the offline checkpoint
git clone https://github.com/facebookresearch/co-tracker "$CT" && git -C "$CT" checkout 82e02e80
hf download facebook/cotracker3 scaled_offline.pth --local-dir "$CKPT_DIR"

# 1. tracks (GPU; an environment with torch + CUDA and OpenCV, with the co-tracker repo on PYTHONPATH)
PYTHONPATH="$PWD:$CT" python -m tools.track3.referee tracks --frames "$RUN/frames" --masks "$RUN/masks/1" \
    --checkpoint "$CKPT_DIR/scaled_offline.pth" --output "$TRACKS/ep31_mini_sweeper.npz"
#    (or --video <the undistorted episode video> instead of --frames)

# 2a. one object (CPU; the tools environment)
uv run --project tools python -m tools.track3.referee relabel --run "$RUN" \
    --tracks "$TRACKS/ep31_mini_sweeper.npz" --output "$OUT/poses_refereed"

# 2b. or a whole namespace; track files are named ep<EP>_<object>.npz, EP not zero-padded
uv run --project tools python -m tools.track3.referee apply --outputs "$PWD/baseline/reconstruction/data/outputs" \
    --source my_namespace --target my_namespace_ref --tracks-dir "$TRACKS"

# 3. bundles (reconstruction host environment; it reads this repo's baseline/reconstruction/data/outputs), then score
python tools/track3/write_unrefined_bundle.py my_namespace_ref my_namespace_ref_noref 31:mini_sweeper 31:plastic_dish_rack
```

`apply` prints one line per object (refereed with its number of relabelled frames, skipped, or unchanged with the
reason) and writes `<target>/_referee/summary.json` plus each object's report. It hard-links the source runs into the
target (copies where linking fails), so replace files in the target, never edit them in place. `--skip` adds objects to
the default skip list; `--anchor ref` re-bases on the run's registration frame instead of frame 0 (not measured on
scores).

Cost: the tracks took 4-5 s per object on an H100 (290 and 235 frames) plus reading the frames, and about 50 s for
290 frames on an A6000; `apply` on 20 public objects finished in about two minutes of CPU.
