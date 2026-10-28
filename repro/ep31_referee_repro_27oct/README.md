# ep31 CoTracker referee 复现包 / reproduction package (27 Oct 2026, Ken)

**Case:** public ep31, object `mini_sweeper` (anchor), run `babel_fs_s3mf_d60` — SAM 3D mesh drawn at frame 60, SAM 3 object
masks, FoundationPose tracked from frame 0, EKF smoothed. The CoTracker3 referee relabels the raw FP poses among the mesh's
180° twins (PCA axes) relative to frame 0 and writes the relabelled **raw** poses as `poses_smoothed` (EKF not re-run, as shipped).
**Official replay AUC (unrefined bundle, partner plastic_dish_rack from the same run): 0.771 → 0.882** (`scores/`).
Same referee over 40 ep31 runs: `scores/ep31_referee_before_after.csv`.

| Path | Content |
|---|---|
| `cotracker/ep31_sweeper.npz` | `tracks` (290, 640, 2) float32 pixel xy per frame; `vis` (290, 640) bool; `queries` (640, 3) = (frame, x, y), query frames 0, 30, …, 270 (64 points each); `size` (H, W) |
| `cotracker/referee_summary_babel_fs_s3mf_d60.json` | referee report: anchor frame 0, PCA axes/extents, query frames, points/pairs, labelled frames, twin segments `[start, end, twin_id]` |
| `cotracker/referee_relabelled_raw_poses/` | referee output, 290 JSON (= `poses/referee_corrected`, byte-identical) |
| `mesh/mesh_scaled.obj` (+ material, `provided_mesh_dir/`) | the exact mesh tracked and scored |
| `poses/raw_fp/` | raw FoundationPose poses, 290 frames |
| `poses/ekf_smoothed/` | EKF-smoothed poses (pipeline call-site q = r = 0.02), the "before" input to scoring |
| `poses/referee_corrected/` | relabelled raw poses, the "after" input to scoring |
| `camera/intrinsics_stable.json`, `camera/depth_scale.json`, `camera/source_run.json` | K (fx, fy, cx, cy, W, H) + depth scale + the run's command / reference frame |
| `depth_query_frames/`, `rgb_query_frames/` | stereo depth (pipeline DepthImage PNG; read with `v2d.common.datatypes.DepthImage.load`) and RGB at the 10 query frames |
| `scores/score_records_d60_before_after.json` | every official `recon_ourmesh` result line for the two namespaces (`babel_fs_s3mf_d60_noref` vs `babel_fs_c0_s3mf_d60_noref`), incl. per-object diag |
| `code/` | `cotrack_eval.py` (referee used here) and `score_ep.sh` (scoring) |

Pose JSON = FoundationPose pose of the **uncentred** mesh in the OpenCV camera frame (placed pose = pose @ inv(tf_to_centered)).
Scoring: `ep/score_ep.sh <tag> 31 "31:mini_sweeper 31:plastic_dish_rack" babel_fs_s3mf_d60 $D/diag/cotrack/ep31_sweeper.npz mini_sweeper`
(builds `_noref` bundles, then `tools/track3/score_playback.py`). Ground truth was used only to compute the scores.
