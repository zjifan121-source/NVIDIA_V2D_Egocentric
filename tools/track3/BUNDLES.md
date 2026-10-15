# Sharing reconstruction results

After you reconstruct an object (stage 1), upload the result so load, retarget and training (stages 2–4) can run on it. Everything goes to one **private** Hugging Face dataset, `rose-tea-cafe/v2d-t3-recon`, in one fixed layout, about 10 MB per object.

**Bundle** = the final output of one reconstruction run for one object: per-frame camera, hand and object poses (`result.npz`) plus the object mesh.

**Check what's there and how good it is** at any time:
```bash
export V2D_BUNDLE_REPO=rose-tea-cafe/v2d-t3-recon
uv run --project tools python tools/track3/bundles.py audit --split evaluation
```
It lists every object the 20 scored episodes need (`--split public` for the practice episodes) with one state per object:

| State | Meaning | What to do |
|---|---|---|
| `MISSING` | Not uploaded | Reconstruct it |
| `REDO` | A **Must** row below fails | Rerun it |
| `SHOULD` | A **Should** row fails | Usable; explain in `run.json` `notes` |
| `OK` | Meets every standard | Nothing |

`--out report.md` also writes the table and the full numbers (`report.json`).

## 1. Standards

Kaggle scores the object's path in metres, without rescaling. It also registers our mesh onto the real scan before comparing poses. So a world that is too large, or a mesh of the wrong object, costs points on every metric. A bundle that fails a **Must** row gets redone. For a **Should** row, upload anyway and explain it in `run.json` `notes`.

| | Standard | How it's checked | Why |
|---|---|---|---|
| **Must** | Every object of the episode is reconstructed. Two-object episodes need both, with the **same reference frame**. | `audit` shows `MISSING` | A scored episode with one object can't be merged, trained or submitted |
| **Must** | Whole clip, every frame | frame count = the video's (`num_frames` in `episodes.json`) | The scorer matches frame by frame |
| **Must** | The object is tracked: no NaN, object valid on ≥ 95% of frames | `valid_fraction.object` | Gaps become guesses |
| **Must** | The mesh is the right object at the right size: **0.8–1.25×** the public scan | `mesh/scan` column (every object except `wooden_spoon` has a scan) | A blob or a wrong object breaks Kaggle's registration. We have seen meshes 3–12× too large. |
| **Must** | `run.json` and at least one overlay `.mp4` are uploaded | `run.json` / `overlay` columns | Without them nobody can tell how it was made or review it |
| **Must** | Hand scale **0.7–1.5** on both hands | `hand scale L/R` | Hands are the metric ruler. 2.6 means the whole world is about 2× too big. |
| Should | Hand scale **0.85–1.3**, left and right close to each other | same | Same reason, tighter bar |
| Should | Each hand valid on ≥ 95% of the frames where it's visible | `valid_fraction.hand_*` | A missing hand can't be retargeted to the robot |
| Should | Object moves ≤ 3 cm in the first 30 frames | `drift 30f` | In every public ground-truth clip the objects start at rest |

Handled on our side (stages 2–3), no action needed: the object never holding still at the start (`still from` = `never`, caused by frame-to-frame jitter), an object resting on an edge, and robot fingertip error. These still show in the audit so we can track them.

### What usually fixes a failure

- **Mesh far too large, or a box-like blob:** the mask picked up the background or the wrong object. Check the `prompt` and choose a reference frame where the object is clearly visible and not occluded, then rerun.
- **Hand scale far from 1:** usually a bad depth scale or a hand that is barely visible. Choose a reference frame where both hands and the object are visible.
- **Lid or flat object too thick:** the mask probably includes what's under it, such as the pot. Same fix as a blob.

## 2. One-time setup

1. Send Edmond your Hugging Face username. You'll get an invite to the `rose-tea-cafe` organization with the **write** role.
2. Create a classic **Write** token at https://huggingface.co/settings/tokens. Fine-grained tokens only work if `rose-tea-cafe/v2d-t3-recon` is added to them explicitly.
3. Log in and set the dataset:
   ```bash
   hf auth login            # or: uv run --project tools hf auth login
   export V2D_BUNDLE_REPO=rose-tea-cafe/v2d-t3-recon   # put this in your ~/.bashrc
   ```

## 3. Before you run: claim and configure

- **Claim** the episode in the team channel before starting, so nobody runs it twice.
- **Run with `tools/track3/run_baseline_reconstruction.py`.** It uses the agreed settings and writes `run.json` (command, git commit, prompt, reference frame) next to the output:
  ```bash
  baseline/reconstruction/.venv/bin/python tools/track3/run_baseline_reconstruction.py --episode 41
  ```
  If you run `run_ego_reconstruction.py` by hand instead, keep these the same:

  | Setting | Value |
  |---|---|
  | Flags | `--undistort --hand_tracking hamer --run_droid_slam --run_gravity_alignment --run_gsplat_refinement --export_threejs_result` |
  | Object prompt | the `prompt` for that object in `tools/track3/episodes.json` |
  | Reference frame | the `suggested` frame in `tools/track3/reference_frames.json` (a frame where both hands and the object are visible). If you override it, say why in `notes`. |
  | Video | the whole clip, every frame. Don't trim. |

- **Two-object episodes:** one run per object, each in its own folder, with the same reference frame for both. Don't merge them yourself; merging happens in stage 2.

## 4. Folder layout

Names must match exactly, because the tools look things up by path:

```
episode_000041/                  6-digit, zero-padded episode number
└── white_pot/                   object name exactly as in episodes.json (not "pot", not "White Pot")
    ├── result_slam_gravity_aligned/     the final bundle, as the pipeline wrote it
    │   ├── result.npz                   per-frame camera, object and hand poses
    │   ├── mesh.obj + material.mtl + material_0.png
    │   └── manifest.json
    ├── *overlay*.mp4                    at least one overlay video, for visual review
    ├── run.json                         how it was made (written by the runner)
    └── status.json, provenance.json, numeric_qa.json   optional, uploaded if present
```

**Don't upload:** `frames/`, `depth/`, `masks/`, SLAM caches or other per-frame intermediates (GBs, and they can be regenerated), `threejs_scene/`, logs, weights, or MANO files. The MANO licence forbids redistributing them.

If you ran without the runner, write `run.json` yourself:
```json
{"episode": 41, "object": "white_pot", "prompt": "white pot", "reference_frame": 8,
 "git_commit": "<sha>", "images": "afe7dad", "who": "<name>", "finished_utc": "2026-10-14T16:00:00Z",
 "notes": "anything unusual: retries, overridden reference frame, failed Should rows and why"}
```

## 5. Check, upload, download

1. **Watch the overlay video.** Look for the wrong object, a lost mask, a flipped object, or hands on the wrong side. Most failures are visible here.
2. **Run the bundle checks** before uploading (CPU is fine, about a minute). The `warnings` list in `report.json` maps to the table in section 1:
   ```bash
   $RETARGET_VENV/bin/python tools/track3/run_load_retarget.py \
       --bundle <your output>/result_slam_gravity_aligned --object white_pot \
       --sequence-id check_ep41_white_pot --out /tmp/check_ep41 --skip-video [--device cpu]
   ```
3. **Upload** from the repo root. Do a dry run first: it prints every file and the total size and uploads nothing.
   ```bash
   uv run --project tools python tools/track3/bundles.py upload --episode 41 --object white_pot \
       --src baseline/reconstruction/data/outputs/<branch>/episode_000041/white_pot --dry-run
   # then again without --dry-run
   ```
   `upload` takes only the files in section 4, from the run folder (the one that holds `result_slam_gravity_aligned/`).
4. **Audit** (top of this page) to confirm your object now shows up.

```bash
uv run --project tools python tools/track3/bundles.py list                  # what's there
uv run --project tools python tools/track3/bundles.py download --episode 41 # -> data/outputs/hub/
```
Downloads land in `baseline/reconstruction/data/outputs/hub/episode_XXXXXX/<object>/`, where `run_load_retarget.py --episode N --object NAME` finds them.

**Re-uploading** the same episode and object replaces it; Hugging Face keeps the old version in its history. Say in the team channel when you replace someone else's bundle, and why.

**No upload access?** Zip only the files in section 4 (`episode_XXXXXX/<object>/…`, not the whole output folder) and send it to Edmond.

## 6. Rules

- **Keep it private.** Our reconstructions of the scored episodes are a competitive advantage, and the challenge rules restrict sharing outside the team. Never upload to a public repo or post bundles outside the team.
- **Disk:** on Babel, keep outputs in `/data/user_data/<you>/`, never in the lab's shared `/data/group_data/maxlab/common_datasets`. Delete per-frame intermediates once the bundle is uploaded and checked.
