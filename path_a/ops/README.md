# Hyperstack reconstruction operations

Use the repository root. This is the `wentacc_dev` experiment lane on
`creative-tesla`; the baseline instructions remain in the root README.

## Storage and receipts

The data volume is mounted at `/mnt/v2d`. Docker **and containerd** image stores,
weights, caches, datasets, temporary files and outputs are on this volume.
`setup_hyperstack_storage.sh` is a guarded, one-time provisioning script, not a
routine startup command. The original root-disk stores have been preserved.

| Content | Host path |
|---|---|
| Checkout | `/mnt/v2d/code/NVIDIA_V2D_Egocentric` |
| Shell environment | `/mnt/v2d/code/v2d.env` |
| Dataset | `/mnt/v2d/datasets/track_3` |
| Models | `/mnt/v2d/weights` |
| Experiment outputs | `/mnt/v2d/outputs/wentacc_dev/epNN/OBJECT` |
| Setup, image and model receipts | `/mnt/v2d/logs` |

The dataset revision is `5f68335f3acc802033d1e80728c1633197521de8`.
`descending_queue_inventory.json` was generated on 2026-10-13 from actual
`ego_cam_a` files and ffprobe frame counts, public Parquet `objects` metadata,
and the official evaluation object roster. It covers 29 episodes, 58 objects,
and 7,185 source frames at or below ep33. Preserve each episode's object order.
This inventory describes available work; it does not claim anything has run.

## Image and model compatibility

Reuse the 12 GHCR `afe7dad` images with immutable digests recorded in
`reconstruction_images_afe7dad.json`. R570.195.03 and the A6000 pass the
container GPU check. FlashCHORD's separate driver requirements are out of scope
for reconstruction setup.

Three compatibility fixes were needed on this host:

1. The published FoundationPose image lacks the TensorRT runtime needed by the
   checkout's deployable model backend. `FoundationPose.TRT.Dockerfile` adds the
   pinned runtime to that existing image. Record its identity in
   `/mnt/v2d/logs/image_overrides.json` and pass that file to the runner.
   **Re-running the image pull/tag script resets the local `latest` aliases;**
   restore the recorded derived FoundationPose alias before a run.
2. Weight downloading uses `--dev` to run the matching checkout source.
   `prepare_weights.py` primes the public MoGe cache outside the old SAM3D
   image to avoid its incompatible httpx2/Brotli downloader combination.
3. WiLoR downloads `pretrained_models/MANO_RIGHT.pkl`, but its manotorch
   renderer expects `pretrained_models/models/MANO_RIGHT.pkl`.
   `ensure_mano_layout.py` copies the user's licensed model into the missing
   location. It checks hashes, is idempotent and refuses conflicting files.
   It does not redistribute models.

HF credentials are passed through encrypted SSH stdin for setup, never saved
in this repository or command arguments. Model-license acceptance is separate
from technical setup. Existing authorization must be respected.

## Run, resume and review

```bash
source /mnt/v2d/code/v2d.env
cd /mnt/v2d/code/NVIDIA_V2D_Egocentric
python path_a/ops/run_episode.py --config path_a/ops/ep33.json \
  --object blue_cup --image-overrides /mnt/v2d/logs/image_overrides.json
python path_a/ops/status_report.py --write
```

Launch long runs using a unique detached systemd unit with user `ubuntu`, the
checkout as working directory, and append-only service logs under
`/mnt/v2d/logs`. The runner takes a shared single-GPU lock. On this host the
initial unit is `v2d-ep33-blue-cup`; the first recovery unit is
`v2d-ep33-blue-cup-resume1`. Inspect actual status before creating another unit.

Each attempt keeps its command, commit, image IDs/digests, configuration and
hashes, tracked source patch, package versions, setup receipts and pipeline log
under `runs/<UTC timestamp>/`. Each object's `status.json` and the experiment
`episode_status.jsonl` retain results. `status_report.py --write` rebuilds
`CHECKLIST.md` and `checklist.json` and appends capacity observations.

Use `--resume` only after checking the failed stage and upstream cache coverage.
The upstream pipeline sometimes treats any file in a directory as completion;
a partial directory is therefore unsafe. Preserve the failed log and record
why a stage is reused or moved aside. Never overwrite a complete valid result
merely to make a new attempt.

The 2026-10-13 blue-cup attempt stopped at MANO prompt rendering after all 348
frames of RGB, depth, intrinsics, SLAM and WiLoR had finished. Recovery checked
every expected file and decoded/parsed the caches, then continued from prompt
rendering. The evidence is in that first run's `recovery_review.json`.

Runner success yields `NEEDS_VISUAL_REVIEW`. Before marking `VALIDATED`, inspect:

- Frame coverage, finite rigid transforms, mesh geometry/materials, gravity
  metadata and diagnostic video decoding from `numeric_qa.json`.
- Object identity and scale, masks and pose overlays at the beginning, middle,
  end and occlusions; temporal jumps and failed hand/object contact.
- Mesh shape from several angles and the final Three.js scene. Keep sampled
  images and a dated visual-review record in the output directory, with any
  limitations. A decoded video alone is not visual validation.

Run the wooden spoon only after the blue cup passes review. Finish both objects
before proceeding to ep32, then follow the actual inventory descending. Check
each new video's orientation/reference frame before creating its run config.
When reusing between objects, only complete, parameter-identical, object-
independent outputs may be reused: calibration, RGB/depth/intrinsics, SLAM and
raw WiLoR. Object masks/mesh/poses and joint hand-object refinement are separate.
The independently refined object bundles also require a common-frame check
before downstream multi-object use.

## Capacity and timing

Warn below 60 GiB free; the runner refuses a new job below 40 GiB. These are
initial safety margins, to be raised if observed transient usage requires it.
Do not purchase expansion or delete existing artifacts automatically.

At the early ep33 stage, common per-video outputs occupied about 2.2 GiB for
348 frames. Duplicating that footprint per object across the current inventory
would use approximately 91 GiB **before** object-specific outputs. Sharing one
complete common cache per episode would reduce that component to about 45 GiB;
neither estimate includes temporary render frames or later refinements.

As an explicit planning scenario, 1–3 additional GiB per object, about 139 GiB
of fixed environment/data, duplicated common caches and 40 GiB headroom imply
roughly 328–444 GiB total. This is not a measured final requirement. The current
280 GiB volume may need expansion or a reviewed artifact-retention strategy
for the full batch; ep33 is safe at the currently observed free space. Recompute
after the first complete object and after any longer or more difficult clip.

The initial 348-frame preprocessing/detection path took about 11 minutes before
the MANO layout failure. SAM3D, first-time TensorRT compilation and refinement
had not yet been measured at that point, so a full-run ETA is provisional.
Reserve 2–6 hours for the first two-object episode as a planning allowance;
replace that range with observed per-stage times. Do not multiply the short
preprocessing time into a claimed whole-batch completion time.

## Team handoff proposal

Recommended ownership for this lane: **multi-object reconstruction consistency
and public-set validation**, including a stable reconstruction-to-retargeting
interface. This is a proposed division of work, not an agreement with teammates.

- Ryan can own shared image releases and their smoke-test receipts, with the
  above compatibility fixes upstreamed into a future image revision.
- A grounding owner can develop retargeting, policy training and official
  rollout evaluation concurrently, using a versioned bundle from this lane.
- Path B can retain its independent approach while sharing inputs, validation
  conventions and final comparisons, rather than duplicating infrastructure.

Start with a public-set diagnostic and one focused comparison: unchanged
baseline versus a shared camera/metric-scale treatment across both objects.
Only after measuring errors should we choose scale correction, temporal pose
stability, or contact-aware refinement. Useful deliverables are error curves,
object-relative trajectories, runtime/storage cost and paired ablations on
the same clips. Avoid tuning solely on ep33, whose ground truth is hidden.

This direction is motivated by the official kit: RPE measures relative object
positions, MP-SR requires every object to succeed, and mesh registration does
not rescale rollout translations. Check body frames, metres and xyzw rollout
quaternions across the interface. Cup axial spin is treated as symmetric; cup
tilt, translation and spoon orientation remain meaningful error targets.

Suggested team update format: branch/commit, exact completed objects and QA
state, failure/fix, output schema and coordinate convention, next owned episode
range, and the one dependency needed from another owner. Do not call a rendered
reconstruction a valid scored Track 3 submission: policy checkpoints, official
rollouts, every required object mesh and packaging are still needed. Formal
submission remains subject to the user's confirmation.

Sources: [repository conventions](../../README.md),
[official Track 3 kit](../../assets/data/v2d_submission_kit/tracks/track_3/README.md),
[challenge site](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/).
