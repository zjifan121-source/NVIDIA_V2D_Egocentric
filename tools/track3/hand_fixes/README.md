# Hand fixes for scored episodes 3, 10 and 20 (haisenk, 2 Nov 2026)

Reference-hand fixes found by tracing why the policies fail on these episodes. Each was tested by training
3 seeds with the team recipe (`tracking_100m_object2x`, 100M steps, final checkpoint, NJMAX 1024) on our
`ken_c_scored` reconstruction and scoring own-ref add_auc (rollout vs our own reference; GT-free).
Data (scenes, rollouts, policies, ep3 left-hand npz): Hugging Face dataset `rose-tea-cafe/v2d-t3-recon`,
folder `experiments/haisenk/hand_fixes_v1_20261102/` (`<dl>` below). Code: this directory, plus two env hooks in
`tools/track3/run_load_retarget.py`: `H5_MERGED_NPZ` (replace the merged result.npz before cleanup) and `H5_PATCH`
(exec a file that redefines cleanup functions before `cleaned_bundle()`); both unset = unchanged behaviour.

## Why the policies failed

The tracking reward has contact terms driven by the reference's MANO contact labels (`missed_contact` -1,
`unintended_contact` -10 per object body). Where the reference hand is not on the object, the policy is
punished for holding it:
- **ep3:** the physical left hand was fitted as a right hand and mirrored, 23 cm from the pot. It had zero contact
  labels in all 313 frames, so the pot was never lifted.
- **ep20:** the right hand is undetected for frames 36-100 while it lifts the pitcher. The straight-line gap fill
  leaves it 11-15 cm off, with no pitcher contact for frames 49-84, so the pitcher was knocked over.
- **ep10:** FoundationPose spins the round blue cup 177 deg about its own axis. The attached hand is dragged 19 cm
  with it, and the release jumps 151 deg.

## Results (own-ref add_auc, seeds 42 / 43 / 44)

| ep | fix | fix runs | mean | baseline |
|---|---|---|---|---|
| 3 | correct-handedness left hand (A1, ships as is) | 0.914 / 0.803 / 0.895 | **0.871*** | 0.386 0.316 0.347 0.487 0.321 (mean 0.37) |
| 3 | same hand, rest hold pinned to the old frames (A2, comparable) | 0.755 / 0.647 / 0.650 | 0.684 | Edmond's 32 cm slide on the same base: 0.642 / 0.512 / 0.486 |
| 10 | cup spin taken from the hand (respinf), spin-invariant own-ref | 0.711 / 0.721 / 0.718 | **0.717** | control 0.690 / 0.667 / 0.701 (0.686) |
| 20 | right-hand gap refilled in the pitcher's frame (objrel) | 0.730 / 0.904 / 0.881 | **0.838** | 0.373 0.444 0.364 0.360 + control 0.353 (mean 0.38) |

\* A1 is scored against its own pot reference. With the correct hand, the pot starts moving at ~f65, as the raw
track and the video show; the old wrong hand froze the pot on the table until f122. So A1's number is not directly
comparable with the baseline. A2 keeps the old pot reference and shows the hand fix alone: 0.37 -> 0.68.

## Contents

```
episode_000003/  scene/ (A1, full load_retarget output incl. score_input/meshes), inputs/ep3_left_hand_merged.npz,
                 runs/ep3_h5A1_s4{2,3,4}/ (results/episode_000003.{parquet,json}, checkpoints.json,
                 policy_100003840.safetensors), alt_A2/{scene,runs}
episode_000010/  scene/ (respinf), runs/ep10_h5respinf_s4{2,3,4}/
episode_000020/  scene/ (objrel), runs/ep20_h5objrel_s4{2,3,4}/
code/            the branch's tools/track3 files (hooked run_load_retarget.py, hand_fixes/)
SUMMARY.json     per-run scores and the recommended run per episode
```

## How to use

**A. Submit as is.** Add the episodes to the `tools/track3/pack_submission.py` plan. Each run dir has the evaluator
output; each scene dir has `score_input/meshes`:

```json
{"3":  {"run": "<dl>/episode_000003/runs/ep3_h5A1_s42",      "scene": "<dl>/episode_000003/scene"},
 "10": {"run": "<dl>/episode_000010/runs/ep10_h5respinf_s43", "scene": "<dl>/episode_000010/scene"},
 "20": {"run": "<dl>/episode_000020/runs/ep20_h5objrel_s43",  "scene": "<dl>/episode_000020/scene"}}
```

Picking a seed by own-ref is GT-free, but it carries a winner's curse: the seeds of each arm are within noise of
each other, so plan on the arm means.

**B. Retrain with your recipe.** Rebuild the scene in your clone with the branch, from the same `ken_c_scored`
bundles. The flags are those of `kenc/scene.sh`:

```bash
# ep3 (A1): only the merged bundle's left-hand arrays change
H5_MERGED_NPZ=<dl>/episode_000003/inputs/ep3_left_hand_merged.npz V2D_BUNDLE_NS=ken_c_scored \
  python tools/track3/run_load_retarget.py --episode 3 --merge --attach-hands --smooth-hands 2 --skip-video \
  --device cpu --sequence-id ep3_handfix --out <out>
# ep10: cup spin from the hand, applied to blue_cup / beige_cup only
H5_PATCH=tools/track3/hand_fixes/ep10/patch_respinf.py V2D_BUNDLE_NS=ken_c_scored python tools/track3/run_load_retarget.py --episode 10 ...same flags...
# ep20: right-hand gap refilled in the lifted object's frame
H5_PATCH=tools/track3/hand_fixes/ep20/p20.py V2D_BUNDLE_NS=ken_c_scored python tools/track3/run_load_retarget.py --episode 20 ...same flags...
```

The included scene parquets reference meshes and URDFs by absolute Babel paths under
`/data/user_data/haisenk/v2d/hand5/scenes/`, so rebuilding is simpler than retraining on them directly.

## Caveats

- These are own-ref results only; no official score yet.
- The scenes inherit the `ken_c_scored` meshes, the same practice as the team submission: the ep20 pitcher is the
  ep22 donor, and the ep3 pot is the ep43 donor.
- **ep10:** the gain is small (+0.03, 3/3 seeds above every control). It is all on the beige cup, and control
  training showed many more NJMAX overflow warnings than respinf. The official scorer ignores cup spin, so
  respinf's reference change is score-neutral there.
- **ep20:** do not apply `p20.py` blindly to other episodes. On ep1 it fires a false positive.
- **ep3:** in every arm the spoon starts 3.7 cm above the table. A seat fix is a likely cheap next step.
- `ep3_left_hand_merged.npz` is the `ken_c_scored` ep3 merged bundle with the 7 `hand_left_*` arrays swapped. The
  arrays come from ChatGPT's left-label HaMeR re-run of track 3 (31 Oct), which used no GT. For a different bundle,
  transplant those arrays (`hand_fixes/ep3/make_left_hand_npz.py`) only if its camera and right-hand arrays match
  ours.

## Files

| path | what |
|---|---|
| `ep3/make_left_hand_npz.py` | builds the ep3 merged npz: baseline merged bundle + the 7 left-hand arrays of the left-label re-run |
| `ep3/patch_A2.py` | alt A2 only: pins the rest hold to the old frames (pot 122, spoon 21) for a like-for-like own-ref A/B |
| `ep10/patch_respinf.py`, `ep10/sym_common.py` | redefines `attach_hands`: for blue_cup/beige_cup the axial spin comes from the holding hand (frame-0 spin otherwise), then the normal attach with a shortest-path blend; other objects unchanged. Diagnostics to `$H5_DIAG_DIR` |
| `ep10/sym_ownref.py` | spin-invariant own-ref (min over 360 axial rotations, as the official AUC.py treats cups) from a rollout parquet |
| `ep20/p20.py` | redefines `fill_hand`/`attach_hands`/`cleaned_bundle`: a hand gap >= 10 frames while an object is lifted is refilled in that object's frame between GT-free anchors (default arm; `H5_PUSH=1` push-out variant was not better) |
| `bench/contact_audit.py` | per hand/object: lifted frames that have reference contact labels (what the contact reward sees) |
| `bench/cmp_pq.py` | column-wise diff of two scenes' processed parquets |

Tested: each scene was rebuilt from exactly these files (team `main_buffer` `run_load_retarget.py` + hooks) and its
`bundle_cleaned` matched the trained scene (see commit message).
