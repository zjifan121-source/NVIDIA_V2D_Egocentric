# Scene fixes for scored episodes 1 and 5 (haisenk, 3 Nov 2026)

These scripts built the two scenes behind the ep1 and ep5 rollouts: `w_hold_ep1` (run `o_ep1_hold_s44`) and
`w7_attcom_ep5` (run `o_ep5_attcom_s42`). Both start from Wentao's reconstruction bundles (HF `rose-tea-cafe/v2d-t3-recon`,
`experiments/wentacc/...`) and change only what is listed below. Paths inside the scripts are the Babel paths they ran
with. They are byte-identical to the copies that ran (sha256 checked 4 Nov 2026).

Loader: `tools/track3/run_load_retarget.py` with the two opt-in env hooks of commit `5294c6b`
(`H5_PATCH`, `H5_MERGED_NPZ`; unset = unchanged behaviour). The 8-line diff is in `reference/`. ep1 uses `H5_PATCH`;
ep5 uses no hook.

## ep1: hold windows (scene `w_hold_ep1`)

- **Base:** `scored20_contactshape_proxy_mix_v1_20261103` (HF rev `fb11a3ce`), episode 1. Kept from Wentao:
  - objects `white_pot` and `white_pot_lid`, the same objects as in his `lead_sprint_v2` for ep1;
  - camera;
  - `support_v1` table.
- **Hands** (`wentao_base/wprep.py rawhands`):
  - His `hand_*` arrays are our cleaned hands, attached to our OLD object tracks.
  - They are replaced with our raw hands (`ken_c_scored_attach/bundle_merged`), re-expressed in his world as
    `camera_to_world @ wrist_to_camera`.
  - The loader then cleans the hands and attaches them to his objects:
    `--no-rest-hold --no-object-clean --no-level --attach-hands --smooth-hands 2`.
- **`ep1_hold_window/patch_whold.py`** (via `H5_PATCH`) redefines `attach_hands`. It keeps the team rule, but forces the
  hold windows shown by GT-free evidence:

  | hand → object | team rule (attaches only while the object is lifted) | forced window | evidence |
  |---|---|---|---|
  | left → pot | [127,221) | [32,304) | MANO contact labels f28-308; raw grip stable in the pot frame f30-305 |
  | right → lid | [47,59), [255,272) | [45,68), [250,274) | lid moves > 0.5 cm/frame |

  - The grip is the median hand-in-object offset over the hold interior. The team rule uses the offset at the single
    closest-wrist frame.
  - Only the `hand_*_wrist` arrays change.
- **Table** (`wentao_base/wsupport.py`):
  - Copied from the control scene, after checking that the frame-0 object poses are equal.
  - Wentao's `install_support.py` pins the bundle hash, so it rejects a changed `result.npz`.
- **Build:** `ep1_hold_window/build_w_hold_ep1.sh`, which calls `wentao_base/wbuild2.sh`.

## ep5: hand re-attach + pot COM (scene `w7_attcom_ep5`)

- **Base:** `scored20_lead_sprint_v2_20261103`, episode 5. Kept from Wentao:
  - objects `white_pot` and the new `wooden_spoon` mesh and track;
  - camera;
  - seated `support_v1` table.
- **Hands** (`ep5_attach_com/w7prep.py rawhands`, which is `wprep.py` pointed at v2):
  - Our raw hands, put in his world, then cleaned and attached to his objects by the loader.
  - Same flags as ep1, no hook.
  - Object arrays stay byte-identical.
- **Pot COM** (`ep5_attach_com/w7edit.py --com`):
  - The `white_pot` URDF `<inertial><origin>` moves from the mesh body origin (`0 0 0`) to the area-weighted shell
    centroid of the pot's own visual mesh: `(-0.008729, -0.063857, 0.048846)` m, a shift of 8.09 cm.
  - Mass and inertia are unchanged.
- **Table:** copied from the v2 control scene (`wsupport.py`).
- **Build:** `ep5_attach_com/w7build.sh`, then `w7fix.sh`. In `w7build`, the support install was refused for the
  changed bundle; `w7fix` adds the support and the COM edit.

## Training and evaluation

- **Training:**
  - Team repo `edm_training` @ `1a6bae7b`, `tools/track3/jobs/gpu_queue.sh`.
  - Overrides: `experiment=sharpa_flash_sac curriculum=tracking_100m_object2x task.motion_speed=1.0
    sim.fps=20.254119873046875 termination.wrist_position.threshold=0.3
    training.checkpoint.save_interval_environment_steps=5000192 sim.njmax=1024 sim.nconmax=512
    training.total_environment_steps=100003840`.
  - Seed 44 (ep1), 42 (ep5).
  - Everything else is the experiment default (object termination 0.1 m / 0.5 rad).
- **Evaluation:** `v2d_submission_kit/examples/run_policy_evaluation.py`.
  - ep1: final checkpoint, 100,003,840 steps, sha256 `969b9cef…`.
  - ep5: checkpoint at 95,006,720 steps, sha256 `a408e294…`. Chosen by own-reference add_auc over the checkpoints from
    70M to 100M (GT-free).
