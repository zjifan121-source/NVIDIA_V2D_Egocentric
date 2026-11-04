from pathlib import Path
import json, hashlib
W = Path(__file__).resolve().parent; S = W / 'stage'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
(S / 'run_kenf16_s203/SCENE').write_text('<local>/ep28_hold  (the scene/ directory of this release, after tools/relocate_scene.py)\n')
val = json.loads((S / 'VALIDATION.json').read_text())
curve = ', '.join(f"{k} {v['auc']:.3f} (rack {v['per_body'][0]:.2f}, pot {v['per_body'][1]:.2f})" for k, v in val['own_ref_by_checkpoint'].items())
README = f"""# ep28: observed-size pot + per-frame hold rule + centre-of-mass fix, trained with the 0.1 m / 0.5 rad object termination

Episode 28 (plastic_dish_rack, white_pot), reconstruction line, 2026-11-04. One episode only; every other
episode stays as in the team base. State: trained, evaluated, judged by the ground-truth-free checks, **not
uploaded to Kaggle** (the AUC board had no quota left on the final day).

## What is here

| Path | Content |
|---|---|
| `result.npz`, `mesh.obj`, `mesh_1.obj`, materials, `optimized_assets/` | the bundle the scene was built from (objects + camera + hands written in). Slot 0 = plastic_dish_rack (lead_v3 reconstruction), slot 1 = white_pot (the lead_v3 / alignment_mix pot, i.e. the same pot as the team base bob1 ep28; **not** the v8 donor pot). |
| `ATTACH_HOLD.json`, `HANDFIX_PREP.json`, `MESH_POSE_PAIR.json` | hand stage records: team raw hands -> loader clean -> per-frame hold rule (a hand holds the object only on frames where it does; unheld lifted frames go to the last holder; a net displacement under 1 cm is not a lift). |
| `scene/` | the FlashCHORD scene exactly as trained: `assets/human_motion_data/ego_recon/processed/` (reference parquet, meshes, URDFs with the inertial origin moved to the shell centroid: `SCENE_COM_FIX.json`), `score_input/` (meshes + reference results for packing), support seated at the lowest point minus 2 mm (`SUPPORT_V1_SEATED_INSTALLED.json`), contact and output checks. Run `tools/relocate_scene.py scene` once after download to rewrite absolute paths. |
| `run_kenf16_s203/` | the training run: `checkpoint_selection.json` / `checkpoint_evaluations.log` (own-reference AUC of every checkpoint from 70M), `policy_90005504.safetensors` (the selected checkpoint), `early_90005504/results/episode_000028.parquet` + `.json` (the recorded rollout of that checkpoint from the official evaluator), `JUDGE.md` (ground-truth-free judge lines). |
| `pack/` | six-board file = team base (bob1 + ep1 T2 + ep3 T5 + ep5 T8 + ep8 T6, sha256 c8fcd7df...) with only ep28 replaced by this rollout: `..._merged.parquet` (sha256 {val['six_board_pack_sha256'][:16]}...), the single-swap file (bob1 + ep28 only), the pack plan and its check. Not uploaded. |
| `VALIDATION.json`, `manifest.json` | hashes and the numbers below. |

## Numbers

- Own-reference AUC by checkpoint (KEN termination, seed 203): {curve}.
- Selected: 90M, **0.930** (rack 0.951, pot 0.909). Team base bob1 on this episode: 0.906 (rack 1.00, pot 0.81).
- Ground-truth-free judge at 90M (board emulator with the incumbent's reference track as the hidden-solution stand-in; camera proxy on the scored frames): trusted-frame proxy +0.012, emulator +0.026 (0.927 vs 0.901 for the incumbent on the same reference), camera +0.012; two-check passed at 70M, 75M, 85M, 90M and 100M.
- Same scene under the DEF recipe (0.3 m / 1.5 rad termination, 1M replay): three seeds, best checkpoints 0.914 / 0.899 / 0.594; the DEF recipe is seed-fragile here, the KEN termination tracked the pot from its first checkpoint.
- Expected board effect if swapped into the base: about +0.024 own-ref on one episode, i.e. about +0.001 on the AUC mean; RPE/MPPE of this episode 2.12 / 1.09 cm against 2.41 / 1.25 for bob1 (judge's own-reference numbers).

## Why this scene works where the earlier ep28 scenes did not

The new-pot scenes (v8 donor mesh, 2 cm shorter) left a 3.5 cm gap between the wrist and the pot surface, and the
policy dropped the pot at the left-hand hand-off (frames 66-72; pot 0.22). Restoring the observed-size pot with the
per-frame hold rule brings the wrist back to 0.8 cm from the surface; the policy then tracks the pot through the
hand-off. The centre-of-mass fix (pot origin 8 cm off its axis, 25 degrees of tip at rest) and the seated support are
hygiene; bob1 trained without them.

## Provenance

- Scene built on the A100 as `handspin28oldr2` (old pot + hold rule) -> `handspin28oldr2com` (+ inertial origins); the same scene byte-for-byte on the A6000 as `extra_scenes/ep28_hold_oldpot` (partition columns aside).
- Scripts: team repository, pull request #24 (`path_a/ops/experiments_20261103/lead_handspin_v1`, `lead_hand_reattach_v1`, `lead_support_seating_v1`, `lead_probe_review_v1`), commits f377f3b / f22da61 and later on `wentacc_dev`.
- Training code: FlashCHORD pipeline in the team repository at `main_buffer` de66fea3; evaluator at FlashCHORD revision 8c778f90 (the kit's).
- Judge: `lead_probe_review_v1/board_emulate.py` and `rollout_proxy.py`; ledger in `work/reports/20261103-lead-sprint/CHECKPOINT_LEDGER.md`.

See `TRAINING_SUMMARY.md` for how to train on this scene.
"""
(S / 'README.md').write_text(README)
SUMMARY = """# Training summary for the policy-training agent (ep28)

Goal: reproduce or beat own-reference AUC 0.930 on this scene and, if it beats it, produce a rollout for a
single-episode swap into the team base.

## Scene

`scene/` (after `python tools/relocate_scene.py scene`). Reference parquet:
`scene/assets/human_motion_data/ego_recon/processed/sequence_id=ep28_lead_hold/robot_name=sharpa_wave/*.parquet`.
Objects: plastic_dish_rack (slot 0), white_pot (slot 1). 220 frames at 20.254 fps.

## Recipe that produced 0.930 (run `ep28_holdold_kenf16_s203`)

```
python scripts/train_flash_sac.py experiment=sharpa_flash_sac \\
  "task.parquet='<scene>/assets/human_motion_data/ego_recon/processed'" \\
  reset.seed=203 curriculum=tracking_100m_object2x training.total_environment_steps=100003840 \\
  task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 \\
  termination.wrist_position.threshold=0.3 \\
  termination.object_position.threshold=0.1 termination.object_orientation.threshold=0.5 \\
  training.replay.capacity=4194304 training.replay.observation_storage_dtype=float16 \\
  training.checkpoint.save_interval_environment_steps=5000192 training.checkpoint.save_replay=false \\
  logging.mode=offline logging.upload_checkpoints=false
```

The two settings that matter against the DEF default are the object termination (0.1 m / 0.5 rad instead of
0.3 / 1.5) and the replay capacity (4M float16 instead of 1M). The run took about 4 h on a shared A100
(7.2k env steps/s) and uses about 30 GB of GPU memory with the 4M replay.

## What to do

1. Train 2-3 more seeds with the recipe above (any seed; 203 is taken). A 4M float32 replay (the original team
   recipe) is also fine where memory allows.
2. Evaluate every checkpoint from 70M with the official evaluator (deterministic actor, from frame 0) and keep
   the best own-reference AUC; expect the best between 85M and 100M. Seed 203 gave
   70M 0.913, 75M 0.916, 80M 0.911, 85M 0.904, 90M 0.930, 95M 0.910, 100M 0.918.
3. A checkpoint is worth a board slot only if it beats the incumbent (bob1 ep28, 0.906 own-ref; emulator
   0.901 on its own reference) on both ground-truth-free checks: the board emulator with bob1's reference and the
   camera proxy (`lead_probe_review_v1/board_emulate.py`, `rollout_proxy.py`; see `run_kenf16_s203/JUDGE.md`
   for the format). Own-reference AUC alone does not transfer across references.
4. Pack as a single-episode swap on the current team base (`pack/plan.json` is the plan used here; keep the base
   file and replace only episode 28) and upload the same file to all six boards.

## Known limits

- One KEN seed so far; the DEF recipe on the same scene gave 0.914 / 0.899 / 0.594 over three seeds.
- The rack mesh is the lead_v3 reconstruction (48.6 cm long against 47.4 cm in bob1's file), so CD-O changes
  slightly with this swap; the pot mesh is bob1's.
- No Kaggle measurement of this scene exists yet; the expected AUC effect is about +0.001 on the mean.
"""
(S / 'TRAINING_SUMMARY.md').write_text(SUMMARY)
rows = [dict(relative=str(f.relative_to(S)), sha256=sha(f), bytes=f.stat().st_size) for f in sorted(S.rglob('*')) if f.is_file()]
plan = json.loads((W / 'UPLOAD_DRY_RUN.json').read_text()); plan['files'] = rows; plan['bytes'] = sum(r['bytes'] for r in rows)
(W / 'UPLOAD_DRY_RUN.json').write_text(json.dumps(plan, indent=2) + '\n')
print('docs written; files', len(rows), 'bytes', plan['bytes'])
