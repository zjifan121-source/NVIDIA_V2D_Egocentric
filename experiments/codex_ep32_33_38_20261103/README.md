# EP32 / EP33 / EP38 policy experiment

This workflow trains policies on the latest Wentao-v2 EP32/33 scenes and replays
the existing Ken-C EP38 policy as a control. It uses the existing FlashCHORD
runtime and the official explicit-reset policy evaluator, with no rollout helper.
Training uses seed 42, the tracking_100m_object2x curriculum, and 100,003,840
environment steps. Replay-buffer checkpoint storage is disabled.

`prepare_scenes.py` relocates scene dependency paths and checks scene closure.
`run_baselines.py` runs bounded, sequential remote training and full-sequence
evaluation. Check its dated deadline before reuse. No inference runs on a client
machine. Inputs and results are under `/data/v2d` on the attached remote volume.

`pack_candidate.py` replaces only selected EP32/33/38 rows in an immutable
historical full20 PathB control. It reads achieved policy trajectories through
the official reader, verifies evaluator hashes, and uses matching meshes from
the exact training scene. Reconstructed object names are mapped to the official
roster by the validated simulator body order. Unaffected episodes retain their
poses, mesh payloads, and checkpoint hashes. Global producing-commit and packing
provenance describe this replacement operation. The official uploader validates
the resulting full20 pack before submission.

Example replacement manifest:

```json
{
  "38": {
    "rollout": "/data/v2d/runs/ep38_control_replay/results/episode_000038.parquet",
    "scene_parquet": "/absolute/path/to/the/exact/training_scene.parquet"
  }
}
```

Run the packer with the submission environment, a public commit URL containing
this code, `--manifest`, and `--out`. Never replace evaluator hashes to make
incompatible runs appear compatible. No reference trajectories are submitted as
achieved policy trajectories.

Self-reference tracking metrics measure execution fidelity, not confidential-GT
performance. The historical PathB comparison control includes policies trained
on still references; a high tracking score does not establish manipulation or
official leaderboard accuracy. Report Kaggle scores separately. Input snapshots,
training receipts, checkpoint selections, packing audits and official submission
IDs belong in the private experiment archive, never alongside credentials.

## Current spin-corrected scenes

The subsequent run uses the team's Wentao v3-r1 EP32/33 scenes, pinned to HF
revision `27578b2ae0249cdf8d8c830c6841b8bd0aaa501f`. These correct cup axial
rotation while retaining mesh, axis and positional geometry. `prepare_v3_scenes.py`
imports isolated copies and validates dependency closure; its explicit
`--repair-incomplete` option only repairs an unvalidated copy. It changes the
upstream raw-bundle absolute path to the accompanying bundle without changing
numerical trajectories. `run_v3_spin.py` waits for the original baseline queue,
then runs EP32 and EP33 sequentially with the same KEN-style recipe and seed42.
Neither runner submits to Kaggle.

Use a current team comparison base, not the historical PathB control, for future
candidate decisions. The packer accepts `--base PATH --base-sha256 SHA256`; both
must match an independently validated immutable base. The offline bob1 re-pack
from its published real rollouts and matching meshes has 9,120 rows and SHA256
`58969e96a90005501eb8db1376841a8018741789cc32a2e42ff88e0fb786c899`.
It has not been separately submitted or officially rescored. Teammate-reported
official scores belong to the original bob1 submission.

Kaggle quota is shared. Do not use submissions as an exploratory oracle, upload
every checkpoint, or resubmit unchanged meshes. A gain against a failed tracking
baseline is insufficient to claim an improvement over the team's official best.
