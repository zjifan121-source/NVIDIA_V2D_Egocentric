# Policy training guide (V2D Track 3)

Written 2026-11-03 for the final push (leaderboard freeze **2026-11-04, 5 pm EST**). Each person owns a few episodes end to end:
reconstruction → scene → policy → pick. This is where to start, what recipe to use, and what the numbers say after 440
finished training runs.

## 1. Recipe card

| Setting | Use | Why |
|---|---|---|
| Length | **100M environment steps** (fresh run) | Best checkpoint lands at 90–100M (median 95M). ~1 h on one L40S / 6000 Ada / H100 |
| Arms per scene | **DEF + WIDE, 2 seeds each = 4 runs**. On scenes Ken fixed: **KEN + WIDE** | No recipe wins everywhere (below) |
| Seeds | **At least 2 per arm, never judge a scene on 1** | Seeds usually agree within 0.04, but some crash (ep20: 0.23 / 0.73 / 0.70) |
| Checkpoint | **Evaluate every checkpoint from 70M, keep the best** (automatic) | The best isn't the last in 262 of 371 runs, and on average it's 0.06 better |
| Scene variant | **raw** (no object-rotation smoothing) | Smoothing didn't beat raw on average |
| Continuation (`next`) | Only if GPUs are idle | +0.007 on average; helped 24 of 69 runs, hurt 12 (the original is kept) |
| Re-scoring the same checkpoint | Skip | The selected score already captures the eval noise: the best of ~5 repeats came out 0.002 *lower* |
| Resources | 1 GPU, 8 CPU, 64 GB (ep10: 128 GB), 2 h limit | |

**Biggest lesson: fix the scene before tuning the recipe.** The largest single gains came from repairing the reference
motion, not from training settings. On ep14, the reference hand let go of the cup mid-air, so the policy was punished for
holding it. Locking the hand to the cup (`tools/track3/hold_hand.py`) lifted fresh seeds from **0.58 to 0.82–0.85**.
Ken's hand fixes did the same on ep3, ep10 and ep20. Before you train on a new scene, run the checks in section 4.

### The recipes (`campaign.py`, `RECIPES`)

- **DEF**: our default. Object termination 0.3 m / 1.5 rad, replay 1M float16.
- **KEN**: Ken's team recipe, read from the config stored in his 3 Nov policies. Object termination **0.1 m / 0.5 rad**, replay **4M float32**.
  On ep10_kw it scored 0.709–0.713 against 0.594 for DEF. Use it on Ken-fixed scenes, and try it wherever DEF stalls on a hard object.
  This is new: ep10 is the only head-to-head result so far.
- **WIDE**: linear action mapping, wider exploration (initial std 0.3, target sigma 0.3). Across 19 scenes it beat DEF only 7 times
  (mean −0.03), but where it wins it wins big: ep5 +0.13, ep25 +0.09, ep36 +0.20. It lost on ep19 (−0.28) and ep10 (−0.08). Keep it as the second arm.
- RELAXA (relaxed hand curriculum): mixed results, left out.

### Seeds, in numbers

From 27 groups of same-scene, same-recipe seeds: median spread 0.042, 75th percentile 0.076, max 0.331.
In groups with ≥3 seeds, the expected best of 1 / 2 / 3 seeds was 0.695 / 0.706 / 0.711. A third seed buys ~0.005, so the
second arm (WIDE or KEN) is a better use of a GPU than a third seed of the same arm.

## 2. Setup on your own account

`/data/user_data/edmondwa` isn't readable by other users, so use your own storage and venvs. The scripts default to Edmond's
paths and read these environment variables (they're exported into the Slurm job):

```bash
export V2D_REPO=$HOME/NVIDIA_V2D_Egocentric        # this repo, on main_buffer
export V2D_DATA=/data/user_data/$USER/v2d          # runs, scenes and venvs go here
export V2D_USER=$USER                              # submit_trial.py refuses to submit as anyone else
export V2D_USERS=$USER                             # whose queue counts as "still running"
export V2D_PARTITION=general V2D_QOS=normal        # your allocation; default is preempt / preempt_qos
# export V2D_SLURM_ACCOUNT=...                     # only if your partition needs --account
```

Venvs under `$V2D_DATA/venvs/`:
- `flash_chord`: training and eval. `uv sync` in `baseline/robotic_grounding/flash_chord` (see its README).
- `retarget`: scene building. `tools/track3/load_retarget_requirements.txt`.
- `kit`: packing and submission. The Kaggle kit.

Jobs stop themselves (`scontrol requeuehold`) when `$V2D_DATA` has under 30 GB free. If jobs sit in the queue doing nothing,
check `df`, run `python3 campaign.py prune NAME`, then `scontrol release <jobid>`. On 11-03, 40 jobs sat silently held for 3.5 h.

## 3. Workflow

All in `reports/policy_targeted_2026-10-28/`. Campaign state lives in `campaigns/NAME.json`.

```bash
python3 campaign.py init  myfix /path/to/bundles      # bundles/episode_000014/bundle/result.npz
python3 campaign.py build myfix --eps 14              # scene build on maxlab-cpu (~minutes)
python3 campaign.py check myfix                       # scene_qa.py on the built scenes
python3 campaign.py train myfix --eps 14 --recipe DEF,WIDE --seeds 2 --dry-run   # inspect, then drop --dry-run
python3 campaign.py status myfix                      # best tracking per episode (run@step)
python3 campaign.py next   myfix                      # optional: +25M at constant LR 5e-5 from each episode's best
python3 campaign.py prune  myfix                      # keep only the best-step learner state of finished runs
```

Runs land in `$V2D_DATA/flash_runs/diag/<name>/`. `checkpoint_evaluations.log` holds one JSON row per checkpoint
(`auc` = own-reference tracking); the best is replayed into `early_<step>/results/`, and that's what gets packed.

Other tools you'll want:
- `tools/track3/hold_hand.py`: lock a hand's object-relative grip over a frame range (the ep14 fix).
- `tools/track3/import_scene.py`: import a scene someone else built (e.g. Ken's `_kw` scenes).
- `tools/track3/hand_fixes/`: Ken's per-episode hand patches (ep3, ep10, ep20), with a README.
- `cfgdiff.py A.safetensors B.safetensors`, `recipe_overrides.py POLICY.safetensors`: diff or print the exact recipe stored in any policy file. **Diff configs before
  comparing your number with a teammate's.** Our DEF and Ken's recipe differ, and that alone was 0.12 on ep10.

## 4. Check the scene before training

```bash
$V2D_DATA/venvs/retarget/bin/python scene_qa.py SCENES_ROOT --out SCENES_ROOT/scene_qa.json
$V2D_DATA/venvs/retarget/bin/python contact_union.py SCENE_DIR ...   # lifted frames with no hand contact near the object
$V2D_DATA/venvs/retarget/bin/python ghost_motion.py PLAN.json        # object moving (>1 cm or >3 deg/frame) with no hand on it
```

Found a released grip? Fix it at the bundle level and rebuild:
`python tools/track3/hold_hand.py --npz IN/result.npz --out OUT/result.npz --side right --object blue_cup --start 81 --end 107 --grip-frames 74 78`.

The things that hold tracking down are almost always in the reference: a hand releasing an object mid-air, an object floating
or jumping, a hand passing through the object, wrong scale. Watch the per-body numbers in the eval output. If one object sits far
below the others (ep14 cup: 0.69, ep10 beige cup: 0.24 under DEF), the problem is that object's reference or grip, not the seed.

## 5. From tracking to Kaggle

Kaggle AUC ≈ own-reference tracking × a reconstruction factor. Measured on whole submissions: **Ken bundle 0.81, Wentao bundle 0.86**.
So a Wentao-based policy beats a Ken-based one only if `wentao_tracking > ken_tracking × 0.81 / 0.86` (≈ 0.94 × Ken's).
Example: on ep20, Wentao + Ken's fix tracks 0.730 against Ken's 0.904, which is 0.63 against 0.73 Kaggle, so Ken stays.

`make_bob_plan.py` applies this per episode over every finished run and writes the plan that gets packed
(`tools/track3/pack_submission.py`). When your episode is ready, send Edmond: **episode, scene path, run, step, tracking**.

Measuring an episode on Kaggle: resubmit the current best with **only your episode swapped**. The score change ×20 is your
episode's change. Changing several episodes at once (bob4) hides which one helped. We get 20 submissions a day, shared, so ask first.

## 6. Timeline for the freeze

One round = ~1 h of training once a GPU is free, plus queue time, plus ~10 min of checkpoint evals. Aim to have your final
runs finished by **~13:00 EST on 11-04**, so there's time to pack, check and submit before 5 pm.
