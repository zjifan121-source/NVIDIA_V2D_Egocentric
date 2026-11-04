"""Set up constant-low-LR continuations of selected runs: hard-link the source state (and replay) into a new run, write a trial manifest."""
import json, os, subprocess
from pathlib import Path
D = Path("/data/user_data/edmondwa/v2d/flash_runs/diag"); H = Path(__file__).resolve().parent
plan = json.load(open(H.parent / "submission_candidate_2026-10-28/plan.json"))
base = json.load(open(H / "ep10_nest_train_trial.json"))
EXTRA_STEPS = 25001984
for src, step, ep in [("ep3_grip_OT1M_s43", 100003840, 3), ("ep33_WIDE_s42", 95006720, 33), ("kv_ep19_OT_s43", 100003840, 19), ("kv_ep20_OT_s42", 100003840, 20), ("kv_ep5_OT_s42", 95006720, 5)]:
    name = f"ep{ep}_LOWLR_from_{src}"
    rec = json.loads(subprocess.check_output(["python3", str(H / "recipe_overrides.py"), str(D / src / f"train/policy_{step}.safetensors")]))
    train = D / name / "train"; train.mkdir(parents=True, exist_ok=True)
    for kind in ("state", "replay"):
        s = D / src / f"train/{kind}_{step}.safetensors"
        if s.exists() and not (train / s.name).exists():
            os.link(s, train / s.name)
    replay = "kept" if (train / f"replay_{step}.safetensors").exists() else "fresh"
    extra = rec["overrides"].replace("training.replay.capacity=4194304", "training.replay.capacity=1048576").replace("observation_storage_dtype=float32", "observation_storage_dtype=float16")
    extra += " training.checkpoint.save_interval_environment_steps=5000192 training.optimization.learning_rate_initial=0.00005 training.optimization.learning_rate_peak=0.00005 training.optimization.learning_rate_end=0.00005"
    t = dict(base, name=name, episode=ep, seed=rec["seed"], parquet=str(Path(rec["parquet"]).parent), scene=rec["parquet"].split("/assets/")[0],
             total_environment_steps=step + EXTRA_STEPS, baseline_episode_auc=plan[str(ep)]["tracking"], memory="64G", time="01:00:00", extra=extra,
             hypothesis=f"ep25 constant-LR 5e-5 continuation from 95M gave .827->.862. Continue {src} from {step} for 25M steps at LR 5e-5 (replay {replay}).")
    json.dump(t, open(H / f"ep{ep}_lowlr_trial.json", "w"), indent=2)
    print(name, sorted(p.name for p in train.iterdir()), t["total_environment_steps"])
