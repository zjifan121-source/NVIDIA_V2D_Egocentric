"""Second-stage low-LR continuations: link a source state plus a (possibly older) replay of the same run, write trials."""
import json, os, shutil, subprocess
from pathlib import Path
D = Path("/data/user_data/edmondwa/v2d/flash_runs/diag"); H = Path(__file__).resolve().parent
plan = json.load(open(H.parent / "submission_candidate_2026-10-28/plan.json"))
base = json.load(open(H / "ep10_nest_train_trial.json"))
EXTRA_STEPS = 25001984
# (name, source run, state step, replay source run, replay step, episode, lr)
SPECS = [("ep5_DEF46_LOWLR", "ep5_DEF_s46", 95006720, "ep5_DEF_s46", 95006720, 5, 5e-5)]
for name, src, step, rsrc, rstep, ep, lr in SPECS:
    rec = json.loads(subprocess.check_output(["python3", str(H / "recipe_overrides.py"), str(D / src / f"train/policy_{step}.safetensors")]))
    train = D / name / "train"; train.mkdir(parents=True, exist_ok=True)
    for s, d in [(D / src / f"train/state_{step}.safetensors", train / f"state_{step}.safetensors"),
                 (D / rsrc / f"train/replay_{rstep}.safetensors", train / f"replay_{step}.safetensors")]:
        if not d.exists():
            try:
                os.link(s, d)
            except PermissionError:  # the other account owns it; hard links to foreign files are refused
                shutil.copy2(s, d)
    extra = rec["overrides"] + (" training.checkpoint.save_interval_environment_steps=5000192 "
                                f"training.optimization.learning_rate_initial={lr} training.optimization.learning_rate_peak={lr} training.optimization.learning_rate_end={lr}")
    if ep == 25:
        extra = extra.replace("sim.njmax=1024 sim.nconmax=512", "sim.njmax=2048 sim.nconmax=1024")
    t = dict(base, name=name, episode=ep, seed=rec["seed"], parquet=str(Path(rec["parquet"]).parent), scene=rec["parquet"].split("/assets/")[0],
             total_environment_steps=step + EXTRA_STEPS, baseline_episode_auc=plan[str(ep)]["tracking"], memory="64G" if ep != 25 else "128G",
             time="01:00:00", extra=extra,
             hypothesis=f"Second low-LR stage: continue {src}@{step} for 25M at LR {lr}, replay from {rsrc}@{rstep}. First stage gained .08-.09 with replay.")
    json.dump(t, open(H / f"{name.lower()}_trial.json", "w"), indent=2)
    print(name, sorted(p.name for p in train.iterdir()), t["total_environment_steps"], rec["parquet"][-60:])
