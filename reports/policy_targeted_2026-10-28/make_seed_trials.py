"""Default-recipe seed trials on each episode's selected scene (edmondwa preempt), saving replay so they can be continued.

python make_seed_trials.py EP[,EP...] SEED[,SEED...]
"""
import json, glob, sys
from pathlib import Path
H = Path(__file__).resolve().parent
plan = json.load(open(H.parent / "submission_candidate_2026-10-28/plan.json"))
base = json.load(open(H / "ep10_nest_train_trial.json"))
DEFAULT = ("task.motion_speed=1.0 sim.fps=20.254119873046875 sim.njmax=1024 sim.nconmax=512 termination.wrist_position.threshold=0.3 "
           "termination.object_position.threshold=0.3 termination.object_orientation.threshold=1.5 training.checkpoint.save_interval_environment_steps=5000192 "
           "training.replay.capacity=1048576 training.replay.observation_storage_dtype=float16")
for ep in map(int, sys.argv[1].split(',')):
    sc = plan[str(ep)]["scene"]
    pq = glob.glob(sc + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave")
    assert len(pq) == 1, (ep, pq)
    for seed in map(int, sys.argv[2].split(',')):
        t = dict(base, name=f"ep{ep}_DEF_s{seed}", episode=ep, seed=seed, account="edmondwa", partition="preempt", qos="preempt_qos",
                 scene=sc, parquet=pq[0], total_environment_steps=100003840, baseline_episode_auc=plan[str(ep)]["tracking"],
                 memory="64G", time="02:00:00", extra=DEFAULT, save_replay="true",
                 hypothesis="Seed spread on one scene is large (ep25 .83 vs .65); the selected run is best of 2-3 seeds. More default-recipe seeds.")
        t.pop("slurm_account", None)
        if ep == 10:
            t["extra"] = DEFAULT.replace("sim.njmax=1024 sim.nconmax=512", "sim.njmax=2048 sim.nconmax=1024"); t["memory"] = "128G"
        json.dump(t, open(H / f"ep{ep}_def_s{seed}_trial.json", "w"), indent=2)
        print(t["name"], pq[0][-45:])
