"""Write WIDE-recipe trial manifests for the current selected scene of each listed episode (run on a compute node)."""
import json, glob
w = json.load(open("ep30_wide_trial.json")); plan = json.load(open("../submission_candidate_2026-10-28/plan.json"))
jobs = [(19, 42, "sarveshp"), (10, 42, "sarveshp"), (20, 42, "sarveshp"), (1, 42, "edmondwa"), (14, 42, "edmondwa"), (3, 42, "edmondwa"), (16, 42, "edmondwa"), (19, 43, "edmondwa")]
for ep, seed, acct in jobs:
    sc = plan[str(ep)]["scene"]; t = dict(w)
    pq = glob.glob(sc + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave")
    assert len(pq) == 1, (ep, pq)
    t.update(name=f"ep{ep}_WIDE_s{seed}", episode=ep, seed=seed, account=acct, scene=sc, parquet=pq[0], launcher="fresh.sbatch",
             baseline_episode_auc=plan[str(ep)]["tracking"], cpus=8, memory="64G", time="02:00:00",
             hypothesis="ep30 WIDE (linear mapping, std .3, log_std_max log .6, target sigma .3) lifted sweeper .32->.85 (episode .645->.909) on an unchanged scene; test the same recipe here.")
    for k in ("baseline_sweeper_auc", "baseline_rack_auc"):
        t.pop(k, None)
    if ep in (1, 10):
        t["extra"] = t["extra"].replace("sim.njmax=1024 sim.nconmax=512", "sim.njmax=2048 sim.nconmax=1024"); t["memory"] = "128G"
    if acct == "sarveshp":
        t.update(partition="general", qos="normal", slurm_account="msimchow")
    else:
        t.update(partition="preempt", qos="preempt_qos"); t.pop("slurm_account", None)
    json.dump(t, open(f"ep{ep}_wide_s{seed}_trial.json", "w"), indent=2); print(ep, seed, acct, pq[0][-50:])
