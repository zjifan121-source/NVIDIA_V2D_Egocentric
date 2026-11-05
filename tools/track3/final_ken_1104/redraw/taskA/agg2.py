# agg2.py : per-episode distribution of own-ref add_auc over re-evaluations of the bob1 checkpoint (d*/t* = PYTHONPATH de66;
# n* = control without PYTHONPATH), plus own-ref SR / RPE / MPPE of the bob1 recorded rollout and of the best draw
import glob, json, os, statistics as st
W = "/data/user_data/haisenk/v2d/push1104/taskA"
H = W + "/hf/experiments/edmond/bob1_submission_20261103"
eps = [l.split() for l in open(os.path.expanduser("~/v2d_ops/kenc/push1104/taskA/eps.txt")) if l.strip()]
def info(f):
    d = json.load(open(f)); m = d["metrics"]
    pos = m.get("object_position_error_per_body_m") or []; ori = m.get("object_orientation_error_per_body_rad") or []
    return dict(auc=round(m["add_auc"], 4), pb=[round(x, 3) for x in (m.get("add_auc_per_body") or [])],
                mp=m.get("maniptrans_sr"), sp=m.get("spider_sr_uncentered"),
                pos_cm=[round(100 * x, 2) for x in pos], ori_deg=[round(57.29578 * x, 1) for x in ori],
                mppe=round(m.get("mppe_cm", float("nan")), 3), rpe=round(m.get("relative_position_error_cm", float("nan")), 3),
                ck=d.get("checkpoint_sha256", "")[:12], names=m.get("object_body_names"))
def draws(ep, pat):
    out = []
    for f in sorted(glob.glob(f"{W}/draws/ep{ep}/{pat}/results/episode_{ep:06d}.json")):
        try:
            x = info(f); x["draw"] = f.split("/draws/")[1].split("/")[1]; out.append(x)
        except Exception:
            pass
    return out
res = {}
for ep, step, R in eps:
    ep = int(ep); R = float(R)
    recd = info(f"{H}/episode_{ep:06d}/policy/results/episode_{ep:06d}.json")
    dd = draws(ep, "[dt][0-9]*"); nn = draws(ep, "n[0-9]*")
    a = [x["auc"] for x in dd]
    row = dict(ep=ep, step=int(step), readme=R, recorded=recd, n=len(a), control_nopp=[x["auc"] for x in nn])
    if a:
        b = max(dd, key=lambda x: x["auc"])
        row.update(min=min(a), median=round(st.median(a), 4), mean=round(st.mean(a), 4), max=max(a), sd=round(st.pstdev(a), 4),
                   n_ge_readme=sum(v >= R for v in a), best=b, flag=bool(max(a) > R + 0.02 and st.median(a) >= R),
                   best_gain=round(max(a) - R, 4), mp_pass_rate=round(st.mean([1.0 if x["mp"] else 0.0 for x in dd]), 3),
                   sp_pass_rate=round(st.mean([1.0 if x["sp"] else 0.0 for x in dd]), 3),
                   pb_median=[round(st.median([x["pb"][i] for x in dd]), 3) for i in range(len(dd[0]["pb"]))],
                   all=sorted(a, reverse=True))
    res[ep] = row
json.dump(res, open(W + "/agg2.json", "w"), indent=1)
print("ep  R(readme) n   min    med    max   sd    #>=R gain   FLAG | bob1-rec MP SP pos_cm ori_deg rpe mppe | best draw MP SP pos_cm ori_deg rpe mppe | MPpass% ctrl(noPP)")
for ep, r in res.items():
    if not r["n"]:
        print(ep, "no draws"); continue
    c, b = r["recorded"], r["best"]
    print(f"{ep:>2} {r['readme']:.4f} {r['n']:>2} {r['min']:.3f} {r['median']:.3f} {r['max']:.3f} {r['sd']:.3f} {r['n_ge_readme']:>3} {r['best_gain']:+.3f} {'YES' if r['flag'] else 'no '}"
          f" | {int(c['mp'] or 0)} {int(c['sp'] or 0)} {c['pos_cm']} {c['ori_deg']} {c['rpe']:.2f} {c['mppe']:.2f}"
          f" | {b['draw']} {b['auc']:.4f} pb={b['pb']} {int(b['mp'] or 0)} {int(b['sp'] or 0)} {b['pos_cm']} {b['ori_deg']} {b['rpe']:.2f} {b['mppe']:.2f}"
          f" | {r['mp_pass_rate']:.2f} pbmed={r['pb_median']} {r['control_nopp']}")
