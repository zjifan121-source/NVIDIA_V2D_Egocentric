# agg.py : per-episode distribution of own-ref add_auc over re-evaluation draws of the bob1 checkpoint
import glob, json, os, statistics as st, sys
W = "/data/user_data/haisenk/v2d/push1104/taskA"
H = W + "/hf/experiments/edmond/bob1_submission_20261103"
eps = [l.split() for l in open(os.path.expanduser("~/v2d_ops/kenc/push1104/taskA/eps.txt")) if l.strip()]
def info(f):
    d = json.load(open(f)); m = d["metrics"]
    pos = m.get("object_position_error_per_body_m") or []; ori = m.get("object_orientation_error_per_body_rad") or []
    return dict(auc=m["add_auc"], pb=[round(x, 3) for x in (m.get("add_auc_per_body") or [])],
                mp=m.get("maniptrans_sr"), sp=m.get("spider_sr_uncentered"), chord=m.get("chord_sr"),
                pos_cm=[round(100 * x, 2) for x in pos], ori_deg=[round(57.29578 * x, 1) for x in ori],
                mppe=round(m.get("mppe_cm", float("nan")), 3), rpe=round(m.get("relative_position_error_cm", float("nan")), 3),
                ck=d.get("checkpoint_sha256", "")[:10], names=m.get("object_body_names"))
out = {}
for ep, step, R in eps:
    ep = int(ep); R = float(R)
    rec = glob.glob(f"{H}/episode_{ep:06d}/policy/results/episode_{ep:06d}.json")
    recd = info(rec[0]) if rec else None
    draws = []
    for f in sorted(glob.glob(f"{W}/draws/ep{ep}/*/results/episode_{ep:06d}.json")):
        try:
            x = info(f); x["draw"] = f.split("/draws/")[1].split("/")[1]; draws.append(x)
        except Exception as e:
            pass
    a = [x["auc"] for x in draws]
    row = dict(ep=ep, readme=R, recorded=recd, n=len(a))
    if a:
        b = max(draws, key=lambda x: x["auc"])
        row.update(min=round(min(a), 4), median=round(st.median(a), 4), mean=round(st.mean(a), 4), max=round(max(a), 4),
                   sd=round(st.pstdev(a), 4), n_ge_readme=sum(v >= R for v in a), best=b,
                   flag=bool(max(a) > R + 0.02 and st.median(a) >= R),
                   mp_rate=round(st.mean([x["mp"] or 0 for x in draws]), 3), sp_rate=round(st.mean([x["sp"] or 0 for x in draws]), 3),
                   all=sorted([round(v, 3) for v in a], reverse=True))
    out[ep] = row
    if a:
        print(f"ep{ep:>2} R={R:.4f} rec={recd['auc'] if recd else float('nan'):.4f} n={len(a):>2} min={row['min']:.3f} med={row['median']:.3f} "
              f"max={row['max']:.3f} sd={row['sd']:.3f} n>=R {row['n_ge_readme']:>2} FLAG={'YES' if row['flag'] else 'no '} best={b['draw']} pb={b['pb']} "
              f"MP={b['mp']} SP={b['sp']} pos={b['pos_cm']} ori={b['ori_deg']} | rec MP={recd and recd['mp']} SP={recd and recd['sp']} pos={recd and recd['pos_cm']} ori={recd and recd['ori_deg']} | mp_rate={row['mp_rate']}")
    else:
        print(f"ep{ep:>2} R={R:.4f} rec={recd and recd['auc']} no draws yet")
json.dump(out, open(W + "/agg.json", "w"), indent=1)
