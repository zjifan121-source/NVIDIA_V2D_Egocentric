# agg3.py : final per-episode summary. main = draws d*/t* (PYTHONPATH = Edmond flash_chord de66fea3, as our queue);
# ctrl = draws n* (no PYTHONPATH = kit venv flash_chord); pooled = both. Flag rule (task): max > README + 0.02 AND median >= README.
import glob, json, os, statistics as st
W = "/data/user_data/haisenk/v2d/push1104/taskA"
H = W + "/hf/experiments/edmond/bob1_submission_20261103"
eps = [l.split() for l in open(os.path.expanduser("~/v2d_ops/kenc/push1104/taskA/eps.txt")) if l.strip()]
def info(f):
    d = json.load(open(f)); m = d["metrics"]
    pos = m.get("object_position_error_per_body_m") or []; ori = m.get("object_orientation_error_per_body_rad") or []
    return dict(auc=round(m["add_auc"], 4), pb=[round(x, 3) for x in (m.get("add_auc_per_body") or [])],
                mp=int(bool(m.get("maniptrans_sr"))), sp=int(bool(m.get("spider_sr_uncentered"))),
                pos_cm=[round(100 * x, 2) for x in pos], ori_deg=[round(57.29578 * x, 1) for x in ori],
                mppe=round(m.get("mppe_cm", float("nan")), 3), rpe=round(m.get("relative_position_error_cm", float("nan")), 3),
                ck=d.get("checkpoint_sha256", "")[:12],
                names=[n.split("_attach_")[-1].split("_mix_")[-1] for n in (m.get("object_body_names") or [])])
def draws(ep, pat):
    out = []
    for f in sorted(glob.glob(f"{W}/draws/ep{ep}/{pat}/results/episode_{ep:06d}.json")):
        try:
            x = info(f); x["draw"] = f.split("/draws/")[1].split("/")[1]; out.append(x)
        except Exception:
            pass
    return out
def stats(dd, R):
    a = [x["auc"] for x in dd]
    if not a: return None
    b = max(dd, key=lambda x: x["auc"])
    return dict(n=len(a), min=min(a), median=round(st.median(a), 4), mean=round(st.mean(a), 4), max=max(a),
                sd=round(st.pstdev(a), 4), n_ge_readme=sum(v >= R for v in a), best_draw=b["draw"], best_gain=round(max(a) - R, 4),
                flag=bool(max(a) > R + 0.02 and st.median(a) >= R),
                mp_pass_rate=round(st.mean([x["mp"] for x in dd]), 3), sp_pass_rate=round(st.mean([x["sp"] for x in dd]), 3),
                pb_median=[round(st.median([x["pb"][i] for x in dd]), 3) for i in range(len(dd[0]["pb"]))],
                pos_cm_median=[round(st.median([x["pos_cm"][i] for x in dd]), 2) for i in range(len(dd[0]["pos_cm"]))],
                rpe_median=round(st.median([x["rpe"] for x in dd]), 3), mppe_median=round(st.median([x["mppe"] for x in dd]), 3),
                values=sorted(a, reverse=True), best=b)
res = {}
for ep, step, R in eps:
    ep = int(ep); R = float(R)
    rec = info(f"{H}/episode_{ep:06d}/policy/results/episode_{ep:06d}.json")
    main, ctrl = draws(ep, "[dt][0-9]*"), draws(ep, "n[0-9]*")
    res[ep] = dict(ep=ep, step=int(step), readme=R, recorded=rec, main=stats(main, R), ctrl=stats(ctrl, R), pooled=stats(main + ctrl, R))
json.dump(res, open(W + "/agg3.json", "w"), indent=1)
print("ep  README  | main(de66): n  min   med   max   sd   #>=R gain  FLAG | ctrl(noPP): n med max | pooled: n med max FLAG | best main draw (auc, pb, MP SP pos ori rpe mppe) | bob1 rec (pb MP SP pos ori rpe mppe) | MP%main objs")
for ep, r in res.items():
    m, c, p, rec = r["main"], r["ctrl"], r["pooled"], r["recorded"]
    if not m: print(ep, "no draws"); continue
    b = m["best"]
    cs = f"{c['n']:>2} {c['median']:.3f} {c['max']:.3f}" if c else " 0   -     -  "
    print(f"{ep:>2} {r['readme']:.4f} | {m['n']:>2} {m['min']:.3f} {m['median']:.3f} {m['max']:.3f} {m['sd']:.3f} {m['n_ge_readme']:>2} {m['best_gain']:+.3f} {'YES' if m['flag'] else 'no '}"
          f" | {cs} | {p['n']:>2} {p['median']:.3f} {p['max']:.3f} {'YES' if p['flag'] else 'no '}"
          f" | {b['draw']} {b['auc']:.4f} {b['pb']} {b['mp']} {b['sp']} {b['pos_cm']} {b['ori_deg']} {b['rpe']:.2f} {b['mppe']:.2f}"
          f" | {rec['pb']} {rec['mp']} {rec['sp']} {rec['pos_cm']} {rec['ori_deg']} {rec['rpe']:.2f} {rec['mppe']:.2f} | {m['mp_pass_rate']:.2f} {rec['names']}")
