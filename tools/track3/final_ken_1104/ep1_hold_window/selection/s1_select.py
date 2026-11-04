"""ep1 refine retry (4 Nov, ~12:15 EST): seed selection on the adopted scene w_hold_ep1.
Candidates b_ep1_wh_K_s48 @100M, s50 @100M, s47 @95M vs T2 = o_ep1_hold_s44 @100M (adopt_roll.parquet).
Emulator = copy of ep_status_1104\\ep1_work\\emu_boards.py (+ _ext): identical truth construction (ref / jb / seat; GT pot =
our pot trimmed t cm rim-fixed), identical kit scorer functions (syn_lib -> in/AUC_kit.py md5 93a20d94), same meshes MA (same scene).
16 Kaggle-consistent cells = t in {0.70,1.00,1.20,1.50,2.07} x {ref,jb,seat} + t0.00_seat.
Pre-registered decision rule: notes.md, section 'RETRY' (written before this script ran).
Also: mechanism windows (s4_lidoffset, d1 push/rest, lid lift peak, pot post-return lift). Read-only on all inputs.
Writes only refine1104\\ep1\\s1_select.json (+ prints)."""
import os, sys, json, time
import numpy as np
SYN = r"C:\v2d_demo\diag\ep1_verify\synthesis\py"
sys.path.insert(0, SYN)
from syn_lib import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "s1_select.json")
WH = os.path.join(HERE, "in_wh")
t0 = time.time()
achA, refA = load_rollout(os.path.join(IN, "adopt_roll.parquet"))
achC, refC = load_rollout(os.path.join(IN, "cand_roll.parquet"))
ach100r, ref100r = load_rollout(os.path.join(IN, "o44_100003840.parquet"))
_, MA, _, _ = load_packed(os.path.join(IN, "adopt_one.parquet"))

CANDS = {"wh_s48_100M": ("b_ep1_wh_K_s48", "100003840"), "wh_s50_100M": ("b_ep1_wh_K_s50", "100003840"),
         "wh_s47_95M": ("b_ep1_wh_K_s47", "95006720")}
CONTEXT = {"wh_s48_95M": ("b_ep1_wh_K_s48", "95006720"), "wh_s50_95M": ("b_ep1_wh_K_s50", "95006720"),
           "wh_s47_100M": ("b_ep1_wh_K_s47", "100003840")}
ROLL = {"adopt_T2_100M": achA, "o44_100M_reeval": ach100r}
sanity = {}
for lab, (run, ck) in {**CANDS, **CONTEXT}.items():
    a, r = load_rollout(os.path.join(WH, run, f"early_{ck}", "results", "episode_000001.parquet"))
    d = float(np.nanmax(np.abs(r - refA)))
    sanity[lab] = dict(ref_maxabs_vs_adopt=d, ok=bool(d < 1e-5), N=int(a.shape[0]))
    print(f"sanity {lab}: N {a.shape[0]} ref max|diff| vs adopt {d:.2e}", flush=True)
    if sanity[lab]["ok"]:
        ROLL[lab] = a

# ---------------------------------------------------------------- truth construction, verbatim from emu_boards.py
TA, TC = X_to_T(refA), X_to_T(refC)
G = TC[0, 0] @ np.linalg.inv(TA[0, 0])
nz = np.linalg.inv(G)[:3, :3] @ np.array([0, 0, 1.0]); nz /= np.linalg.norm(nz)
Vw, Fw = load_obj(os.path.join(IN, "w_pot.obj"))
Vr, Fr = load_obj(os.path.join(IN, "rl_pot.obj"))
Vl, Fl = load_obj(os.path.join(IN, "w_lid.obj"))
dV = Vr - Vw
FULLTRIM = 2.07
def standin(t):
    return (Vw + (t / FULLTRIM) * dV, Fw)
Vl_c = Vl.mean(0)
_c = Vl - Vl_c; _w, _E = np.linalg.eigh(_c.T @ _c); LAX = _E[:, 0]
TRUTH_JB = np.stack([T_to_X(np.asarray(json.load(open(os.path.join(IN, "base_truth.json")))["objects"][n]["T"], float))
                     for n in NAMES], 1)
def seat(truthX, Vpot, win=(67, 249)):
    T = X_to_T(truthX).copy()
    h0 = (xf(T[0, 0], Vpot) @ nz).min()
    for f in range(win[0], win[1] + 1):
        Tl = T[f, 1]; up = Tl[:3, :3] @ LAX; up = up if up @ nz > 0 else -up
        Rl = Rot.align_vectors([nz], [up])[0].as_matrix(); cw = Tl[:3, :3] @ Vl_c + Tl[:3, 3]
        Tn = np.eye(4); Tn[:3, :3] = Rl @ Tl[:3, :3]; Tn[:3, 3] = Rl @ (Tl[:3, 3] - cw) + cw
        h = (xf(Tn, Vl) @ nz).min(); Tn[:3, 3] -= (h - h0) * nz; T[f, 1] = Tn
    return T_to_X(T)
def truths_for(t):
    return {"ref": refA, "jb": TRUTH_JB, "seat": seat(refA, standin(t)[0])}

CELLS = []
R = {}
PHASE = {}
for t in (0.0, 0.7, 1.0, 1.2, 1.5, 2.07):
    gt = GT(standin(t), (Vl, Fl))
    for tn, truth in truths_for(t).items():
        key = f"t{t:.2f}_{tn}"
        if key in ("t0.00_ref", "t0.00_jb"):
            continue
        CELLS.append(key); R[key] = {}
        for lab, a in ROLL.items():
            add, regs, al, rf = add_series(a, MA, truth, gt)
            o = other_metrics(al, rf)
            R[key][lab] = dict(auc=round(float(auc(add)), 4), **{k: (round(float(v), 4) if not isinstance(v, list) else v)
                                                                  for k, v in o.items()})
            if key == "t1.00_ref":
                PHASE[lab] = {p: {k2: round(float(v2), 3) for k2, v2 in d.items()} for p, d in phases(add).items()}
        line = " | ".join(f"{lab} {v['auc']:.3f} MP{v['maniptrans_sr']:.0f} lid{v['mean_pos_cm'][1]:.2f} rpe {v['rpe_cm']:.2f} "
                          f"mppe {v['mppe_cm']:.2f}" for lab, v in R[key].items())
        print(f"{key:<11} {line}  ({time.time() - t0:.0f}s)", flush=True)
assert len(CELLS) == 16, CELLS

# ---------------------------------------------------------------- pre-registered decision rule
def summ(lab):
    au = np.array([R[c][lab]["auc"] for c in CELLS]); rp = np.array([R[c][lab]["rpe_cm"] for c in CELLS])
    mp = np.array([R[c][lab]["mppe_cm"] for c in CELLS]); ms = np.array([R[c][lab]["maniptrans_sr"] for c in CELLS])
    lidm = np.array([3.0 - R[c][lab]["mean_pos_cm"][1] for c in CELLS])
    return au, rp, mp, ms, lidm
aT, rT, mT, sT, lT = summ("adopt_T2_100M")
DEC = {}
for lab in ROLL:
    if lab == "adopt_T2_100M":
        continue
    a, r, m, s, l = summ(lab)
    nb = int((a > aT).sum())
    c1 = nb >= 13; c2 = bool(np.median(r) <= np.median(rT) and np.median(m) <= np.median(mT)); c3 = bool((s == 1).all())
    DEC[lab] = dict(cells_auc_better=nb, dAUC_median=round(float(np.median(a - aT)), 4),
                    dAUC_range=[round(float((a - aT).min()), 4), round(float((a - aT).max()), 4)],
                    auc_median=round(float(np.median(a)), 4), auc_range=[round(float(a.min()), 4), round(float(a.max()), 4)],
                    rpe_median=round(float(np.median(r)), 3), rpe_median_T2=round(float(np.median(rT)), 3),
                    mppe_median=round(float(np.median(m)), 3), mppe_median_T2=round(float(np.median(mT)), 3),
                    rpe_cells_better=int((r < rT).sum()), mppe_cells_better=int((m < mT).sum()),
                    mpsr_pass_cells=int((s == 1).sum()), lid_margin_min=round(float(l.min()), 2),
                    lid_margin_min_T2=round(float(lT.min()), 2),
                    rule=dict(c1_auc_13of16=c1, c2_rpe_mppe_median=c2, c3_mpsr_16of16=c3,
                              decision_eligible=(lab in CANDS), passes=bool(c1 and c2 and c3)))
    print(f"DECISION {lab:<16} {DEC[lab]}", flush=True)

# ---------------------------------------------------------------- mechanism windows (not deciding)
TOP = 0.9971161200823125
Vp = Vw; cl, cp = Vl.mean(0), Vw.mean(0)
MECH = {}
for lab, a in ROLL.items():
    Ta = X_to_T(a)
    Aal = TA[0, 0] @ np.linalg.inv(Ta[0, 0])
    Tal = np.einsum("ij,nkjl->nkil", Aal, Ta)
    r = {}
    for nm, k, c, (s, e) in (("lid_table_f86_244", 1, cl, (86, 244)), ("pot_Brest_f67_99", 0, cp, (67, 99)),
                             ("pot_DE_f275_334", 0, cp, (275, 334))):
        d = np.array([(Tal[f, k, :3, :3] @ c + Tal[f, k, :3, 3]) - (TA[f, k, :3, :3] @ c + TA[f, k, :3, 3]) for f in range(s, e + 1)])
        v = d @ nz; hz = np.linalg.norm(d - v[:, None] * nz, axis=1)
        r[nm + "_side_cm"] = round(100 * float(np.median(hz)), 2); r[nm + "_normal_cm"] = round(100 * float(np.median(v)), 2)
    # d1 definitions (scene world)
    Tl_a, Tl_r = Ta[:, 1], TA[:, 1]
    za = xf(Tl_a, Vl)[..., 2].min(1); zr = xf(Tl_r, Vl)[..., 2].min(1)
    ca = xf(Tl_a, cl[None])[:, 0]; cr = xf(Tl_r, cl[None])[:, 0]
    off = np.linalg.norm(ca[:, :2] - cr[:, :2], axis=1) * 100
    r["lid_lift_peak_f45_66_cm"] = round(100 * float((za[45:67] - TOP).max()), 2)
    r["lid_lift_peak_f45_66_cm_ref"] = round(100 * float((zr[45:67] - TOP).max()), 2)
    r["twist_push_cm"] = round(float(off[92] - off[78]), 2); r["off_f88_cm"] = round(float(off[88]), 2)
    r["lid_rest_off_f100_236_cm"] = round(float(np.median(off[100:237])), 2)
    zp = xf(Ta[:, 0], Vp)[..., 2].min(1) - TOP; zpr = xf(TA[:, 0], Vp)[..., 2].min(1) - TOP
    r["pot_lift_max_f260_280_cm"] = round(100 * float(zp[260:281].max()), 2)
    r["pot_dz_med_f278_334_cm"] = round(100 * float(np.median(zp[278:335])), 2)
    r["pot_dz_med_f278_334_cm_ref"] = round(100 * float(np.median(zpr[278:335])), 2)
    r["lid_E_off_f308_334_cm"] = round(float(np.median(off[308:335])), 2)
    MECH[lab] = r
    print(f"MECH {lab:<16} " + " ".join(f"{k} {v}" for k, v in r.items()), flush=True)

json.dump(dict(cells=CELLS, sanity=sanity, emu=R, decision=DEC, phases_t1_00_ref=PHASE, mech=MECH,
               note="decision_eligible marks the 3 pre-registered candidates; others are context only"),
          open(OUTJ, "w"), indent=1)
print("wrote", OUTJ, f"({time.time() - t0:.0f}s)")
