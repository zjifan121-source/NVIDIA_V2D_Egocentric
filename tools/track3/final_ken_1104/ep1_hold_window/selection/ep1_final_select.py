"""ep1 FINAL pick (4 Nov ~13:12 EST). Emulator + rule = EXACT copy of refine1104\\ep1\\s1_select.py (this morning), which is a copy
of ep_status_1104\\ep1_work\\emu_boards.py (+ _ext): identical truth construction (ref / jb / seat; GT pot = our pot trimmed t cm,
rim-fixed), identical kit scorer functions (syn_lib -> in/AUC_kit.py), same meshes MA (adopted scene w_hold_ep1).
16 Kaggle-consistent cells = t in {0.70,1.00,1.20,1.50,2.07} x {ref,jb,seat} + t0.00_seat.

Candidates (pre-registered BEFORE scoring, 13:12 EST): current local adoptee b_ep1_wh_K_s48 @100M + every other new seed on
w_hold_ep1 whose sweep-best own-ref >= 0.75: s50 @100M (0.8043), s56 @100M (0.7956). s51 (0.7149), s54 (0.3284), s52 (0.1943),
s53 (0.2895), s55 (0.6966) are below the bar (all sweeps finished 13:04 EST; kq_stat_b.sh 13:06).
Rule (notes.md RETRY, unchanged): a seed qualifies vs T2 (o_ep1_hold_s44 @100M, adopt_roll.parquet) iff
  c1 emulated AUC > T2 (strict) in >= 13/16 cells; c2 median RPE <= T2's and median MPPE <= T2's (16 cells); c3 MP-SR = 1 in 16/16.
Among qualifiers: highest 16-cell median emulated AUC; ties (equal at 4 dp) broken by the larger MP-SR lid margin (min over cells
of 3 cm - lid mean position error). Head-to-head vs s48 (same c1-c3 with s48 as the baseline) is REPORTED, not deciding; a
conflict (pick != s48 but the pick does not also pass vs s48) is flagged in the output.
Context only (not candidates): s56 @95M, s56 @90M (neighbouring checkpoints, seed robustness), o44 @100M re-eval (noise).
Read-only on all inputs. Writes only wrapup\\ep1_final\\ep1_final_select.json (+ prints)."""
import os, sys, json, time
import numpy as np
SYN = r"C:\v2d_demo\diag\ep1_verify\synthesis\py"
sys.path.insert(0, SYN)
from syn_lib import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
OUTJ = os.path.join(HERE, "ep1_final_select.json")
WH0 = r"C:\v2d_demo\diag\refine1104\ep1\in_wh"      # this morning's pulls (s48, s50; sha256 re-verified vs Babel 13:07)
WH1 = os.path.join(HERE, "in_wh")                   # 13:07 pull (s56)
MORNING = r"C:\v2d_demo\diag\refine1104\ep1\s1_select.json"
t0 = time.time()
achA, refA = load_rollout(os.path.join(IN, "adopt_roll.parquet"))
achC, refC = load_rollout(os.path.join(IN, "cand_roll.parquet"))
ach100r, ref100r = load_rollout(os.path.join(IN, "o44_100003840.parquet"))
_, MA, _, _ = load_packed(os.path.join(IN, "adopt_one.parquet"))

CANDS = {"wh_s48_100M": (WH0, "b_ep1_wh_K_s48", "100003840"), "wh_s50_100M": (WH0, "b_ep1_wh_K_s50", "100003840"),
         "wh_s56_100M": (WH1, "b_ep1_wh_K_s56", "100003840")}
CONTEXT = {"wh_s56_95M": (WH1, "b_ep1_wh_K_s56", "95006720"), "wh_s56_90M": (WH1, "b_ep1_wh_K_s56", "90005504")}
ROLL = {"adopt_T2_100M": achA, "o44_100M_reeval": ach100r}
sanity = {}
for lab, (wh, run, ck) in {**CANDS, **CONTEXT}.items():
    a, r = load_rollout(os.path.join(wh, run, f"early_{ck}", "results", "episode_000001.parquet"))
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

# ---------------------------------------------------------------- reproducibility check vs this morning's run (same emulator)
repro = {}
try:
    M = json.load(open(MORNING))
    for lab in ("adopt_T2_100M", "o44_100M_reeval", "wh_s48_100M", "wh_s50_100M"):
        diffs = []
        for c in CELLS:
            for k in ("auc", "rpe_cm", "mppe_cm", "maniptrans_sr"):
                diffs.append(abs(float(M["emu"][c][lab][k]) - float(R[c][lab][k])))
        repro[lab] = dict(max_abs_diff_vs_morning=round(max(diffs), 6), identical=bool(max(diffs) == 0.0))
except Exception as e:  # noqa: BLE001
    repro["error"] = repr(e)
print("REPRO", repro, flush=True)

# ---------------------------------------------------------------- pre-registered decision rule (unchanged) + reported head-to-head
def summ(lab):
    au = np.array([R[c][lab]["auc"] for c in CELLS]); rp = np.array([R[c][lab]["rpe_cm"] for c in CELLS])
    mp = np.array([R[c][lab]["mppe_cm"] for c in CELLS]); ms = np.array([R[c][lab]["maniptrans_sr"] for c in CELLS])
    lidm = np.array([3.0 - R[c][lab]["mean_pos_cm"][1] for c in CELLS])
    return au, rp, mp, ms, lidm
def compare(lab, base):
    aT, rT, mT, sT, lT = summ(base)
    a, r, m, s, l = summ(lab)
    nb = int((a > aT).sum())
    c1 = nb >= 13; c2 = bool(np.median(r) <= np.median(rT) and np.median(m) <= np.median(mT)); c3 = bool((s == 1).all())
    return dict(baseline=base, cells_auc_better=nb, dAUC_median=round(float(np.median(a - aT)), 4),
                dAUC_range=[round(float((a - aT).min()), 4), round(float((a - aT).max()), 4)],
                auc_median=round(float(np.median(a)), 4), auc_range=[round(float(a.min()), 4), round(float(a.max()), 4)],
                rpe_median=round(float(np.median(r)), 3), rpe_median_base=round(float(np.median(rT)), 3),
                mppe_median=round(float(np.median(m)), 3), mppe_median_base=round(float(np.median(mT)), 3),
                rpe_cells_better=int((r < rT).sum()), mppe_cells_better=int((m < mT).sum()),
                mpsr_pass_cells=int((s == 1).sum()), lid_margin_min=round(float(l.min()), 2),
                lid_margin_median=round(float(np.median(l)), 2), lid_margin_min_base=round(float(lT.min()), 2),
                rule=dict(c1_auc_13of16=c1, c2_rpe_mppe_median=c2, c3_mpsr_16of16=c3, passes=bool(c1 and c2 and c3)))
DEC, H2H = {}, {}
for lab in ROLL:
    if lab == "adopt_T2_100M":
        continue
    DEC[lab] = compare(lab, "adopt_T2_100M"); DEC[lab]["decision_eligible"] = lab in CANDS
    print(f"DECISION vsT2 {lab:<16} {DEC[lab]}", flush=True)
    if lab != "wh_s48_100M":
        H2H[lab] = compare(lab, "wh_s48_100M")
        print(f"H2H  vs s48 {lab:<16} {H2H[lab]}", flush=True)
qual = [lab for lab in CANDS if lab in DEC and DEC[lab]["rule"]["passes"]]
order = sorted(qual, key=lambda L: (DEC[L]["auc_median"], DEC[L]["lid_margin_min"]), reverse=True)
pick = order[0] if order else "adopt_T2_100M (no qualifier: T2 stays)"
tie = bool(len(order) > 1 and DEC[order[0]]["auc_median"] == DEC[order[1]]["auc_median"])
conflict = bool(order and pick != "wh_s48_100M" and not H2H[pick]["rule"]["passes"])
FINAL = dict(qualifiers=qual, ranking_by_median_auc=[(L, DEC[L]["auc_median"], DEC[L]["lid_margin_min"]) for L in order],
             pick=pick, tie_break_used=tie, conflict_with_h2h_vs_s48=conflict)
print("FINAL", FINAL, flush=True)

# ---------------------------------------------------------------- mechanism windows (not deciding), verbatim
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

json.dump(dict(cells=CELLS, sanity=sanity, repro_vs_morning=repro, emu=R, decision_vs_T2=DEC, h2h_vs_s48=H2H, final=FINAL,
               phases_t1_00_ref=PHASE, mech=MECH,
               note="decision_eligible marks the 3 pre-registered candidates; s56 @95M/@90M and o44 re-eval are context only"),
          open(OUTJ, "w"), indent=1)
print("wrote", OUTJ, f"({time.time() - t0:.0f}s)")
