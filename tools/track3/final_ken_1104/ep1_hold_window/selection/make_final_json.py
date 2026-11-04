"""Assemble C:\\v2d_demo\\diag\\refine1104\\wrapup\\ep1_final.json from ep1_final_select.json (+ anchored own-ref logs, sweep bests,
pot-event rows). Read-only on inputs; writes only the wrapup ep1_final.json (new file; refuses to overwrite)."""
import os, sys, json, re, time
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "ep1_final.json")
ARCH = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
S = json.load(open(os.path.join(HERE, "ep1_final_select.json")))
# anchored / plain own-ref (anch.py, adopted reference, anchor white_pot)
OWN = {}
def read_anch(path, tag):
    for line in open(path, encoding="utf-8-sig"):
        if "\t" not in line:
            continue
        p, js = line.rstrip("\n").split("\t", 1)
        d = json.loads(js)
        m = re.search(r"(b_ep1_wh_K_s\d+)[\\/]early_(\d+)", p)
        if m:
            lab = f"wh_s{m.group(1)[-2:]}_{int(m.group(2)) // 1000000}M"
        elif p.endswith("adopt_roll.parquet"):
            lab = "adopt_T2_100M"
        elif p.endswith("o44_100003840.parquet"):
            lab = "o44_100M_reeval"
        else:
            continue
        OWN[lab] = dict(plain=d["plain"]["auc"], anchored=d["anchored"]["auc"], source=tag)
read_anch(r"C:\v2d_demo\diag\refine1104\ep1\s2_anch.log", "refine1104/ep1/s2_anch.log (morning)")
read_anch(os.path.join(HERE, "anch_s56.log"), "wrapup/ep1_final/anch_s56.log")
# sweep bests (kq sweeps, own-ref add_auc at the sweep-best checkpoint)
BEST = {}
for s in (48, 50, 51, 52, 53, 54, 55, 56):
    for base in (os.path.join(HERE, "in_wh"), r"C:\v2d_demo\diag\refine1104\ep1\in_wh"):
        p = os.path.join(base, f"b_ep1_wh_K_s{s}", "best.txt")
        if os.path.exists(p):
            BEST[f"b_ep1_wh_K_s{s}"] = open(p).read().strip(); break
# pot-event rows (d6 definitions; reported only)
POT = {}
for path in (r"C:\v2d_demo\diag\refine1104\ep1\d6_potevent.log", os.path.join(HERE, "d6_potevent.log")):
    for line in open(path):
        m = re.match(r"^(T2|wh_s48_100M|wh_s50_100M|wh_s56_100M|wh_s56_95M|wh_s56_90M)\s+(.*)$", line.strip())
        if m:
            toks = m.group(2).split()
            POT["adopt_T2_100M" if m.group(1) == "T2" else m.group(1)] = {toks[i]: float(toks[i + 1]) for i in range(0, len(toks) - 1, 2)}
DEC, H2H, FIN = S["decision_vs_T2"], S["h2h_vs_s48"], S["final"]
ORDER = ["wh_s48_100M", "wh_s50_100M", "wh_s56_100M", "wh_s56_95M", "wh_s56_90M", "o44_100M_reeval"]
RUNMAP = {"wh_s48_100M": "b_ep1_wh_K_s48@100003840", "wh_s50_100M": "b_ep1_wh_K_s50@100003840",
          "wh_s56_100M": "b_ep1_wh_K_s56@100003840", "wh_s56_95M": "b_ep1_wh_K_s56@95006720",
          "wh_s56_90M": "b_ep1_wh_K_s56@90005504", "o44_100M_reeval": "o_ep1_hold_s44@100003840 (T2's checkpoint re-evaluated)"}
table = []
for lab in ORDER:
    if lab not in DEC:
        continue
    d = DEC[lab]; h = H2H.get(lab)
    table.append(dict(
        label=lab, run_at_step=RUNMAP[lab], role="candidate" if d["decision_eligible"] else "context (not deciding)",
        auc_cells_gt_T2=f"{d['cells_auc_better']}/16", dAUC_vs_T2_median=d["dAUC_median"], dAUC_vs_T2_range=d["dAUC_range"],
        auc_median=d["auc_median"], auc_range=d["auc_range"],
        rpe_median_cm=d["rpe_median"], mppe_median_cm=d["mppe_median"], mpsr_pass_cells=f"{d['mpsr_pass_cells']}/16",
        lid_margin_min_cm=d["lid_margin_min"], lid_margin_median_cm=d["lid_margin_median"],
        passes_rule_vs_T2=d["rule"]["passes"], rule_parts=d["rule"],
        h2h_vs_s48=(None if h is None else dict(auc_cells_gt_s48=f"{h['cells_auc_better']}/16", dAUC_median=h["dAUC_median"],
                                                dAUC_range=h["dAUC_range"], rpe_median_s48=h["rpe_median_base"],
                                                mppe_median_s48=h["mppe_median_base"], passes=h["rule"]["passes"])),
        own_ref_plain=OWN.get(lab, {}).get("plain"), own_ref_anchored=OWN.get(lab, {}).get("anchored"),
        mech=S["mech"].get(lab), pot_event=POT.get(lab)))
T2 = dict(label="adopt_T2_100M", run_at_step="o_ep1_hold_s44@100003840 (T2, team base on Kaggle)",
          rpe_median_cm=DEC["wh_s48_100M"]["rpe_median_base"], mppe_median_cm=DEC["wh_s48_100M"]["mppe_median_base"],
          lid_margin_min_cm=DEC["wh_s48_100M"]["lid_margin_min_base"],
          own_ref_plain=OWN.get("adopt_T2_100M", {}).get("plain"), own_ref_anchored=OWN.get("adopt_T2_100M", {}).get("anchored"),
          mech=S["mech"].get("adopt_T2_100M"), pot_event=POT.get("adopt_T2_100M"))
aT = [S["emu"][c]["adopt_T2_100M"]["auc"] for c in S["cells"]]
T2["auc_median"] = round(sorted(aT)[7] / 2 + sorted(aT)[8] / 2, 4); T2["auc_range"] = [min(aT), max(aT)]
T2["mpsr_pass_cells"] = f"{sum(int(S['emu'][c]['adopt_T2_100M']['maniptrans_sr'] == 1) for c in S['cells'])}/16"
pick = FIN["pick"]
res = dict(
    episode=1, time=time.strftime("%Y-%m-%d %H:%M:%S local (EST)"),
    kind="FINAL local ep1 pick among seeds on the adopted scene w_hold_ep1 (local adoption only; NOTHING submitted - Kaggle stopped)",
    decision=("KEEP b_ep1_wh_K_s48 @100M (step 100003840) as the locally adopted ep1 candidate" if pick == "wh_s48_100M"
              else f"{RUNMAP.get(pick, pick)} replaces b_ep1_wh_K_s48 @100M as the locally adopted ep1 candidate"),
    pick=RUNMAP.get(pick, pick), pick_label=pick,
    candidates=dict(rule_bar="sweep-best own-ref >= 0.75 on w_hold_ep1 (kq sweeps; all s51-s56 sweeps finished 13:04 EST)",
                    eligible={"b_ep1_wh_K_s48": BEST.get("b_ep1_wh_K_s48"), "b_ep1_wh_K_s50": BEST.get("b_ep1_wh_K_s50"),
                              "b_ep1_wh_K_s56": BEST.get("b_ep1_wh_K_s56")},
                    below_bar={k: v for k, v in BEST.items() if k not in ("b_ep1_wh_K_s48", "b_ep1_wh_K_s50", "b_ep1_wh_K_s56")}),
    rule=("pre-registered (refine1104/ep1/notes.md RETRY, unchanged): qualifies vs T2 = o_ep1_hold_s44 @100M iff emulated AUC > T2 "
          "(strict) in >= 13/16 Kaggle-consistent cells, median emulated RPE <= T2's and median MPPE <= T2's, MP-SR pass in 16/16; "
          "among qualifiers the highest 16-cell median emulated AUC wins, ties (4 dp) broken by the MP-SR lid margin. "
          "Head-to-head vs s48 reported, not deciding."),
    emulator=dict(script="wrapup/ep1_final/ep1_final_select.py = s1_select.py with only the candidate list and the head-to-head "
                         "added; truth+cell block and mech block byte-identical (diff checked)",
                  basis="copy of ep_status_1104/ep1_work/emu_boards.py (+_ext); scorer syn_lib -> in/AUC_kit.py md5 93a20d94",
                  cells=S["cells"], reproducibility_vs_morning=S["repro_vs_morning"], sanity_reference_equal=S["sanity"]),
    final=FIN, per_seed_table=table, T2_baseline=T2,
    rollout_sha256={"b_ep1_wh_K_s48@100M": "b3fbfe3b1e8cbab6b2ed2fe917c0822d2f3a291dae66272e431a0f678cc155b6",
                    "b_ep1_wh_K_s50@100M": "10f9123336b222e3e47fb72dfb4f3e17bca8b7bee2b324271a661784f54c5ef4",
                    "b_ep1_wh_K_s56@100M": "88897a4c48a4d262060f876f9ac02ec66f0bb831cf3362fd22b200ecb8455901"},
    policy_sha256={"b_ep1_wh_K_s48@100M": "6aaab5d64f317cf5690f89f296114b12e5aebe33121fe950b925094ce1e51152",
                   "b_ep1_wh_K_s50@100M": "bf4e9610b6ab0a65f91e220fd18528f3271a646936758d3b539f229938fb1b85",
                   "b_ep1_wh_K_s56@100M": "85c969166a87f2ec778a951672f172e9702dd317f5b3da08fb2f8fd9f818276a"},
    archive=ARCH,
    files=dict(select_json=os.path.join(HERE, "ep1_final_select.json"), select_log=os.path.join(HERE, "ep1_final_select.log"),
               anch_log=os.path.join(HERE, "anch_s56.log"), pot_event_log=os.path.join(HERE, "d6_potevent.log"),
               pull_log=os.path.join(HERE, "pull_s56.log"),
               babel_pull="/data/user_data/haisenk/v2d/refine1104/wrapup/ep1_final/wh_new_rollouts_130739.tgz"),
    hard_rules="no Kaggle submission (none attempted), no scene/run/queue changes, no deletes/overwrites outside own output dirs")
if os.path.exists(OUT):
    sys.exit(f"refusing to overwrite existing {OUT}")
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
print(json.dumps(dict(decision=res["decision"], final=FIN), indent=1))
