"""Apply the pre-registered decision rule (decision_rule_retry.json = notes.md 'Judge plan' item 4) to the judge outputs.
Inputs: judge_family.json (adopted sc_ep8 family), judge_rf_s42/43/44.json (rf_ep8_ds sweep-best), rollouts/<run>/sweep.json,
optional judge_rf_100M_REF.json (supplementary). Writes judge_rf.json (merged rf results) and verdict_rf.json."""
import json, os
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda p: json.load(open(os.path.join(H, p)))
fam = J("judge_family.json")
rf = {}
for s in (42, 43, 44):
    rf.update(J(f"judge_rf_s{s}.json"))
json.dump(rf, open(os.path.join(H, "judge_rf.json"), "w"), indent=1)
SK = ["REF|t6cup", "REF_DS|t6cup", "REF|b1cup", "SEAT|t6cup", "RAY|t6cup"]
T6 = fam["h_ep8_sc_K_s42"]
KEN = ["h_ep8_sc_K_s42", "h_ep8_sc_K_s43", "h_ep8_sc_K_s44", "h_ep8_sc_K_s45"]
RF = ["rf_ep8_ds_K_s42", "rf_ep8_ds_K_s43", "rf_ep8_ds_K_s44"]
pot_tot = lambda r: sum(w["lost"] for w in r["win"]["pot"].values())
cupw = lambda r, w: r["win"]["cup"][w]["lost"]
out = {"rule": "decision_rule_retry.json (= notes.md Judge plan item 4, pre-registered 10:51 EST)", "checks": {}}
# a
burst = {r: rf[r]["REF|t6cup"]["win"]["cup"]["burst f62-69"]["med_cm"] for r in RF}
na = sum(v < 2.0 for v in burst.values())
out["checks"]["a_mechanism"] = {"burst_med_cm_REF": burst, "n_below_2cm": na, "T6": T6["REF|t6cup"]["win"]["cup"]["burst f62-69"]["med_cm"],
                                "pass": na >= 2}
# b
m_rf = sum(rf[r]["REF|t6cup"]["add_auc"] for r in RF) / 3
m_ken = sum(fam[r]["REF|t6cup"]["add_auc"] for r in KEN) / 4
out["checks"]["b_family"] = {"mean_rf_REF": round(m_rf, 5), "mean_KEN_REF": round(m_ken, 5), "delta": round(m_rf - m_ken, 5),
                             "threshold": round(m_ken + 0.004, 5), "pass": m_rf >= m_ken + 0.004}
# c
best = max(RF, key=lambda r: rf[r]["REF|t6cup"]["add_auc"])
c = {"chosen": best, "REF": rf[best]["REF|t6cup"]["add_auc"], "T6_REF": T6["REF|t6cup"]["add_auc"],
     "delta_REF": round(rf[best]["REF|t6cup"]["add_auc"] - T6["REF|t6cup"]["add_auc"], 5)}
okc = c["delta_REF"] >= 0.005
for sk in SK[1:]:
    d = round(rf[best][sk]["add_auc"] - T6[sk]["add_auc"], 5)
    c[f"delta_{sk}"] = d
    okc = okc and d >= -0.003
c["pass"] = bool(okc)
out["checks"]["c_candidate"] = c
# d
d = {}
okd = True
for sk in SK:
    r, t = rf[best][sk], T6[sk]
    e = {"sp": r["spider_sr"], "mp": r["maniptrans_sr"],
         "cup_end_lost": cupw(r, "blend-out+setdown+rest f176-218"), "T6_cup_end_lost": cupw(t, "blend-out+setdown+rest f176-218"),
         "cup_rest0_lost": cupw(r, "rest f0-31"), "T6_cup_rest0_lost": cupw(t, "rest f0-31"),
         "pot_tot_lost": round(pot_tot(r), 4), "T6_pot_tot_lost": round(pot_tot(t), 4)}
    e["pass"] = (e["sp"] == 1.0 and e["mp"] == 1.0 and e["cup_end_lost"] <= e["T6_cup_end_lost"] + 0.002 + 1e-9
                 and e["cup_rest0_lost"] <= e["T6_cup_rest0_lost"] + 0.002 + 1e-9 and e["pot_tot_lost"] <= e["T6_pot_tot_lost"] + 0.003 + 1e-9)
    okd = okd and e["pass"]
    d[sk] = e
d["pass"] = bool(okd)
out["checks"]["d_no_new_failure"] = d
# e
sw = J(os.path.join("rollouts", best, "sweep.json"))
late = {s["step"]: round(s["auc"], 4) for s in sw if s["step"] >= 90000000}
bst = max(s["auc"] for s in sw)
out["checks"]["e_training_stability"] = {"best": round(bst, 4), "late_90_100M": late, "pass": bst >= 0.94 and min(late.values()) >= 0.94}
out["all_pass"] = all(v["pass"] for v in out["checks"].values())
out["verdict"] = ("ADOPT-LOCALLY rf_ep8_ds (" + best + ") over T6" if out["all_pass"] else "KEEP T6") + " - nothing submitted (Ken's stop)"
# summary tables
tab = {}
for r in KEN + ["h_ep8_sc_W_s42"]:
    tab[r] = {sk: fam[r][sk]["add_auc"] for sk in SK} | {"step": fam[r]["step"], "burst_REF": fam[r]["REF|t6cup"]["win"]["cup"]["burst f62-69"]["med_cm"]}
for r in RF:
    tab[r] = {sk: rf[r][sk]["add_auc"] for sk in SK} | {"step": rf[r]["step"], "burst_REF": burst[r]}
out["table_add_auc"] = tab
for sk in SK:
    out.setdefault("family_means", {})[sk] = {"KEN4": round(sum(fam[r][sk]["add_auc"] for r in KEN) / 4, 5),
                                              "RF3": round(sum(rf[r][sk]["add_auc"] for r in RF) / 3, 5)}
for m in ("rpe_cm", "mppe_cm"):
    out.setdefault("family_means_other", {})[m] = {sk: {"KEN4": round(sum(fam[r][sk][m] for r in KEN) / 4, 4),
                                                        "RF3": round(sum(rf[r][sk][m] for r in RF) / 3, 4),
                                                        "T6": T6[sk][m]} for sk in SK}
# per-window mean lost under REF (KEN4 vs RF3)
wl = {}
for key in ("cup", "pot"):
    for w in T6["REF|t6cup"]["win"][key]:
        wl[f"{key} {w}"] = {"T6": T6["REF|t6cup"]["win"][key][w]["lost"],
                            "KEN4": round(sum(fam[r]["REF|t6cup"]["win"][key][w]["lost"] for r in KEN) / 4, 4),
                            "RF3": round(sum(rf[r]["REF|t6cup"]["win"][key][w]["lost"] for r in RF) / 3, 4),
                            "KEN4_med_cm": round(sum(fam[r]["REF|t6cup"]["win"][key][w]["med_cm"] for r in KEN) / 4, 2),
                            "RF3_med_cm": round(sum(rf[r]["REF|t6cup"]["win"][key][w]["med_cm"] for r in RF) / 3, 2)}
out["windows_REF_mean_lost"] = wl
if os.path.exists(os.path.join(H, "judge_rf_100M_REF.json")):
    s100 = J("judge_rf_100M_REF.json")
    out["supplementary_100M_REF"] = {r: {"add_auc": v["REF|t6cup"]["add_auc"], "burst": v["REF|t6cup"]["win"]["cup"]["burst f62-69"]["med_cm"]}
                                     for r, v in s100.items()}
json.dump(out, open(os.path.join(H, "verdict_rf.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
