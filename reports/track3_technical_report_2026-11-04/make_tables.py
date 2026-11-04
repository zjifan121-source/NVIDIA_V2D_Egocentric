#!/usr/bin/env python3
"""Regenerate table_episodes.tex and table_progression.tex from the lead-sprint incumbent ledger.

usage: python3 make_tables.py /path/to/INCUMBENTS.md
The incumbent ledger is written by lead_probe_review_v1/board_watch.py from the Kaggle boards.
"""
import sys

ROSTER = {1: ["white_pot", "white_pot_lid"], 3: ["white_pot", "wooden_spoon"], 5: ["white_pot", "wooden_spoon"],
          6: ["white_pot", "wooden_spoon"], 8: ["beige_cup", "white_pot"], 10: ["blue_cup", "beige_cup"],
          14: ["blue_cup", "plastic_dish_rack"], 15: ["plastic_dish_rack", "wooden_spoon"],
          16: ["plastic_dish_rack", "wooden_spoon"], 19: ["mini_sweeper", "mini_dust_pan"],
          20: ["blue_cup", "water_pitcher"], 25: ["wooden_piece_1", "wooden_piece_2"],
          28: ["plastic_dish_rack", "white_pot"], 30: ["mini_sweeper", "plastic_dish_rack"],
          32: ["blue_cup", "wooden_spoon"], 33: ["blue_cup", "wooden_spoon"], 35: ["wooden_spoon"],
          36: ["wooden_spoon"], 37: ["wooden_spoon"], 38: ["mini_dust_pan"]}
PROGRESSION = [("bob1 (team base, all 20 from the policy line)", "57934736", "0.75166", "bob1"),
               ("+ ep1 T2, ep5 T3 (hand-fix line, our objects)", "57972914", "0.75682", "0.75681"),
               ("+ ep8 T4", "57973212", "0.75918", "0.75917"),
               ("+ ep3 T5", "57975024", "0.76057", "0.76056"),
               ("ep8 T4 $\\to$ T6", "57977436", "0.76161", "0.76159"),
               ("ep5 T3 $\\to$ T8 (final)", "57978563", "0.76526", "0.76524")]


def tex(s):
    return s.replace("_", "\\_")


def main(path):
    rows, inside = [], False
    for ln in open(path, encoding="utf-8").read().splitlines():
        if ln.startswith("## Incumbents"):
            inside = True
            continue
        if inside and ln.startswith("## "):
            break
        if inside and ln.startswith("|") and not ln.startswith("| Ep") and not ln.startswith("|---"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            rows.append(dict(ep=int(c[0]), source=c[1], own_ref=c[3], delta=c[4],
                             ckpt=c[7].strip("`"), rows=c[8].strip("`")))
    out = ["\\begin{tabular}{@{}rllrrll@{}}", "\\toprule",
           "Ep & Objects & Source & Own-ref & $\\Delta$AUC$\\times$20 & Checkpoint & Rows \\\\", "\\midrule"]
    for r in rows:
        src = "swap " + r["source"].split()[0] if r["source"].startswith("T") else "base"
        out.append(f"{r['ep']} & {tex(', '.join(ROSTER[r['ep']]))} & {src} & {r['own_ref']} & {r['delta']} & "
                   f"\\texttt{{{r['ckpt'][:8]}}} & \\texttt{{{r['rows'][:8]}}} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    open("table_episodes.tex", "w").write("\n".join(out) + "\n")
    t = ["\\begin{tabular}{@{}llrr@{}}", "\\toprule", "File & Kaggle ref & AUC & Predicted \\\\", "\\midrule"]
    t += [f"{a} & {b} & {c} & {d} \\\\" for a, b, c, d in PROGRESSION]
    t += ["\\bottomrule", "\\end{tabular}"]
    open("table_progression.tex", "w").write("\n".join(t) + "\n")
    print(f"{len(rows)} episodes written")


if __name__ == "__main__":
    main(sys.argv[1])
