"""Priority order for the wentao3 campaign: reorder the queue file and print `jobid nice` for our pending jobs.
Order: seed index, then scene priority (Ken fixes on Wentao first), then KEN before WIDE."""
import os, re, subprocess, sys
H = "/home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28"
PRI = ["ep14_kw", "ep3_kw", "ep10_kw", "ep1_kw", "ep32_kw", "ep6_kw", "ep20_kw", "ep32_tr", "ep33_tr", "ep33_v2", "ep32_v2",
       "ep1_wa", "ep6_wa", "ep20_wa", "ep25_wa", "ep5_vs", "ep25_vs", "ep8_vs", "ep19_vs", "ep36_vs", "ep35_vs", "ep28_vs",
       "ep30_vs", "ep37_hh"]
def rank(name):
    n = name.lower(); s = int(re.search(r"_s(\d+)", n)[1]) - 600
    p = next((i for i, x in enumerate(PRI) if n.startswith(x + "_")), len(PRI))
    return s * 1000 + p * 10 + (0 if "_ken_" in n else 1)
q = [l.strip() for l in open(f"{H}/.w3_queue.txt") if l.strip()]
q.sort(key=lambda t: rank(os.path.basename(t)))
open(f"{H}/.w3_queue.txt", "w").write("\n".join(q) + "\n")
for line in sys.stdin:
    jid, name = line.split()
    if re.match(r"ep\d+_(KW|V2|VS|HH|WA|TR)_(KEN|WIDE|DEF)_s6\d\d$", name):
        r = rank(name); print(jid, min(r // 5, 2000))
