"""Best checkpoint per run for the wentao3 campaign (and ep14 hold runs), grouped by scene tag."""
import glob, json, os, re, collections
D = "/data/user_data/edmondwa/v2d/flash_runs/diag"
rows = collections.defaultdict(list)
for log in glob.glob(f"{D}/ep*_*/checkpoint_evaluations.log"):
    run = log.split("/")[-2]
    m = re.match(r"ep(\d+)_(KW|V2|VS|HH|WA|HD)_(DEF|WIDE)_s(\d+)$", run) or re.match(r"ep(\d+)_WHCc_\d+M_(HD)_(DEF)_s(\d+)$", run)
    if not m: continue
    r = [json.loads(l) for l in open(log) if l.startswith("{")]
    if not r: continue
    b = max(r, key=lambda x: x["auc"])
    done = os.path.exists(f"{D}/{run}/train.done")
    rows[(int(m[1]), m[2] if "WHCc" not in run else "HDc")].append((round(b["auc"], 3), m[3] if "WHCc" not in run else "cont", done))
for k in sorted(rows):
    v = sorted(rows[k], reverse=True)
    print(f"ep{k[0]:>2} {k[1]:4} best {v[0][0]:.3f}  all {[x[0] for x in v]}  done {sum(x[2] for x in v)}/{len(v)}")
