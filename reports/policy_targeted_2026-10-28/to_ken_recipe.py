"""Turn queued DEF trials into Ken's exact team recipe (KEN): termination 0.1 m / 0.5 rad, replay 4M float32,
njmax 1024 / nconmax 512, as in his 3 Nov policies' stored config. Writes <name with KEN> trials, rewrites the w3 queue
(DEF entries replaced in place), and prints the names of submitted-but-pending DEF trials to cancel.
usage: to_ken_recipe.py PENDING_NAMES_FILE"""
import json, os, re, sys
from pathlib import Path
H = Path(__file__).resolve().parent
pending = set(Path(sys.argv[1]).read_text().split())
SUBS = [("termination.object_position.threshold=0.3", "termination.object_position.threshold=0.1"),
        ("termination.object_orientation.threshold=1.5", "termination.object_orientation.threshold=0.5"),
        ("training.replay.capacity=1048576", "training.replay.capacity=4194304"),
        ("training.replay.observation_storage_dtype=float16", "training.replay.observation_storage_dtype=float32"),
        ("sim.njmax=2048 sim.nconmax=1024", "sim.njmax=1024 sim.nconmax=512")]
q = [l.strip() for l in open(H / ".w3_queue.txt") if l.strip()]
out, cancel = [], []
for t in q:
    name = os.path.basename(t).replace("_trial.json", "")
    tr = json.loads(Path(t).read_text())
    submitted = Path(t.replace(".json", ".job.json")).exists()
    if "_DEF_" not in tr["name"] or (submitted and tr["name"] not in pending):
        out.append(t); continue          # WIDE, or a DEF run already started/finished: keep
    k = dict(tr, name=tr["name"].replace("_DEF_", "_KEN_"))
    for a, b in SUBS:
        assert a in k["extra"] or b in k["extra"] or "njmax" in a, (a, k["name"])
        k["extra"] = k["extra"].replace(a, b)
    k["hypothesis"] = k["hypothesis"].replace("(DEF)", "(KEN: Ken's exact team recipe)")
    p = H / f"{k['name'].lower()}_trial.json"
    p.write_text(json.dumps(k, indent=2)); out.append(str(p))
    if submitted: cancel.append(tr["name"])
(H / ".w3_queue.txt").write_text("\n".join(out) + "\n")
print("cancel", " ".join(cancel))
print("queue", len(out), "KEN", sum("_ken_" in x for x in out))
