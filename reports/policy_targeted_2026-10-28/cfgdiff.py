"""Diff the resolved training configs stored in two policy .safetensors files. usage: cfgdiff.py A B"""
import json, sys
from safetensors import safe_open
def cfg(p):
    with safe_open(p, "np") as f: return json.loads(f.metadata()["user.resolved_config_json"])
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, pre + k + "."))
        else: out[pre + k] = v
    return out
a, b = flat(cfg(sys.argv[1])), flat(cfg(sys.argv[2]))
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k) and not any(s in k for s in ("output", "dir", "path", "wandb", "name", "seed")):
        print(f"{k}: A={str(a.get(k))[:90]} | B={str(b.get(k))[:90]}")
