# patch_ckpt.py SRC SCENE DST : copy a flash_chord checkpoint, rewriting ONLY the stored task.parquet path (metadata)
# to the matching path inside the relocated scene copy. Tensor bytes are copied unchanged.
import json, os, struct, sys
src, scene, dst = sys.argv[1:4]
raw = open(src, "rb").read()
n = struct.unpack("<Q", raw[:8])[0]
hdr = json.loads(raw[8:8 + n].decode("utf-8"))
meta = hdr["__metadata__"]
olds = set()
for k, v in meta.items():
    if '"parquet"' in v:
        try:
            olds.add(json.loads(v)["task"]["parquet"])
        except Exception:
            pass
assert len(olds) == 1, olds
old = olds.pop()
parts = [p for p in old.split("/") if p]
new = None
for i in range(len(parts)):
    cand = os.path.join(scene, *parts[i:])
    if os.path.exists(cand):
        new = cand
        break
assert new, f"no match for {old} under {scene}"
cnt = 0
for k in list(meta):
    if old in meta[k]:
        cnt += meta[k].count(old)
        meta[k] = meta[k].replace(old, new)
others = sorted({w for v in meta.values() for w in v.replace('"', ' ').split() if w.startswith(("/data/user_data/edmondwa", "/home/edmondwa", "/scratch/"))})
hb = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
hb += b" " * ((8 - len(hb) % 8) % 8)
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    f.write(raw[8 + n:])
print("patched", old, "->", new, "occurrences", cnt, "other_abs_paths", others[:5])
