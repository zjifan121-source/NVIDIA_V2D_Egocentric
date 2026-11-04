"""Point the absolute file paths inside a shipped scene at the files in this copy of the scene.

usage: python relocate_scene.py <scene dir> [--dry-run]

Each absolute path is matched to this copy by its longest trailing part that exists under <scene dir>. Example:
.../w_respin_ep32/bundle_cleaned/result.npz -> <scene dir>/bundle_cleaned/result.npz. This covers string and
list-of-string columns of the processed parquet (raw_motion_file, object_mesh_paths, object_urdf_paths) and the text of
.urdf/.usda/.mtl files. Originals are kept next to each changed file as <file>.orig. Paths with no match are reported and
left alone.
"""
import glob, os, re, shutil, sys
import pyarrow as pa, pyarrow.parquet as pq


def main():
    sc = os.path.abspath(sys.argv[1]); dry = "--dry-run" in sys.argv
    changed, missing = {}, set()

    def local(v):
        parts = [p for p in v.split("/") if p]
        for i in range(len(parts)):
            cand = os.path.join(sc, *parts[i:])
            if os.path.isfile(cand):
                return cand
        return None

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, str) and v.startswith("/"):
            n = local(v)
            if n is None:
                missing.add(v); return v
            if n != v:
                changed[v] = n
            return n
        return v

    def keep(f):
        if not os.path.exists(f + ".orig"):
            shutil.copy2(f, f + ".orig")

    for f in glob.glob(f"{sc}/assets/**/*.parquet", recursive=True):
        t = pq.read_table(f); cols = []; n0 = len(changed)
        for name in t.column_names:
            col = t.column(name); ty = col.type
            if pa.types.is_string(ty) or (pa.types.is_list(ty) and pa.types.is_string(ty.value_type)):
                col = pa.array([fix(v) for v in col.to_pylist()], type=ty)
            cols.append(col)
        if len(changed) > n0 and not dry:
            keep(f); pq.write_table(pa.Table.from_arrays(cols, schema=t.schema), f)

    pat = re.compile(r"/[^\s\"'<>@()\[\]]+\.[A-Za-z0-9]+")
    for f in [p for ext in ("urdf", "usda", "mtl") for p in glob.glob(f"{sc}/assets/**/*.{ext}", recursive=True)]:
        txt = open(f).read(); new = pat.sub(lambda m: fix(m.group(0)) if local(m.group(0)) else m.group(0), txt)
        if new != txt and not dry:
            keep(f); open(f, "w").write(new)

    print(f"{'would change' if dry else 'changed'} {len(changed)} paths")
    for a, b in sorted(changed.items()):
        print(f"  {a}\n    -> {b}")
    if missing:
        print(f"{len(missing)} absolute paths with no match in the scene (left as is):")
        for m in sorted(missing)[:20]:
            print("  ", m)


if __name__ == "__main__":
    main()
