"""repro_compare.py A_DIR B_DIR : compare two scene dirs file by file.
Text files (.usda .urdf .obj .mtl .json .log) are compared after CRLF->LF normalisation; parquets by every non-string column
(and string columns reported separately); other files by raw bytes. Prints one line per file."""
import hashlib, os, sys
import pyarrow as pa, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
A, B = sys.argv[1], sys.argv[2]
TEXT = (".usda", ".urdf", ".obj", ".mtl", ".json", ".log")


def files(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root).replace("\\", "/")] = p
    return out


def h(p, text):
    b = open(p, "rb").read()
    if text:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.md5(b).hexdigest()


fa, fb = files(A), files(B)
for rel in sorted(set(fa) | set(fb)):
    if rel not in fa or rel not in fb:
        print(f"ONLY_IN_{'A' if rel in fa else 'B'}  {rel}")
        continue
    if rel.endswith(".parquet"):
        ta, tb = pq.read_table(fa[rel]), pq.read_table(fb[rel])
        if ta.column_names != tb.column_names:
            print(f"PARQUET_COLUMNS_DIFFER  {rel}")
            continue
        num_same, str_diff, num_diff = 0, [], []
        for c in ta.column_names:
            ca, cb = ta.column(c), tb.column(c)
            ty = ca.type
            is_str = pa.types.is_string(ty) or (pa.types.is_list(ty) and pa.types.is_string(ty.value_type))
            same = ca.equals(cb)
            if is_str:
                if not same: str_diff.append(c)
            else:
                if same: num_same += 1
                else: num_diff.append(c)
        tag = "PARQUET_NUMERIC_IDENTICAL" if not num_diff else "PARQUET_NUMERIC_DIFFER"
        print(f"{tag}  {rel}  (non-string cols identical {num_same}; differing non-string {num_diff}; differing string cols {str_diff})")
        continue
    text = rel.endswith(TEXT)
    ha, hb = h(fa[rel], text), h(fb[rel], text)
    print(f"{'SAME' if ha == hb else 'DIFF'}  {rel}  {ha}{'' if ha == hb else ' vs ' + hb}{' (text, LF-normalised)' if text else ''}")
