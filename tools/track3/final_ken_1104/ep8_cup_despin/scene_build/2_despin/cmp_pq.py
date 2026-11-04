"""hand5: compare the processed parquet of two scenes column by column without pandas. usage: cmp_pq.py <sceneA> <sceneB>"""
import sys, glob, os
import numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
def files(d): return sorted(glob.glob(os.path.join(d, "assets/human_motion_data/ego_recon/processed/*/robot_name=sharpa_wave/*.parquet")))
def flat(col):
    a = col.combine_chunks()
    while pa.types.is_list(a.type) or pa.types.is_large_list(a.type) or pa.types.is_fixed_size_list(a.type):
        a = pc.list_flatten(a)
    return a
for x, y in zip(files(sys.argv[1]), files(sys.argv[2])):
    ta, tb = pq.read_table(x), pq.read_table(y)
    print(os.path.basename(x), ta.num_rows, tb.num_rows, "cols", len(ta.column_names), len(tb.column_names))
    nd = 0
    for c in ta.column_names:
        if c not in tb.column_names: print("  missing", c); nd += 1; continue
        a, b = flat(ta.column(c)), flat(tb.column(c))
        if pa.types.is_floating(a.type) or pa.types.is_integer(a.type):
            xa, xb = a.to_numpy(zero_copy_only=False).astype(float), b.to_numpy(zero_copy_only=False).astype(float)
            if xa.shape != xb.shape: print("  shape", c, xa.shape, xb.shape); nd += 1; continue
            m = float(np.nanmax(np.abs(xa - xb))) if xa.size else 0.0
            if m > 1e-5: print(f"  {c}: maxabs {m:.3g}"); nd += 1
        elif a.to_pylist() != b.to_pylist():
            s = a.to_pylist(); print(f"  {c}: non-numeric differs ({str(s[:1])[:70]})"); nd += 1
    print("  columns differing:", nd, "of", len(ta.column_names))
print("CMPPQ_DONE")
