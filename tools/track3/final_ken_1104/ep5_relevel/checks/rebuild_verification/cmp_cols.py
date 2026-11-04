import glob, sys
import numpy as np, pyarrow.parquet as pq

a = sorted(glob.glob(sys.argv[1] + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet"))[-1]
b = sorted(glob.glob(sys.argv[2] + "/assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=*/*.parquet"))[-1]
ta, tb = pq.read_table(a), pq.read_table(b)
for c in ("mano_center_idx", "mano_to_robot_scale", "raw_motion_file", "object_mesh_paths", "object_urdf_paths"):
    if c in ta.column_names:
        va, vb = ta.column(c).to_pylist(), tb.column(c).to_pylist()
        print(c, "A:", str(va)[:300])
        print(c, "B:", str(vb)[:300])
        try:
            xa, xb = np.array(va, float), np.array(vb, float)
            print("  equal incl. NaN:", bool(np.array_equal(xa, xb, equal_nan=True)))
        except Exception as e:
            print("  non-numeric", type(e).__name__)
# all numeric columns, NaN-aware equality
bad = []
for c in ta.column_names:
    try:
        xa, xb = np.array(ta.column(c).to_pylist(), float), np.array(tb.column(c).to_pylist(), float)
    except Exception:
        continue
    if not np.array_equal(xa, xb, equal_nan=True):
        bad.append((c, float(np.nanmax(np.abs(xa - xb)))))
print("numeric columns not bit-equal (NaN-aware):", bad)
print("n columns", len(ta.column_names))
