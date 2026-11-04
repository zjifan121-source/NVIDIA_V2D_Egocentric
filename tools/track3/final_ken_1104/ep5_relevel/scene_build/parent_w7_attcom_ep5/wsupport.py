"""hand5w: give a rawhands scene the SAME support (table) as its as-delivered control scene. Valid only if the two scenes'
reference object poses agree at frame 0 (the support transform = parquet pose[0,0] @ inv(bundle pose[0,0]) of object 0,
and our hand-only builds never change object 0's frame-0 pose). Checks that, then copies the control's installed
support USD. usage: wsupport.py <control scene dir> <control seq> <scene dir> <seq>"""
import glob, shutil, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
cdir, cseq, sdir, seq = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
def first(d, s):
    f = glob.glob(str(d / f"assets/human_motion_data/ego_recon/processed/sequence_id={s}/robot_name=sharpa_wave/*.parquet"))[0]
    t = pq.read_table(f)
    p = np.array(t.column("object_body_position").combine_chunks().to_pylist()[0], dtype=float)
    q = np.array(t.column("object_body_wxyz").combine_chunks().to_pylist()[0], dtype=float)
    return p, q
pc, qc = first(cdir, cseq); ps, qs = first(sdir, seq)
dp = float(np.abs(pc[0] - ps[0]).max()); dq = float(np.abs(qc[0] - qs[0]).max())
dall = float(np.abs(pc - ps).max())
print(f"frame-0 object pose diff: pos {dp:.2e} quat {dq:.2e}; all-frame pos diff {dall:.3e}")
if dp > 1e-6 or dq > 1e-6: sys.exit("frame-0 object poses differ: refusing to copy the support")
st = "assets/human_motion_data/ego_recon/reconstructed_stage"
src = cdir / st / f"{cseq}_sharpa_wave_support.usda"; dst = sdir / st / f"{seq}_sharpa_wave_support.usda"
if not src.exists(): sys.exit(f"no control support {src}")
if dst.exists(): dst.rename(dst.with_name(dst.name + ".generated_by_loader"))
shutil.copyfile(src, dst); (sdir / "SUPPORT_V1_COPIED_FROM_CONTROL").write_text(str(src) + "\n")
print("SUPPORT_COPIED", dst)
