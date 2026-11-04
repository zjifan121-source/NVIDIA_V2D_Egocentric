"""GT-free shape audit for mesh swaps, using the kit's own CD-O function BETWEEN our meshes.

CD-O is Sim(3): pose and scale are removed, so only shape counts. The kit's _object_chamfer_cm(a, b) between two of our
meshes bounds how far their CD-O can differ on the board, roughly |CD(a) - CD(b)| <= d(a, b), and each object is 1/36 of
the board. No scan and no GT is read. Reports, per object: shape distance base to lead_v3; and for spoons, which have no
public scan, the pairwise distance matrix and each spoon's mean distance to the others.
Usage: shape_audit.py BASE.parquet OUT.json"""
import sys, os, json
import numpy as np, pandas as pd
KIT = os.environ.get('LEAD_KIT', '/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit'); sys.path.insert(0, KIT)
from v2dlb.track3_pose import _decode_object_mesh as dec
from v2dlb.object_mesh import _object_chamfer_cm as cd
ROSTER = {int(k): v for k, v in json.load(open(KIT + '/data/track_3_evaluation_objects.json'))['episodes'].items()}
LEAD = '/root/v2d/lead_v3/stage/scenes/episode_{e:06d}/scene/score_input/meshes/episode_{e:06d}/{n}.obj'
BOB1 = '/root/probes/incumbents/raw/57934736/57934736.raw'

def read_obj(path):
    vs, fs = [], []
    for line in open(path):
        if line.startswith('v '): vs.append(line.split()[1:4])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            fs += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    return np.array(vs, float), np.array(fs, int)

base = pd.read_parquet(sys.argv[1]).set_index('row_id'); bob1 = pd.read_parquet(BOB1).set_index('row_id')
def mesh(frame, e, slot): return dec(frame.object_mesh_b64[f'track_3/e{e:06d}/w000/f000000/o{slot}'])
rows = []
for e, names in sorted(ROSTER.items()):
    for slot, n in enumerate(names):
        b = mesh(base, e, slot); p = LEAD.format(e=e, n=n)
        d = cd(b, read_obj(p), 1_000_003 * e + 101 * slot) if os.path.isfile(p) else None
        rows.append(dict(episode=e, object=n, slot=slot, shape_d_base_vs_lead=None if d is None else round(d, 3)))
        print(f"ep{e:<3}{n:20} d(base, lead_v3) = {rows[-1]['shape_d_base_vs_lead']} cm", flush=True)
s8, s8b = mesh(bob1, 8, 0), mesh(base, 8, 0)
cal = round(cd(s8, s8b, 1_000_003 * 8), 3)
print(f'CALIBRATION short vs long ep8 cup: d = {cal} cm; board effect was 0.0108 x 36 = 0.39 cm, so board/d = {0.39 / cal:.2f}', flush=True)
sp = [e for e, v in ROSTER.items() if 'wooden_spoon' in v]; ms = {e: mesh(base, e, ROSTER[e].index('wooden_spoon')) for e in sp}
ls = {e: read_obj(LEAD.format(e=e, n='wooden_spoon')) for e in sp}
D = np.zeros((len(sp), len(sp)))
for i, a in enumerate(sp):
    for j, b in enumerate(sp):
        if i < j: D[i, j] = D[j, i] = cd(ms[a], ms[b], 7)
print('SPOONS base mean shape distance to the other nine, cm:', flush=True)
order = np.argsort(-D.sum(1) / (len(sp) - 1))
for i in order: print(f"  ep{sp[i]:<3} {D[i].sum() / (len(sp) - 1):.3f}", flush=True)
L = np.array([[cd(ms[a], ls[b], 7) for b in sp] for a in sp])
print('SPOONS lead_v3 mean shape distance to the nine other base spoons, cm:', flush=True)
for j in np.argsort(-(L.sum(0) - np.diag(L)) / (len(sp) - 1)): print(f"  ep{sp[j]:<3} {(L[:, j].sum() - L[j, j]) / (len(sp) - 1):.3f}", flush=True)
json.dump(dict(rows=rows, calibration=dict(short_vs_long_ep8=cal, board_effect_cm=0.39), spoon_eps=sp, spoon_base_pairwise=D.round(3).tolist(), spoon_base_vs_lead=L.round(3).tolist()), open(sys.argv[2], 'w'), indent=1)
