"""Size audit of every object mesh in a packed Track 3 file, by object class.

Compares each episode's mesh with the other episodes of the same object name: surface PCA extents, surface area, volume,
vertex count. Inputs are only the team's own submitted meshes and our lead_v3 meshes, so no scan and no GT is involved.
A deviation says the episodes disagree, not which one is right: the median can itself be wrong, as the beige cup was.
Reads and reports only. Nothing changes a mesh.
Usage: mesh_size_audit.py BASE.parquet OUT_DIR   (needs the kit on sys.path via LEAD_KIT)"""
import sys, os, json, re
from pathlib import Path
import numpy as np, pandas as pd
KIT = os.environ.get('LEAD_KIT', '/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit'); sys.path.insert(0, KIT)
from v2dlb.track3_pose import _decode_object_mesh as dec
BASE, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
BOB1 = '/root/probes/incumbents/raw/57934736/57934736.raw'
PLAN = json.load(open('/root/bob1_meta/experiments/edmond/bob1_submission_20261103/candidate_bob1.plan.json'))
INC = json.load(open('/root/probes/incumbents/INCUMBENTS.json'))
LEAD_V3 = '/root/v2d/lead_v3/stage/scenes/episode_{e:06d}/scene/score_input/meshes/episode_{e:06d}/{n}.obj'
ROSTER = {int(k): v for k, v in json.load(open(KIT + '/data/track_3_evaluation_objects.json'))['episodes'].items()}
OURS = {25, 28, 33, 19, 30, 16, 38}  # fork 2's list of episodes this line can act on
TH = 0.10

def metrics(v, f):
    v = np.asarray(v, float); f = np.asarray(f, int); a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    vol = abs(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6)
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1); _, cnt = np.unique(e, axis=0, return_counts=True)
    w = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1); t = (a + b + c) / 3; mu = (w[:, None] * t).sum(0) / w.sum()  # area-weighted, no sampling, so reproducible
    C = ((t - mu) * w[:, None]).T @ (t - mu) / w.sum(); q = (v - mu) @ np.linalg.eigh(C)[1]
    L = np.sort(q.max(0) - q.min(0))[::-1] * 100
    return dict(L1=round(float(L[0]), 2), L2=round(float(L[1]), 2), L3=round(float(L[2]), 2), area=round(float(area * 1e4), 1), volume=round(float(vol * 1e6), 1), verts=len(v), faces=len(f), watertight=bool((cnt == 2).all()), open_edge_frac=round(float((cnt == 1).mean()), 4))

def read_obj(path):
    vs, fs = [], []
    for line in open(path):
        if line.startswith('v '): vs.append(line.split()[1:4])
        elif line.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in line.split()[1:]]
            fs += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    return np.array(vs, float), np.array(fs, int)

def source_of(e, changed):
    inc = INC['incumbents'][str(e)]; p = PLAN[str(e)]; scene = p['scene'].rstrip('/').split('/')[-1]
    tag = {'bm_scored_attach': 'Ken bm', 'ken_c_scored_attach': 'Ken C', 'fs_noref_attach': 'Ken stereo', 'scene': 'Ken hand fix'}.get(scene, 'Wentao' if p.get('source') == 'wentao' else f'Ken {scene}')
    if 'wentao' in p['scene'] and 'Wentao' not in tag: tag = 'Wentao'
    if changed:
        return {1: 'Ken T2 on Wentao base', 5: 'Ken T3 on Wentao v2 base', 8: 'Ken T4 long cup'}.get(e, inc['source'])
    return tag + (' (unchanged from bob1)' if inc['submission'] != INC['base']['submission'] else '')

base = pd.read_parquet(BASE).set_index('row_id'); bob1 = pd.read_parquet(BOB1).set_index('row_id')
rows = []
for e, names in sorted(ROSTER.items()):
    for slot, name in enumerate(names):
        rid = f'track_3/e{e:06d}/w000/f000000/o{slot}'; pay = base.object_mesh_b64[rid]; changed = pay != bob1.object_mesh_b64[rid]
        v, f = dec(pay); m = metrics(v, f); r = dict(episode=e, object=name, owner='ours' if e in OURS else 'Ken/Edmond', source=source_of(e, changed), changed_from_bob1=bool(changed), **m)
        lv = LEAD_V3.format(e=e, n=name)
        if os.path.isfile(lv):
            vl, fl = read_obj(lv); ml = metrics(vl, fl); r['lead_v3'] = {k: ml[k] for k in ('L1', 'L2', 'L3', 'area', 'volume', 'verts')}
        rows.append(r)
print('meshes', len(rows), 'classes', len({r['object'] for r in rows}), flush=True)
KEYS = ['L1', 'L2', 'L3', 'area', 'volume']
classes = {}
for r in rows: classes.setdefault(r['object'], []).append(r)
flags = []
for name, rs in classes.items():
    med = {k: float(np.median([r[k] for r in rs])) for k in KEYS}
    for r in rs:
        r['dev'] = {k: round(r[k] / med[k] - 1, 3) if med[k] else None for k in KEYS}
        vol_ok = all(x['open_edge_frac'] == 0 for x in rs)
        r['flagged'] = [k for k in KEYS if len(rs) > 1 and abs(r['dev'][k]) > TH and (k != 'volume' or vol_ok)]
    if len(rs) > 1:
        for r in rs:
            if r['flagged']: flags.append(r)
    rs[0]['_median'] = med
SCAN = {}
for e in json.load(open('/root/v2d/NVIDIA_V2D_Egocentric/tools/track3/episodes.json'))['episodes']:
    for o in e.get('objects', []): SCAN[o['name']] = SCAN.get(o['name'], False) or bool(o.get('has_public_scan'))
for name, rs in classes.items():
    med = rs[0]['_median']; rs[0]['_public_scan'] = SCAN.get(name)
    fam = {}
    for r in rs: fam.setdefault((round(r['area']), round(r['volume'])), []).append(r['episode'])
    rs[0]['_identical'] = [v for v in fam.values() if len(v) > 1]
    for r in rs:
        lv = r.get('lead_v3')
        if lv:
            lv['dev'] = {k: round(lv[k] / med[k] - 1, 3) if med[k] else None for k in KEYS}; lv['vs_base'] = {k: round(lv[k] / r[k] - 1, 3) if r[k] else None for k in KEYS}
            lv['flagged'] = [k for k in KEYS if abs(lv['dev'][k]) > TH]
json.dump(dict(base=str(BASE), threshold=TH, classes={k: v for k, v in classes.items()}), open(OUT / 'MESH_SIZE_AUDIT.json', 'w'), indent=1)
pc = lambda x: '' if x is None else f'{x * 100:+.0f}%'
L = ['# Mesh size audit of the current base', '', f'Base file: `{BASE.name}` (bob1 + ep1 + ep5 + ep8). 36 meshes, every object of every scored episode. Generated by `lead_probe_review_v1/mesh_size_audit.py` on the A100.', '',
     'Method: same object name across episodes is compared. L1 to L3 are extents in cm along the area-weighted principal axes, largest first; for near-round objects the two smaller axes are not well defined, so read L2 and L3 of cups and pots loosely. Area is in cm2, volume in cm3. A flag means more than 10% from the median of that class on L1, L2, L3 or area, or on volume if every mesh of the class is watertight. Only our own submitted meshes and our lead_v3 meshes are used: no scan, no GT. Nothing was changed.', '',
     'Read a flag as a disagreement between episodes, not as proof of which side is wrong. The median can itself be wrong, and with two episodes both sides deviate equally.', '', '## Per object class', '']
for name in sorted(classes, key=lambda n: (-len(classes[n]), n)):
    rs = classes[name]; med = rs[0]['_median']
    same = rs[0]['_identical']; same_txt = (' Identical area and volume, so the same underlying mesh, in episodes ' + '; '.join(','.join(map(str, g)) for g in same) + '.') if same else ''
    scan_txt = ' A public scan exists for this class.' if rs[0]['_public_scan'] else ''
    L += [f'### {name} ({len(rs)} episode{"s" if len(rs) > 1 else ""})', '', f"Median: L1 {med['L1']:.1f}, L2 {med['L2']:.1f}, L3 {med['L3']:.1f} cm, area {med['area']:.0f}, volume {med['volume']:.0f}.{same_txt}{scan_txt}", '',
          '| Ep | Who acts | Source | Verts | L1 | L2 | L3 | Area | Volume | Watertight | dL1 | dL2 | dL3 | dArea | dVol | lead_v3 L1 x L2 x L3 | Flag |', '|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rs:
        d = r['dev']; lv = r.get('lead_v3'); ls = f"{lv['L1']:.1f} x {lv['L2']:.1f} x {lv['L3']:.1f}" if lv else ''
        L.append(f"| {r['episode']} | {r['owner']} | {r['source']} | {r['verts']} | {r['L1']:.1f} | {r['L2']:.1f} | {r['L3']:.1f} | {r['area']:.0f} | {r['volume']:.0f} | {'yes' if r['open_edge_frac'] == 0 else 'no'} | {pc(d['L1'])} | {pc(d['L2'])} | {pc(d['L3'])} | {pc(d['area'])} | {pc(d['volume'])} | {ls} | {','.join(r['flagged'])} |")
    L.append('')
L += ['## Our lead_v3 meshes against the class median', '', 'These are the meshes our training scenes use, so they matter the moment one of our runs replaces an incumbent. For one-episode classes the median is the base mesh itself.', '', '| Ep | Object | lead_v3 L1 x L2 x L3 | Base L1 x L2 x L3 | dL1 | dL2 | dL3 | dArea | dVol | Over 10% |', '|---:|---|---:|---:|---:|---:|---:|---:|---:|---|']
for name in sorted(classes, key=lambda n: (-len(classes[n]), n)):
    for r in classes[name]:
        lv = r.get('lead_v3')
        if lv and lv['flagged']:
            d = lv['dev']; L.append(f"| {r['episode']} | {name} | {lv['L1']:.1f} x {lv['L2']:.1f} x {lv['L3']:.1f} | {r['L1']:.1f} x {r['L2']:.1f} x {r['L3']:.1f} | {pc(d['L1'])} | {pc(d['L2'])} | {pc(d['L3'])} | {pc(d['area'])} | {pc(d['volume'])} | {','.join(lv['flagged'])} |")
(OUT / 'tables.md').write_text('\n'.join(L) + '\n')
print('flags', len(flags), flush=True)
for name in sorted(classes, key=lambda n: (-len(classes[n]), n)):
    rs = classes[name]; med = rs[0]['_median']
    print(f"{name:20} n={len(rs)} scan={rs[0]['_public_scan']} med L {med['L1']:.1f}/{med['L2']:.1f}/{med['L3']:.1f} area {med['area']:.0f} vol {med['volume']:.0f} identical {rs[0]['_identical']}", flush=True)
    for r in rs:
        lv = r.get('lead_v3'); print(f"   ep{r['episode']:<3}{r['owner'][:4]:5}{r['source'][:26]:27}v{r['verts']:<5} L {r['L1']:6.1f}{r['L2']:6.1f}{r['L3']:6.1f} A {r['area']:6.0f} V {r['volume']:7.0f} open {r['open_edge_frac']:.3f} d {pc(r['dev']['L1']):>5}{pc(r['dev']['L2']):>5}{pc(r['dev']['L3']):>5}{pc(r['dev']['area']):>5}{pc(r['dev']['volume']):>6} lv3 {('%.1f/%.1f/%.1f' % (lv['L1'], lv['L2'], lv['L3'])) if lv else '':16} {','.join(r['flagged'])} | lv3 over: {','.join(lv['flagged']) if lv else ''} {pc(lv['dev']['L1']) if lv else ''}{pc(lv['dev']['L2']) if lv else ''}{pc(lv['dev']['L3']) if lv else ''}", flush=True)
