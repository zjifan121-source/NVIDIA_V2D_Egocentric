"""Audit or fix each object's URDF inertial origin against its visual mesh's area-weighted shell centroid (Ken's ep5 fix, w7edit.py, generalised to any object).
FlashCHORD uses the URDF inertial origin as the COM. Where the mesh body origin is far from the shell centroid the object tips at frame 0.
Only the <inertial> origin changes: mass, inertia, mesh, collision and trajectories stay as they are. A backup goes to <scene>/com_backup/, the record to <scene>/SCENE_COM_FIX.json.
usage: com_fix.py <scene dir> [...] [--apply] [--threshold-cm 2.0]   (default is read-only)"""
import glob, json, os, re, shutil, sys
import numpy as np, trimesh
PAT = r'(<inertial>\s*<origin xyz=")([^"]*)(")'
args = sys.argv[1:]; apply = '--apply' in args; thr = 2.0
if '--threshold-cm' in args: thr = float(args[args.index('--threshold-cm') + 1])
scenes = [a for i, a in enumerate(args) if not a.startswith('--') and (i == 0 or args[i - 1] != '--threshold-cm')]
for sc in scenes:
    rows = []
    for u in sorted(glob.glob(f'{sc}/assets/human_motion_data/ego_recon/processed/*_rigid.urdf')):
        txt = open(u).read(); m0 = re.search(PAT, txt)
        vis = re.search(r'<visual>.*?filename="([^"]+)"', txt, re.S).group(1)
        scale = re.search(r'<visual>.*?<mesh[^>]*scale="([^"]+)"', txt, re.S)
        path = vis if os.path.isabs(vis) else os.path.normpath(os.path.join(os.path.dirname(u), vis))
        if not os.path.exists(path): path = os.path.join(os.path.dirname(u), os.path.basename(vis))
        m = trimesh.load(path, force='mesh', process=False)
        c = (m.triangles.mean(1) * m.area_faces[:, None]).sum(0) / m.area_faces.sum()
        old = np.array([float(x) for x in m0.group(2).split()])
        off = float(np.linalg.norm(c - old)) * 100; patched = False
        if apply and off > thr:
            bk = os.path.join(sc, 'com_backup'); os.makedirs(bk, exist_ok=True)
            if not os.path.exists(os.path.join(bk, os.path.basename(u))): shutil.copyfile(u, os.path.join(bk, os.path.basename(u)))
            open(u, 'w').write(re.sub(PAT, lambda q: q.group(1) + f'{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}' + q.group(3), txt, count=1)); patched = True
        rows.append(dict(body=os.path.basename(u).replace('_rigid.urdf', ''), inertial_origin_m=[round(float(x), 5) for x in old], shell_centroid_m=[round(float(x), 5) for x in c], offset_cm=round(off, 2), mesh_extent_cm=[round(float(x) * 100, 1) for x in m.extents], watertight=bool(m.is_watertight), scale=scale.group(1) if scale else None, patched=patched))
    print(os.path.basename(sc.rstrip('/')), json.dumps([{k: r[k] for k in ('body', 'offset_cm', 'mesh_extent_cm', 'scale', 'patched')} for r in rows], ensure_ascii=False), flush=True)
    if apply: json.dump(dict(rule=f'inertial origin moved to the shell centroid where the offset exceeds {thr} cm; mass and inertia unchanged', objects=rows), open(os.path.join(sc, 'SCENE_COM_FIX.json'), 'w'), indent=1)
