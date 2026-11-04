"""hand7 ep5 (3 Nov): post-build edit of a NEW w7 scene (never used on any other scene).
--com : white_pot URDF inertial origin (= the COM FlashCHORD uses; it imports URDFs with ignore_inertial_definitions=False)
        moved from the mesh body origin to the area-weighted shell centroid of the pot's own visual mesh. The body origin of
        this pot mesh sits 8.1 cm off the bowl axis toward the handle (shell centroid is on the bowl axis, same height), so
        with COM = origin the frame-0 reference pose is not statically stable: the pot tips ~25 deg onto its handle (MuJoCo
        settle + every v2/fb11 rollout). Mass (0.3 kg) and inertia unchanged; nothing else in the scene is touched.
Backup of the edited URDF goes to <scene>/w7_backup/. Writes <scene>/W7EDIT.json.  usage: w7edit.py <scene dir> --com"""
import sys, os, re, glob, json, shutil
import numpy as np, trimesh
sc = sys.argv[1].rstrip('/'); args = sys.argv[2:]
assert os.path.basename(sc).startswith('w7_'), 'w7edit only edits w7_* scenes'
bk = os.path.join(sc, 'w7_backup'); os.makedirs(bk, exist_ok=True)
log = {'scene': sc}
if '--com' in args:
    us = glob.glob(f'{sc}/assets/human_motion_data/ego_recon/processed/*white_pot_rigid.urdf'); assert len(us) == 1, us
    u = us[0]; txt = open(u).read()
    pat = r'(<inertial>\s*<origin xyz=")([^"]*)(")'
    old = re.search(pat, txt).group(2); assert old.split() == ['0', '0', '0'], f'pot inertial origin already edited: {old}'
    if not os.path.exists(os.path.join(bk, os.path.basename(u))): shutil.copyfile(u, os.path.join(bk, os.path.basename(u)))
    vis = re.search(r'<visual>.*?filename="([^"]+)"', txt, re.S).group(1); vis = os.path.normpath(os.path.join(os.path.dirname(u), vis))
    m = trimesh.load(vis, force='mesh'); c = (m.triangles.mean(1) * m.area_faces[:, None]).sum(0) / m.area_faces.sum()
    txt2 = re.sub(pat, lambda mm: mm.group(1) + f'{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}' + mm.group(3), txt, count=1)
    open(u, 'w').write(txt2)
    log['com'] = {'urdf': u, 'mesh': vis, 'old_origin': old, 'new_origin_m': [round(float(x), 6) for x in c], 'shift_cm': round(float(np.linalg.norm(c)) * 100, 2)}
    print('COM_OK', json.dumps(log['com']))
json.dump(log, open(os.path.join(sc, 'W7EDIT.json'), 'w'), indent=1)
