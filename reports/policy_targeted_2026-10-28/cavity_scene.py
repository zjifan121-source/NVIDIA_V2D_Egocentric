"""Copy a training scene and give chosen objects finer convex collision pieces (cavities, slots).

Reference motion, visual and scoring meshes stay identical; only the copied URDFs' collision
elements change. Reports how deep the reference poses push each object's collision vertices
into the other objects' new pieces, so a nest/insert that is still blocked shows up before training.

python cavity_scene.py SRC_SCENE DST_SCENE --objects blue_cup beige_cup [--threshold 0.02] [--check-frames 104 130]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import coacd
import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation

REPO = Path('/home/edmondwa/NVIDIA_V2D_Egocentric')
PROCESSED = 'assets/human_motion_data/ego_recon/processed'


def decompose(stl, threshold):
    mesh = trimesh.load(stl, force='mesh')
    parts = coacd.run_coacd(coacd.Mesh(np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                                       np.ascontiguousarray(mesh.faces, dtype=np.int32)),
                            threshold=threshold, preprocess_mode='off', resolution=10000, mcts_iterations=150, seed=0)
    return [trimesh.Trimesh(v, f, process=True).convex_hull for v, f in parts]


def install(urdf, hulls, tag):
    tree = ET.parse(urdf)
    link = tree.getroot().find('link')
    visual, inertial = ET.tostring(link.find('visual')), ET.tostring(link.find('inertial'))
    for old in link.findall('collision'):
        link.remove(old)
    out = urdf.parent / f'{tag}_collision'
    out.mkdir(exist_ok=True)
    for j, hull in enumerate(hulls):
        name = f'{tag}_hull{j:03d}.stl'
        hull.export(out / name)
        c = ET.SubElement(link, 'collision', name=f'{tag}_hull{j:03d}')
        ET.SubElement(c, 'origin', xyz='0 0 0', rpy='0 0 0')
        ET.SubElement(ET.SubElement(c, 'geometry'), 'mesh', filename=f'./{tag}_collision/{name}', scale='1 1 1')
    assert ET.tostring(link.find('visual')) == visual and ET.tostring(link.find('inertial')) == inertial
    ET.indent(tree, space='  ')
    tree.write(urdf, encoding='utf-8', xml_declaration=True)


def penetration(table, names, hulls, frames):
    """Max depth (cm) of each object's hull vertices inside any other object's hull, at reference poses."""
    pos = np.asarray(table['object_body_position'][0].as_py())
    quat = np.asarray(table['object_body_wxyz'][0].as_py())
    rows = []
    for f in frames:
        world = {}
        for slot, name in enumerate(names):
            if name in hulls:
                R = Rotation.from_quat(quat[f, slot, [1, 2, 3, 0]]).as_matrix()
                world[name] = [h.vertices @ R.T + pos[f, slot] for h in hulls[name]], R, pos[f, slot]
        for a in world:
            for b in world:
                if a == b:
                    continue
                _, Rb, pb = world[b]
                depth = 0.0
                for verts in world[a][0]:
                    local = (verts - pb) @ Rb
                    for hull in hulls[b]:
                        signed = local @ hull.face_normals.T - np.einsum('ij,ij->i', hull.face_normals, hull.triangles[:, 0])
                        inside = (signed < 0).all(axis=1)
                        if inside.any():
                            depth = max(depth, float(-signed[inside].max(axis=1).min() * 100))
                rows.append(dict(frame=f, object=a, into=b, max_depth_cm=round(depth, 3)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('src', type=Path)
    p.add_argument('dst', type=Path)
    p.add_argument('--objects', nargs='+', required=True, help='object name suffixes, e.g. blue_cup')
    p.add_argument('--threshold', type=float, default=0.02)
    p.add_argument('--check-frames', type=int, nargs='*', default=[])
    a = p.parse_args()
    if a.dst.exists():
        raise FileExistsError(a.dst)
    shutil.copytree(a.src, a.dst)
    subprocess.run([sys.executable, str(REPO / 'tools/track3/relocate_scene.py'), str(a.dst)], check=True)
    parquet = next(a.dst.glob(f'{PROCESSED}/sequence_id=*/robot_name=sharpa_wave/*.parquet'))
    table = pq.read_table(parquet)
    names = table['object_body_names'][0].as_py()
    hulls, report = {}, dict(source=str(a.src), threshold=a.threshold, preprocess_mode='off',
                             reference_unchanged=True, scoring_meshes_unchanged=True, objects=[])
    for suffix in a.objects:
        name = next(n for n in names if n.endswith('_' + suffix))
        urdf = a.dst / PROCESSED / f'{name}_rigid.urdf'
        stl = a.dst / PROCESSED / f'{name}_visual.stl'
        hulls[name] = decompose(stl, a.threshold)
        install(urdf, hulls[name], suffix)
        report['objects'].append(dict(name=name, hulls=len(hulls[name]), urdf=str(urdf),
                                      visual_sha256=hashlib.sha256(stl.read_bytes()).hexdigest()))
        print(f'{name}: {len(hulls[name])} convex pieces', flush=True)
    if a.check_frames:
        report['reference_penetration'] = penetration(table, names, hulls, a.check_frames)
    (a.dst / 'collision_repair.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
