"""Copy a training scene and change the physics of its objects (mass, inertia, collision thickness).

Every scene object gets mass 0.3 kg and inertia diag(0.001) at the mesh origin from the URDF generator, whatever
the object is, and collides with its full SAM 3D mesh (convex-decomposed by FlashCHORD). This writes a copy with:
  --mass CLASS=KG   mass for objects whose name contains CLASS (e.g. wooden_spoon=0.06); the inertia is then
                    computed from the mesh (uniform density, centre of mass as the inertial origin) when the
                    mesh is watertight, else the old diagonal scaled by the mass ratio
  --thin CLASS=F    collision mesh only (the visual / scored mesh is unchanged): scale the object's thinnest
                    principal axis by F about its lowest point at frame 0, so it still rests on the table
  --table           support: the scene's support is a few small disks (radius 5-12 cm) where objects rested, so an
                    object put down or knocked anywhere else falls 1 m to the floor. Adds a 2 x 2 m table 2 mm under
                    the lowest frame-0 object bottom, and a seat disk 2 mm under every other object resting within
                    6 cm of it (stage 2 can leave a 1-6 cm gap). The robot never collides with support (training
                    collision config), so the table cannot block the reference hands.
The copy's parquet is pointed at the copy (relocate_scene.py), so the scene trains and evaluates as is.
Usage: python tools/track3/scene_variant.py <scene dir> <new scene dir> [--mass C=KG ...] [--thin C=F ...] [--table]
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relocate_scene import relocate  # noqa: E402

PROCESSED = 'assets/human_motion_data/ego_recon/processed'


def parse(pairs: list[str]) -> dict[str, float]:
    return {k: float(v) for k, v in (p.split('=') for p in pairs)}


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def objects(scene: Path) -> list[tuple[str, Path, np.ndarray, np.ndarray]]:
    """(object name, URDF, frame-0 rotation, frame-0 position) per body, in the parquet's body order."""
    table = pq.read_table(next((scene / PROCESSED).glob('sequence_id=*/robot_name=*/*.parquet')))
    urdfs = [Path(p) for p in table.column('object_urdf_paths')[0].as_py()]
    wxyz = np.asarray(table.column('object_body_wxyz')[0].as_py(), dtype=float)  # frames x bodies x 4
    pos = np.asarray(table.column('object_body_position')[0].as_py(), dtype=float)
    names = [m.stem for m in sorted((scene / 'score_input/meshes').glob('*/*.obj'))]
    out = []
    for i, urdf in enumerate(urdfs):
        hit = [n for n in names if urdf.name.endswith(f'_{n}_rigid.urdf')]
        name = hit[0] if hit else (names[0] if len(names) == 1 else urdf.stem)
        out.append((name, urdf, quat_wxyz_to_matrix(wxyz[0, i]), pos[0, i]))
    return out


def set_mass(urdf: Path, mesh: trimesh.Trimesh, kg: float) -> str:
    text = urdf.read_text()
    old_kg = float(re.search(r'<mass value="([^"]+)"', text).group(1))
    if mesh.is_watertight and mesh.volume > 0:
        mesh = mesh.copy()
        mesh.density = kg / mesh.volume
        com, inertia = mesh.center_mass, mesh.moment_inertia
        how = 'mesh'
    else:
        com, inertia = np.zeros(3), np.diag([0.001] * 3) * kg / old_kg
        how = 'scaled diag'
    text = re.sub(r'<origin xyz="[^"]*" rpy="0 0 0"/>\s*<mass value="[^"]+"/>',
                  f'<origin xyz="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" rpy="0 0 0"/>\n         <mass value="{kg}"/>',
                  text, count=1)
    text = re.sub(r'<inertia [^/]*/>',
                  '<inertia ixx="{:.3e}" ixy="{:.3e}" ixz="{:.3e}" iyy="{:.3e}" iyz="{:.3e}" izz="{:.3e}"/>'.format(
                      inertia[0, 0], inertia[0, 1], inertia[0, 2], inertia[1, 1], inertia[1, 2], inertia[2, 2]),
                  text, count=1)
    urdf.write_text(text)
    return f'mass {old_kg} -> {kg} kg, inertia {how}, com {np.round(com * 100, 2)} cm'


def thin_collision(urdf: Path, mesh: trimesh.Trimesh, rot0: np.ndarray, factor: float) -> str:
    v = mesh.vertices
    centre = v.mean(0)
    axis = np.linalg.svd(v - centre, full_matrices=False)[2][2]  # thinnest principal axis (object frame)
    down = rot0.T @ np.array([0.0, 0.0, -1.0])  # world down in the object frame at frame 0
    s = (v - centre) @ axis
    pivot = s.max() if axis @ down > 0 else s.min()  # the face resting on the table stays put
    thin = v + np.outer((s - pivot) * (factor - 1.0), axis)
    out = urdf.with_name(urdf.stem.replace('_rigid', '') + '_collision_thin.stl')
    trimesh.Trimesh(thin, mesh.faces, process=False).export(out)
    text = urdf.read_text()
    text = re.sub(r'(<collision>.*?<mesh filename=")[^"]+(")', rf'\g<1>./{out.name}\g<2>', text, count=1, flags=re.S)
    urdf.write_text(text)
    return (f'collision thickness {np.ptp(s) * 100:.1f} -> {np.ptp(s) * factor * 100:.1f} cm '
            f'(axis tilt from vertical {np.degrees(np.arccos(abs(axis @ down))):.0f} deg)')


def add_table(scene: Path, rested: list[tuple[str, np.ndarray]]) -> str:
    """rested: (name, frame-0 world vertices) per object."""
    usda = next((scene / 'assets/human_motion_data/ego_recon/reconstructed_stage').glob('*.usda'))
    bottoms = {name: v[:, 2].min() for name, v in rested}
    floor = min(bottoms.values())
    centre = np.mean([v[:, :2].mean(0) for _, v in rested], axis=0)
    prims = [('table', 'Cube', (centre[0], centre[1], floor - 0.002 - 0.01), 'double size = 1\n'
              '        float3 xformOp:scale = (2, 2, 0.02)', '"xformOp:translate", "xformOp:scale"')]
    notes = [f'table top z {floor - 0.002:.3f}']
    for name, v in rested:
        gap = bottoms[name] - floor
        if 0.005 < gap < 0.06:
            xy = v[:, :2].mean(0)
            radius = 1.15 * np.linalg.norm(v[:, :2] - xy, axis=1).max()
            prims.append((f'seat_{name}', 'Cylinder', (xy[0], xy[1], bottoms[name] - 0.002 - 0.005),
                          f'uniform token axis = "Z"\n        double height = 0.01\n        double radius = {radius:.4f}',
                          '"xformOp:translate"'))
            notes.append(f'seat {name} (gap {gap * 100:.1f} cm)')
    text = usda.read_text().rstrip()
    assert text.endswith('}')
    body = ''.join(
        f'\n    def {kind} "{name}" (\n        prepend apiSchemas = ["PhysicsCollisionAPI"]\n    )\n    {{\n        {attrs}\n'
        f'        double3 xformOp:translate = ({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})\n'
        f'        uniform token[] xformOpOrder = [{order}]\n    }}\n'
        for name, kind, p, attrs, order in prims)
    usda.write_text(text[:-1] + body + '}\n')
    return '; '.join(notes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('dst', type=Path)
    ap.add_argument('--mass', nargs='*', default=[])
    ap.add_argument('--thin', nargs='*', default=[])
    ap.add_argument('--table', action='store_true')
    a = ap.parse_args()
    if a.dst.exists():
        sys.exit(f'{a.dst} exists')
    shutil.copytree(a.src, a.dst, symlinks=True)
    relocate(a.dst)
    mass, thin = parse(a.mass), parse(a.thin)
    rested = []
    for name, urdf, rot0, pos0 in objects(a.dst):
        mesh_file = urdf.parent / re.search(r'<collision>.*?<mesh filename="\./([^"]+)"', urdf.read_text(), re.S).group(1)
        mesh = trimesh.load(mesh_file, force='mesh')
        rested.append((name, mesh.vertices @ rot0.T + pos0))
        notes = [set_mass(urdf, mesh, kg) for c, kg in mass.items() if c in name]
        notes += [thin_collision(urdf, mesh, rot0, f) for c, f in thin.items() if c in name]
        print(f'{a.dst.name} {name}: ' + ('; '.join(notes) or 'unchanged'))
    if a.table:
        lowest = min(v[:, 2].min() for _, v in rested)
        print(f'{a.dst.name} support: ' + add_table(a.dst, [r for r in rested if r[1][:, 2].min() - lowest < 0.06]))


if __name__ == '__main__':
    main()
