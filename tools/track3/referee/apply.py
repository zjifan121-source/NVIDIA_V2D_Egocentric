"""Apply the referee to every object of a namespace and write the result as a new namespace (CPU).

For every <episode>/<object> run of the source namespace: hard-link the run into the target namespace, leaving out
frames/ and the run's bundles (result/, result_slam/, result_slam_gravity_aligned/, threejs/), which were built from the
unrefereed track; for objects not skipped that have a track file, relabel the run's raw FoundationPose poses (poses/)
and write them as the target run's poses_smoothed/, which tools/track3/write_unrefined_bundle.py reads. This is the
variant measured in the README: refereed objects lose the R10 EKF smoothing; cups and the pot lid are left unchanged.
Rebuild the target's bundles with write_unrefined_bundle.py before scoring it.

The source namespace is never modified: the target holds hard links (copies where linking fails) and new files only.
Because the target's files share storage with the source's, replace files in the target, never edit them in place.
Runs tracked with a provided mesh and no mesh_scaled.obj get a copy of that mesh (and its .mtl and textures) as
mesh_scaled.obj in the target, so the bundle writer can rebuild their bundles.
Track files: <tracks_dir>/ep<EP>_<object>.npz with EP not zero-padded (e.g. ep19_mini_sweeper.npz)."""
import json
import os
import re
import shutil
from pathlib import Path

from . import referee

SKIP_DEFAULT = ('blue_cup', 'beige_cup', 'white_pot_lid')     # symmetric in the metric (score_local SYMMETRY_ORDER)
LEFT_OUT = {'frames', 'result', 'result_slam', 'result_slam_gravity_aligned', 'threejs'}


def _link_or_copy(s, d):
    try:
        os.link(s, d)
    except OSError:
        shutil.copy2(s, d)


def _link_tree(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name in LEFT_OUT:
            continue
        if p.is_symlink():
            os.symlink(os.readlink(p), dst / p.name)
        elif p.is_dir():
            shutil.copytree(p, dst / p.name, copy_function=_link_or_copy, symlinks=True)
        else:
            _link_or_copy(p, dst / p.name)


def _copy_mesh(mesh, dst):
    """mesh -> dst/mesh_scaled.obj, plus the .mtl files it names and the texture maps they name, next to it."""
    shutil.copy2(mesh, dst / 'mesh_scaled.obj')
    src = Path(mesh).parent
    for line in Path(mesh).read_text(errors='ignore').splitlines():
        if line.startswith('mtllib '):
            mtl = src / line.split(None, 1)[1].strip()
            if mtl.is_file() and not (dst / mtl.name).exists():
                shutil.copy2(mtl, dst / mtl.name)
                for l2 in mtl.read_text(errors='ignore').splitlines():
                    if l2.strip().startswith('map_'):
                        tex = src / l2.strip().split()[-1]
                        if tex.is_file() and not (dst / tex.name).exists():
                            shutil.copy2(tex, dst / tex.name)


def run(outputs, source, target, tracks_dir, episodes=None, skip=SKIP_DEFAULT, anchor='0', margin=0.6):
    if Path(target).name != target or target in ('.', '..', source):
        raise SystemExit(f'target must be a new plain namespace name, not {target!r}')
    outputs = Path(outputs)
    src, dst = outputs / source, outputs / target
    if not src.is_dir():
        raise SystemExit(f'no namespace {src}')
    if dst.exists():
        raise SystemExit(f'{dst} exists: choose a new target namespace')
    eps = sorted(p for p in src.iterdir() if p.is_dir() and re.fullmatch(r'episode_\d{6}', p.name))
    if episodes:
        want = {f'episode_{int(e):06d}' for e in episodes}
        eps = [e for e in eps if e.name in want]
    if not eps:
        raise SystemExit(f'no episode folders selected in {src}')
    work = dst / '_referee'
    summary = []
    try:
        for ep_dir in eps:
            ep = int(ep_dir.name.split('_')[1])
            for obj_dir in sorted(p for p in ep_dir.iterdir() if p.is_dir()):
                obj = obj_dir.name
                tgt = dst / ep_dir.name / obj
                row = dict(episode=ep, object=obj)
                summary.append(row)
                try:
                    _link_tree(obj_dir, tgt)
                    if not (tgt / 'mesh_scaled.obj').exists():
                        m = referee.find_mesh(obj_dir)
                        _copy_mesh(m, tgt)
                        row['mesh_copied_from'] = str(m)
                    npz = Path(tracks_dir) / f'ep{ep}_{obj}.npz'
                    if obj in skip:
                        row['status'] = 'skipped (symmetric in the metric)'
                    elif not npz.is_file():
                        row['status'] = f'unchanged (no track file {npz.name})'
                    elif not (obj_dir / 'poses').is_dir():
                        row['status'] = 'unchanged (no FoundationPose poses/)'
                    else:
                        out = work / f'ep{ep}_{obj}'
                        rep = referee.run(obj_dir, npz, out, poses='poses', anchor=anchor, margin=margin)
                        if (tgt / 'poses_smoothed').is_symlink():
                            (tgt / 'poses_smoothed').unlink()
                        elif (tgt / 'poses_smoothed').exists():
                            shutil.rmtree(tgt / 'poses_smoothed')   # this target's hard links only; the source keeps its files
                        shutil.copytree(out, tgt / 'poses_smoothed')
                        row.update(status='refereed', anchor=anchor, twin_frames=rep['twin_frames'],
                                   labeled=rep['labeled'], segments=rep['segments'])
                except SystemExit as e:
                    row['status'] = f'unchanged ({e})'
                except Exception as e:  # noqa: BLE001 - one bad run must not stop the namespace
                    row['status'] = f'unchanged (error: {type(e).__name__}: {e})'
                extra = f" twin frames {row['twin_frames']}" if 'twin_frames' in row else ''
                extra += ' (mesh copied)' if 'mesh_copied_from' in row else ''
                print(f"ep{ep:02d} {obj:20s} {row['status']}{extra}", flush=True)
    finally:
        work.mkdir(parents=True, exist_ok=True)
        (work / 'summary.json').write_text(json.dumps(dict(source=source, target=target, anchor=anchor, skip=list(skip),
                                                           rows=summary), indent=1))
    print('APPLY_OK', dst, '- rebuild its bundles with tools/track3/write_unrefined_bundle.py before scoring')
