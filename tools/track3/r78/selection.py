"""Frozen structure-first selection; never reads scores or scanned geometry."""
from pathlib import Path
import numpy as np
from .io import artifact, fresh, read, seal, sha, check_review, write


def choose(rows):
    """Pareto dominance on both heldout views, otherwise original pool order."""
    if not rows:
        raise ValueError('No structurally accepted, bounded candidate')
    if len(rows) == 1:
        return rows[0], 'Only feasible candidate; selection ability not tested'
    def vector(row):
        return np.array([sign * f['candidate'][key] for f in row['heldout']
                         for key, sign in [('mask_iou', 1), ('median_surface_distance_m', -1),
                                           ('valid_observed_mask_coverage', 1)]])
    a, b = map(vector, rows)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite heldout metrics')
    if np.all(b >= a) and np.any(b > a):
        return rows[1], 'Second candidate Pareto-dominates both heldout views'
    if np.all(a >= b) and np.any(a > b):
        return rows[0], 'First candidate Pareto-dominates both heldout views'
    return rows[0], 'Input evidence does not distinguish: frozen pool-order tie break'


def run(pool, output):
    """Pool must be frozen before fitting/scoring; failed arms stay in receipt."""
    path = Path(pool).resolve()
    spec = read(path)
    if (spec.get('GT_used') is not False or not spec.get('frozen_before_evaluation') or
            not 1 <= len(spec.get('candidates', [])) <= 2):
        raise ValueError('Require a predeclared input-only pool of at most two candidates')
    ids = [c['id'] for c in spec['candidates']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate candidate IDs')
    out = fresh(output)
    write(out / 'frozen_pool.json', spec)
    rows, failures, common = [], [], None
    for candidate in spec['candidates']:
        p = (path.parent / candidate['fit'] / 'artifact.json').resolve()
        if not p.exists():
            failure = p.parent / 'failure.json'
            if not failure.is_file():
                raise ValueError('Candidate missing completion or failure record: ' + candidate['id'])
            failures.append(dict(id=candidate['id'], failure=read(failure), sha256=sha(failure)))
            continue
        d = artifact(p, {'FIT_BOUNDED'})
        check_review(d['structure_review'], d['input_mesh_sha256'])
        signature = (d['context'], d['config'],
                     sorted([(f['id'], f['role'], f['input_sha256']) for f in d['frames']]))
        if common is not None and signature != common:
            raise ValueError('Candidates use different observations, split or solver settings')
        common = signature
        held = sorted([r for r in read(p.parent / 'metrics.json') if r['role'] == 'heldout'], key=lambda r: r['id'])
        if len(held) != 2:
            raise ValueError('Two heldout metrics required')
        rows.append(dict(id=candidate['id'], artifact=str(p), artifact_sha256=sha(p),
                         mesh_sha256=d['mesh_sha256'], heldout=held))
    selected, reason = choose(rows)
    return seal(out, dict(state='SELECTED', selected_candidate=selected['id'],
        selected_artifact=selected['artifact'], selected_artifact_sha256=selected['artifact_sha256'],
        mesh_sha256=selected['mesh_sha256'], reason=reason, candidates=rows, failures=failures,
        pool_sha256=sha(path), GT_used=False))


def selected_asset(path):
    selection = artifact(path, {'SELECTED'})
    p = (Path(path).resolve().parent / selection['selected_artifact']).resolve()
    if sha(p) != selection['selected_artifact_sha256']:
        raise ValueError('Selected fit changed')
    d = artifact(p, {'FIT_BOUNDED'})
    if d['mesh_sha256'] != selection['mesh_sha256']:
        raise ValueError('Selected mesh changed')
    return p, d
