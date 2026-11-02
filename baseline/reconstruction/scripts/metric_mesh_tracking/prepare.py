"""Freeze a portable input package or record an explicit visual anchor review (CPU)."""
import argparse
import json
from pathlib import Path
import run


def freeze(spec_path, payload, runtime_path, source_root, output):
    spec = json.loads(Path(spec_path).read_text())
    payload = Path(payload).resolve()
    if spec.get('metric_mesh_reviewed') is not True or spec.get('anchor_selection_reviewed') is not True:
        raise ValueError('Review metric geometry and input-only anchor selection before freezing')
    # Preserve payload-relative paths; no host details enter the manifest.
    spec['payload_root'] = '.'
    spec['files'] = {}
    required = {spec['mesh_file'], spec['K_file']}
    for row in spec['frames']:
        required.update(row[k] for k in ('rgb', 'depth', 'mask'))
    # Materials/textures must be listed explicitly, including every referenced file.
    required.update(spec.pop('mesh_dependencies', []))
    contact = spec.pop('contact_constraint', None)
    if contact is not None:
        required.update(contact[k] for k in ('initial_pose_file', 'camera_to_world_file', 'plane_file', 'point_file'))
    for name in sorted(required):
        spec['files'][name] = run.sha(run.checked_path(payload, name))
    source_names = ['v2d_foundation_pose/lib/foundation_pose_tracker.py',
                    'v2d_foundation_pose/lib/FoundationPose/estimater.py']
    cfg = {**run.FIXED, **{k: spec[k] for k in ('episode', 'object', 'frame_count', 'anchor_frame')},
           'GPU_window_seconds': 900, 'runner_sha256': run.sha(run.__file__),
           'expected_source_sha256': {n: run.sha(run.checked_path(source_root, n)) for n in source_names},
           'expected_backend_runtime': json.loads(Path(runtime_path).read_text())}
    if contact is not None:
        cfg['contact_constraint'] = contact
        cfg['contact_runner_sha256'] = run.sha(Path(run.__file__).with_name('contact.py'))
    runtime_keys = ('backend','model_registry','model_version','tensorrt_version','gpu_compute_capability','models')
    if any(cfg['expected_backend_runtime'].get(k) is None for k in runtime_keys):
        raise ValueError('Missing native tracker.runtime_info pin')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    run.write_json(output / 'manifest.json', spec)
    cfg['input_manifest_sha256'] = run.sha(output / 'manifest.json')
    run.write_json(output / 'protocol.json', cfg)
    run.load_package(output / 'manifest.json', output / 'protocol.json', payload)
    return output


def review(package, output, reviewer, evidence):
    package, output = Path(package), Path(output)
    m = json.loads((package / 'manifest.json').read_text())
    cfg = json.loads((package / 'protocol.json').read_text())
    provided = cfg.get('contact_constraint') is not None
    receipt = json.loads((output / 'anchor/RECEIPT.json').read_text())
    if receipt['state'] != 'COMPLETE_AWAITING_HUMAN_REVIEW' or receipt['register_calls'] != (0 if provided else 1):
        raise ValueError('No completed unique anchor registration')
    value = dict(proceed_to_fixed_tracking=True, visual_target_identity_checked=True,
                 visible_structure_alignment_checked=True, no_gross_target_miss=True, rank0_only=True,
                 protocol_sha256=run.sha(package/'protocol.json'),
                 input_manifest_sha256=run.sha(package/'manifest.json'),
                 mesh_sha256=m['files'][m['mesh_file']],
                 anchor_public_sha256=run.sha(output/'anchor/anchor_public.npy'),
                 anchor_frame=m['anchor_frame'], reviewer=reviewer, reviewed_evidence=evidence)
    if provided:
        value.pop('rank0_only')
        value['provided_pose_reviewed'] = True
    if receipt['protocol_sha256'] != value['protocol_sha256'] or receipt['input_manifest_sha256'] != value['input_manifest_sha256']:
        raise ValueError('Anchor belongs to different inputs')
    run.validate_review(value, value['protocol_sha256'], value['input_manifest_sha256'],
                        value['mesh_sha256'], output/'anchor/anchor_public.npy', m['anchor_frame'], provided=provided)
    with (output/'ANCHOR_REVIEW.json').open('x') as f:
        json.dump(value, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    for name in ('spec', 'payload', 'runtime-info', 'source-root', 'output'):
        f.add_argument('--'+name, required=True, type=Path)
    r = sub.add_parser('review')
    r.add_argument('--package', required=True, type=Path)
    r.add_argument('--output', required=True, type=Path)
    r.add_argument('--reviewer', required=True)
    r.add_argument('--evidence', required=True, nargs='+')
    r.add_argument('--confirm-identity-structure-alignment', required=True, action='store_true',
                   help='Only pass after visually inspecting the rank-0 anchor overlay; this is not automatic acceptance')
    a=p.parse_args()
    if a.command=='freeze':
        freeze(a.spec,a.payload,a.runtime_info,a.source_root,a.output)
    else:
        review(a.package,a.output,a.reviewer,a.evidence)

if __name__=='__main__':
    main()
