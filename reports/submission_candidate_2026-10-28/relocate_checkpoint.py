"""Point a policy's training-scene path at our relocated copy of the same scene; weights unchanged.

Night-run policies record the builder's private scene path, which the evaluator cannot read here.
python relocate_checkpoint.py POLICY SCENE_DIR OUT_POLICY
"""
import hashlib, json, struct, sys
from pathlib import Path

policy, scene, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
raw = policy.read_bytes()
n = struct.unpack('<Q', raw[:8])[0]
header = json.loads(raw[8:8 + n])
meta = header['__metadata__']
config = json.loads(meta['user.resolved_config_json'])
old = config['task']['parquet']
new = next(scene.glob('assets/human_motion_data/ego_recon/processed/sequence_id=*/robot_name=sharpa_wave'))
if old.endswith('.parquet'):
    new = new / Path(old).name
assert new.exists() and old.split('/sequence_id=')[1] == str(new).split('/sequence_id=')[1], (old, new)
config['task']['parquet'] = str(new)
meta['user.resolved_config_json'] = json.dumps(config, separators=(',', ':'))
meta['user.relocated_from_parquet'] = old
meta['user.relocation_source_sha256'] = hashlib.sha256(raw).hexdigest()
meta['user.relocation_weights_unchanged'] = 'true'
h = json.dumps(header, separators=(',', ':')).encode()
h += b' ' * ((-len(h)) % 8)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(struct.pack('<Q', len(h)) + h + raw[8 + n:])
assert out.read_bytes()[8 + len(h):] == raw[8 + n:]
print(f'{policy.name}: {old} -> {new}')
