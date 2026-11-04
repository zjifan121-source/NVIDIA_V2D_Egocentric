"""Print the Hydra overrides that reproduce a checkpoint's training recipe (for continuing it).

python recipe_overrides.py POLICY.safetensors
"""
import json, struct, sys

raw = open(sys.argv[1], 'rb').read(1 << 22)
n = struct.unpack('<Q', raw[:8])[0]
cfg = json.loads(json.loads(raw[8:8 + n])['__metadata__']['user.resolved_config_json'])
KEYS = ['task.motion_speed', 'sim.fps', 'sim.njmax', 'sim.nconmax',
        'termination.wrist_position.threshold', 'termination.object_position.threshold', 'termination.object_orientation.threshold',
        'action.normalized_mapping', 'training.network.initial_normalized_std', 'training.network.log_std_max',
        'training.exploration.target_sigma', 'training.replay.capacity', 'training.replay.observation_storage_dtype']
out = []
for key in KEYS:
    node = cfg
    for part in key.split('.'):
        node = node.get(part) if isinstance(node, dict) else None
    if node is not None and not isinstance(node, (dict, list)):
        out.append(f'{key}={node}')
cur = cfg.get('curriculum', {})
name = cur.get('name') if isinstance(cur, dict) else None
print(json.dumps(dict(overrides=' '.join(out), curriculum=name, parquet=cfg['task']['parquet'], seed=cfg.get('reset', {}).get('seed'))))
