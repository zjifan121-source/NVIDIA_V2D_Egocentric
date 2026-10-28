"""Select complete learner/replay pairs; retain two for this isolated experiment."""
import json
from pathlib import Path
import struct
import sys


def metadata(path):
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        if size > 16_000_000:
            raise ValueError('invalid header size')
        header = json.loads(f.read(size))
        end = max((v['data_offsets'][1] for k, v in header.items() if k != '__metadata__'), default=0)
        if path.stat().st_size != 8 + size + end:
            raise ValueError('incomplete tensor file')
        return header['__metadata__']


def pairs(directory):
    result = []
    for state in directory.glob('state_*.safetensors'):
        step = int(state.stem.split('_')[1])
        replay = directory / f'replay_{step}.safetensors'
        try:
            a, b = metadata(state), metadata(replay)
            if a.get('user.checkpoint_set_id') and all(a.get(k) == b.get(k) for k in ('user.checkpoint_set_id', 'user.environment_steps')):
                result.append((step, state, replay))
        except (OSError, ValueError, KeyError, struct.error):
            continue
    return sorted(result)


if __name__ == '__main__':
    complete = pairs(Path(sys.argv[2]))
    if sys.argv[1] == 'latest':
        if complete:
            print(complete[-1][1])
    elif sys.argv[1] == 'prune':
        for _, state, replay in complete[:-2]:
            state.unlink(missing_ok=True)
            replay.unlink(missing_ok=True)
    else:
        raise SystemExit('expected latest or prune')
