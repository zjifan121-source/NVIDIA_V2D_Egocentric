#!/usr/bin/env python3
"""Install the user's licensed MANO model at the WiLoR renderer's expected path.

WiLoR downloads pretrained_models/MANO_RIGHT.pkl, whereas manotorch requires
pretrained_models/models/MANO_RIGHT.pkl. Use the user's own hand/models copy;
keep downloaded files unchanged and refuse to overwrite a different model.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ensure_layout(weights):
    source = weights / 'hand/models/MANO_RIGHT.pkl'
    target = weights / 'wilor/pretrained_models/models/MANO_RIGHT.pkl'
    expected = digest(source)
    if target.exists() and digest(target) != expected:
        raise RuntimeError(f'Refusing to replace a different MANO model: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    assert digest(target) == expected
    return {'source': str(source), 'target': str(target), 'sha256': expected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=Path('/mnt/v2d/weights'))
    args = parser.parse_args()
    print(json.dumps(ensure_layout(args.weights), indent=2))
