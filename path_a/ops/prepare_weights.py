#!/usr/bin/env python3
"""Download baseline weights detached; accept HF token over encrypted stdin only.

Example: pipe an existing protected HF token file to this script over SSH with
--launch-from-stdin. No token is written to disk, command arguments or logs.
Requires the user's separate acceptance of the FoundationPose model license.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch-from-stdin', action='store_true')
    mode.add_argument('--worker', action='store_true')
    parser.add_argument('--accept-nvidia-model-eula', action='store_true', required=True)
    args = parser.parse_args()
    root = Path('/mnt/v2d')
    assert root.is_mount()
    repo = Path(__file__).resolve().parents[2]
    if args.launch_from_stdin:
        token = sys.stdin.read().strip()
        if not token.startswith('hf_') or any(c.isspace() for c in token):
            raise ValueError('Expected a HF token on stdin (value is never logged)')
        env = os.environ.copy()
        env['HF_TOKEN'] = token
        with (root / 'logs/weights.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      '--worker', '--accept-nvidia-model-eula'],
                                     cwd=repo, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True, close_fds=True)
        print(json.dumps({'detached_pid': child.pid, 'log': str(root / 'logs/weights.log')}))
        return
    lock = (root / 'weights/.setup.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / 'logs/weights_status.json'
    state = {'started': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'state': 'RUNNING', 'pid': os.getpid(), 'code_commit': subprocess.check_output(
                 ['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()}
    state['download_script_sha256'] = hashlib.sha256((repo / 'baseline/reconstruction/scripts/download_ego_reconstruction_weights.sh').read_bytes()).hexdigest()
    state['git_status'] = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
    status_path.write_text(json.dumps(state, indent=2) + '\n')
    try:
        # The published SAM3D image combines httpx2 with an older Brotli API.
        # Populate its public MoGE cache using the working host downloader.
        # The container then reuses these exact files without changing its ML stack.
        from huggingface_hub import snapshot_download
        state['sam3d_moge_snapshot'] = snapshot_download(
            'Ruicheng/moge-vitl', cache_dir=str(root / 'weights/sam3d/hf_home/hub'), token=False)
        subprocess.run(['bash', 'scripts/download_ego_reconstruction_weights.sh',
                        '--mode', 'hamer_prompt', '--accept-nvidia-model-eula', '--dev'],
                       cwd=repo / 'baseline/reconstruction', check=True)
        records = []
        for path in sorted((root / 'weights').rglob('*')):
            if not path.is_file() or '.cache' in path.parts or path.name.startswith('.'):
                continue
            with path.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            records.append({'path': str(path.relative_to(root / 'weights')), 'bytes': path.stat().st_size,
                            'sha256': digest})
        (root / 'logs/weight_receipts.json').write_text(json.dumps(records, indent=2) + '\n')
        state['state'] = 'DOWNLOADED_AND_HASHED'
        state['files'] = len(records)
    except Exception as exc:
        state['state'] = 'FAILED'
        state['reason'] = str(exc)
        raise
    finally:
        state['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status_path.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
