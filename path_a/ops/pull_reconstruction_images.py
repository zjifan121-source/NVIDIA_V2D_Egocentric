#!/usr/bin/env python3
"""Pull pinned images sequentially onto the mounted V2D data disk and tag wrappers."""
import argparse
import datetime
import json
import shutil
import subprocess
import tomllib
from pathlib import Path


def output(args):
    return subprocess.check_output(args, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=Path(__file__).with_name('reconstruction_images_afe7dad.json'))
    parser.add_argument('--receipts', type=Path, default=Path('/mnt/v2d/logs/image_receipts.jsonl'))
    args = parser.parse_args()
    root = Path('/mnt/v2d')
    assert root.is_mount(), 'Data volume is not mounted'
    assert output(['docker', 'info', '--format', '{{.DockerRootDir}}']) == '/mnt/v2d/docker'
    assert tomllib.loads(Path('/etc/containerd/config.toml').read_text())['root'] == '/mnt/v2d/containerd'
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    for image in json.loads(args.lock.read_text())['images']:
        ref = image['image'] + '@' + image['platform_digest']
        existing = subprocess.run(['docker', 'image', 'inspect', ref], capture_output=True)
        if existing.returncode:
            free = shutil.disk_usage(root).free
            minimum = 35*2**30 + 2*image['compressed_bytes']
            if free < minimum:
                raise RuntimeError(f'Insufficient safe space for {ref}: {free/2**30:.1f} GiB free, require {minimum/2**30:.1f}')
            print('Pulling', ref, flush=True)
            subprocess.run(['docker', 'pull', '--quiet', '--platform', 'linux/amd64', ref], check=True)
        subprocess.run(['docker', 'tag', ref, image['local_tag']], check=True)
        subprocess.run(['docker', 'tag', ref, image['image'] + ':' + image['tag']], check=True)
        actual = json.loads(output(['docker', 'image', 'inspect', ref]))[0]
        assert actual['Architecture'] == 'amd64'
        assert ref in actual['RepoDigests'], (ref, actual['RepoDigests'])
        receipt = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'local_tag': image['local_tag'], 'reference': ref, 'index_digest': image['digest'],
                   'image_id': actual['Id'], 'size': actual['Size']}
        with args.receipts.open('a') as f:
            f.write(json.dumps(receipt) + '\n')
        print('Verified', image['local_tag'], actual['Id'], flush=True)


if __name__ == '__main__':
    main()
