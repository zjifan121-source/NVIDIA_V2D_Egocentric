"""Verified data-volume placement; never move an existing episode or use root disk."""
import json
import os
import shutil
import subprocess
from pathlib import Path

REGISTRY = Path('/mnt/v2d/logs/storage_volumes.json')


def volumes():
    config = json.loads(REGISTRY.read_text())['volumes'] if REGISTRY.exists() else [{'mount': '/mnt/v2d'}]
    rows = []
    for entry in config:
        mount = Path(entry['mount'])
        mounted = mount.is_mount() and mount.stat().st_dev != Path('/').stat().st_dev
        uuid = subprocess.check_output(['findmnt', '-n', '-o', 'UUID', '--target', str(mount)], text=True).strip() if mounted else None
        verified = mounted and (not entry.get('uuid') or uuid == entry['uuid'])
        usage = shutil.disk_usage(mount) if verified else None
        free = usage.free / 2**30 if usage else 0
        rows.append({'mount': str(mount), 'uuid': uuid, 'verified': verified,
                     'free_gib': round(free, 2), 'total_gib': round(usage.total / 2**30, 2) if usage else 0,
                     'state': 'UNAVAILABLE' if not verified else 'STOP_NEW_RUNS' if free < 40 else
                              'CAPACITY_WARNING' if free < 60 else 'OK'})
    return rows


def episode_directory(root, name, available=None):
    """Choose free space for a new episode, keep existing placements unchanged."""
    candidates = volumes() if available is None else available
    logical = root / name
    if os.path.lexists(logical):
        resolved = logical.resolve()
        matches = [v for v in candidates if resolved.is_relative_to(Path(v['mount']) / 'outputs/wentacc_dev')]
        if len(matches) != 1 or not matches[0]['verified']:
            raise RuntimeError(f'Existing episode volume is unavailable or unregistered: {logical}')
        if matches[0]['free_gib'] < 40:
            raise RuntimeError(f'Existing episode volume has less than 40 GiB free: {logical}')
        if not logical.is_dir():
            raise RuntimeError(f'Existing episode is not a readable directory: {logical}')
        return logical
    eligible = [v for v in candidates if v['verified'] and v['free_gib'] >= 40]
    if not eligible:
        raise RuntimeError('No verified data volume has 40 GiB free')
    chosen = max(eligible, key=lambda v: v['free_gib'])
    target = Path(chosen['mount']) / 'outputs/wentacc_dev' / name
    if target != logical.resolve() and target.exists():
        raise RuntimeError(f'Unlinked existing episode requires inspection: {target}')
    target.mkdir(parents=True, exist_ok=True)
    if target != logical.resolve():
        logical.symlink_to(target, target_is_directory=True)
    return logical
