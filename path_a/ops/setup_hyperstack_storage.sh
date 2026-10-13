#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One-time setup for the inspected, empty creative-tesla v2d-data volume.
# Run as root. Refuses an already initialized disk; never removes old storage.
set -euo pipefail
[[ $(id -u) == 0 && $(hostname) == creative-tesla ]]
device=/dev/disk/by-id/virtio-c6ff989f-5847-4f08-b
mountpoint=/mnt/v2d
[[ $(readlink -f "$device") == /dev/vdb ]]
[[ $(blockdev --getsize64 "$device") == 300647710720 ]]
[[ $(lsblk -nr -o TYPE "$device") == disk ]]
[[ -z $(lsblk -nr -o FSTYPE,MOUNTPOINTS "$device" | tr -d '[:space:]') ]]
[[ -z $(wipefs --no-act --noheadings "$device") ]]
[[ -z $(docker ps -q) ]]
[[ ! -e "$mountpoint" ]]
python3 - "$device" <<'PY'
import os, sys
with open(sys.argv[1], 'rb', buffering=0) as f:
    size = os.lseek(f.fileno(), 0, os.SEEK_END)
    for offset in [0, size//4, size//2, 3*size//4, size-1024*1024]:
        f.seek(offset)
        assert not any(f.read(1024*1024)), 'Nonzero disk sample; refusing format'
PY
mkfs.ext4 -L v2d-data -m 0 "$device"
mkdir -p "$mountpoint"
uuid=$(blkid -s UUID -o value "$device")
backup=/root/v2d-storage-backup-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -m 700 "$backup"
cp -a /etc/fstab "$backup/fstab"
cp -a /etc/docker/daemon.json "$backup/docker-daemon.json"
cp -a /etc/containerd/config.toml "$backup/containerd-config.toml"
printf 'UUID=%s /mnt/v2d ext4 defaults,noatime,nofail 0 2\n' "$uuid" >> /etc/fstab
systemctl daemon-reload
mount "$mountpoint"
mountpoint -q "$mountpoint"
mkdir -p "$mountpoint"/{docker,containerd,cache,code,datasets,weights,outputs,logs,tmp,venv}
chown ubuntu:ubuntu "$mountpoint"/{cache,code,datasets,weights,outputs,logs,tmp,venv}
chmod 700 "$mountpoint/cache"
systemctl stop docker.service docker.socket
systemctl stop containerd.service
rsync -aHAX --numeric-ids /var/lib/docker/ "$mountpoint/docker/"
rsync -aHAX --numeric-ids /var/lib/containerd/ "$mountpoint/containerd/"
python3 - <<'PY'
from pathlib import Path
import json, re
p = Path('/etc/docker/daemon.json')
cfg = json.loads(p.read_text())
cfg['data-root'] = '/mnt/v2d/docker'
cfg.setdefault('log-driver', 'json-file')
cfg.setdefault('log-opts', {'max-size': '20m', 'max-file': '3'})
p.write_text(json.dumps(cfg, indent=2) + '\n')
p = Path('/etc/containerd/config.toml')
text = p.read_text()
assert not re.search(r'^root\s*=', text, re.M), 'Inspect existing containerd root first'
p.write_text('root = "/mnt/v2d/containerd"\n' + text)
for service in ['docker', 'containerd']:
    p = Path(f'/etc/systemd/system/{service}.service.d/v2d-storage.conf')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('[Unit]\nRequiresMountsFor=/mnt/v2d\nConditionPathIsMountPoint=/mnt/v2d\n')
PY
systemctl daemon-reload
systemctl start containerd.service docker.service
[[ $(docker info --format '{{.DockerRootDir}}') == /mnt/v2d/docker ]]
containerd config dump | python3 -c 'import sys,tomllib; assert tomllib.loads(sys.stdin.read())["root"] == "/mnt/v2d/containerd"'
findmnt --fstab --evaluate --target /mnt/v2d
findmnt -T /mnt/v2d/docker
findmnt -T /mnt/v2d/containerd
docker image ls
docker ps -a
df -h / /mnt/v2d
printf 'Original Docker/containerd storage retained; config backup: %s\n' "$backup"
