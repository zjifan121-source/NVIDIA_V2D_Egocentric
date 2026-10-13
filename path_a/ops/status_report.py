#!/usr/bin/env python3
"""Read reconstruction states and capacity; optionally save an audited checklist.

Run on the experiment host. RUNNING is a runner receipt, not proof of liveness;
the separate process_alive field, Docker inventory and log age support review.
This tool never launches, retries, deletes or marks outputs as validated.
"""
import argparse
import datetime
import json
import re
import shutil
import subprocess
from pathlib import Path
from storage import volumes


def disk_bytes(path):
    return int(subprocess.check_output(['du', '-s', '-B1', str(path)], text=True).split()[0])


def report(root, inventory):
    now = datetime.datetime.now(datetime.timezone.utc)
    usage = shutil.disk_usage(root)
    rows = []
    for episode in inventory:
        for obj in episode['objects']:
            out = root / f"ep{episode['episode']:02d}" / obj
            row = {'episode': episode['episode'], 'split': episode['split'], 'object': obj,
                   'frames': episode['frames'], 'state': 'NOT_STARTED'}
            status_path = out / 'status.json'
            if status_path.exists():
                state = json.loads(status_path.read_text())
                row.update(state)
                row['output_bytes'] = disk_bytes(out)
                row['process_alive'] = Path(f"/proc/{state.get('pid', -1)}").exists()
                logs = sorted(out.glob('runs/*/pipeline.log'))
                if logs:
                    log = logs[-1]
                    text = log.read_text(errors='replace')
                    stages = re.findall(r'\[run \] ([^\r\n]+)', text)
                    row['last_started_stage'] = stages[-1] if stages else None
                    row['log_age_seconds'] = round(now.timestamp() - log.stat().st_mtime)
                    errors = [line.strip() for line in text.replace('\r', '\n').splitlines()
                              if re.match(r'^[\w.]+(?:Error|Exception):', line)]
                    row['last_error'] = errors[-1][:1000] if errors else None
                row['attempts'] = len(logs)
            rows.append(row)
    disk = {'total_gib': round(usage.total / 2**30, 2), 'free_gib': round(usage.free / 2**30, 2),
            'warning_below_gib': 60, 'launch_floor_gib': 40,
            'state': 'STOP_NEW_RUNS' if usage.free < 40 * 2**30 else
                     'CAPACITY_WARNING' if usage.free < 60 * 2**30 else 'OK'}
    docker = subprocess.check_output(['docker', 'ps', '--format', '{{.Image}} {{.Status}}'], text=True).strip()
    return {'observed_utc': now.isoformat(), 'disk': disk, 'volumes': volumes(), 'docker': docker, 'objects': rows}


def markdown(data):
    lines = ['# Reconstruction checklist', '', f"Observed UTC: {data['observed_utc']}", '',
             *[f"{v['mount']}: {v['free_gib']} GiB free ({v['state']})." for v in data['volumes']],
             'Volumes are separate; a running episode must fit on its own volume.',
             'VALIDATED means numeric QA plus a recorded visual review; it does not mean policy evaluation or an official score.', '',
             '| Episode | Split | Object | State | Latest stage | GiB |',
             '|---|---|---|---|---|---:|']
    for row in data['objects']:
        size = f"{row['output_bytes'] / 2**30:.2f}" if 'output_bytes' in row else ''
        stage = (row.get('last_started_stage') or '').replace('|', '/')
        lines.append(f"| {row['episode']} | {row['split']} | {row['object']} | {row['state']} | {stage} | {size} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/v2d/outputs/wentacc_dev'))
    parser.add_argument('--inventory', type=Path, default=Path('/mnt/v2d/logs/descending_queue_inventory.json'))
    parser.add_argument('--write', action='store_true', help='Write reports and append a capacity observation in output root')
    args = parser.parse_args()
    data = report(args.root, json.loads(args.inventory.read_text()))
    if args.write:
        for name, content in [('checklist.json', json.dumps(data, indent=2) + '\n'),
                              ('CHECKLIST.md', markdown(data))]:
            tmp = args.root / (name + '.tmp')
            tmp.write_text(content)
            tmp.replace(args.root / name)
        with (args.root / 'capacity_observations.jsonl').open('a') as stream:
            stream.write(json.dumps(data) + '\n')
    print(markdown(data), end='')


if __name__ == '__main__':
    main()
