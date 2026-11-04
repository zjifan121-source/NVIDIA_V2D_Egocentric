"""Delete resume-only files (state_*, replay_*) from finished V2D runs that are not selected and not running.

Policies, results and logs stay, so every checkpoint can still be evaluated. Writes a manifest of what went.
python prune_runs.py [--dry-run]
"""
import json, subprocess, sys, time
STALE_S = 24 * 3600  # an unfinished run untouched this long is abandoned
from pathlib import Path
DIAG = Path('/data/user_data/edmondwa/v2d/flash_runs/diag')
HERE = Path('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/policy_targeted_2026-10-28')
plan = json.load(open('/home/edmondwa/NVIDIA_V2D_Egocentric/reports/submission_candidate_2026-10-28/plan.json'))
keep = {Path(v['run']).parent.name if Path(v['run']).name.startswith('early_') else Path(v['run']).name for v in plan.values()}
keep |= {k[:-3] for k in keep if k.endswith('_RS')}  # a re-scored winner's source run
running = set(subprocess.run('squeue -h -u edmondwa,sarveshp -o %j', shell=True, capture_output=True, text=True).stdout.split())
dry = '--dry-run' in sys.argv
freed, removed, skipped = 0, [], []
for run in sorted(DIAG.iterdir()):
    train = run / 'train'
    if not train.is_dir() or run.name in keep or run.name in running:
        continue
    newest = max((f.stat().st_mtime for f in train.iterdir()), default=0)
    if not (run / 'train.done').exists() and time.time() - newest < STALE_S:
        skipped.append(run.name)
        continue
    for f in train.iterdir():
        if f.name.startswith(('state_', 'replay_')):
            freed += f.stat().st_size
            removed.append(str(f))
            if not dry:
                f.unlink()
manifest = HERE / f'prune_manifest_{time.strftime("%Y%m%d_%H%M")}.json'
manifest.write_text(json.dumps(dict(dry_run=dry, kept_selected=sorted(keep), skipped_unfinished=skipped, removed=removed, freed_gb=freed / 1024**3), indent=1))
print(f'{"would free" if dry else "freed"} {freed / 1024**3:.1f} GB in {len(removed)} files; unfinished skipped: {len(skipped)}; manifest {manifest}')
