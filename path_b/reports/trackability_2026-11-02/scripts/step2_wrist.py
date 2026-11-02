"""Add the carrier-wrist features to F2's 39 trackability.json files: pathb.trackability.features(rss=False) with the
current module; the RSS block is kept from step2_pairs.py's run (its code did not change). Prints the wrist terms.

    python step2_wrist.py facts02.json
"""
import sys, json
from pathlib import Path
REPO = Path('/Users/jzzzzz79/V2D/NVIDIA_V2D_Egocentric')
sys.path.insert(0, str(REPO / 'path_b'))
from pathb import trackability as T
OUT = REPO / 'path_b/data/outputs'
VM = OUT / 'v4_h1_vm'
for r in [r for r in json.load(open(sys.argv[1])) if 'error' not in r]:
    ep, cand, run = r['episode'], r['candidate'], r['playback_run']
    suffix = '_pre_k7' if r['folder'].endswith('_pre_k7') else ''
    reports = {}
    for o in r['objects']:
        f = OUT / run / f'episode_{ep:06d}/s3c_invert/{o}/invert_report.json'
        reports[o] = json.loads(f.read_text()) if f.is_file() else None
    dest = VM / f'episode_{ep:06d}/s4b2_trackability{suffix}/{cand}/trackability.json'
    old = json.loads(dest.read_text())
    new = T.features(VM / f'episode_{ep:06d}/s5_task{suffix}/{cand}', cand, reports, rss=False,
                     s3a_root=OUT / run / f'episode_{ep:06d}')
    new.update(rss=old['rss'], rss_from='step2_pairs.py run (RSS code unchanged)', reference_run=run,
               trained_md5=r['trained_md5'], source=old['source'])
    assert abs(new['ums'] - old['ums']) < 1e-9 and new['n_moving'] == old['n_moving'], (ep, cand)  # same UMS as before
    dest.write_text(json.dumps(new, indent=1, default=float) + '\n')
    print(f"ep{ep:>2}{suffix:7} {cand:8} wrist p95 {new['wrist_p95']:5.2f} max {new['wrist_max']:6.2f} m/s, fast {new['wrist_fast']:.3f}, "
          f"acc p95 {new['wrist_acc_p95']:6.1f} m/s2, ang p95 {new['wrist_ang_p95']:6.1f} deg/s", flush=True)
