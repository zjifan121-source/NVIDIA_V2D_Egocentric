"""Step 2 on F2's pairs: pathb.trackability.features per episode x candidate on F2's S5 tasks (copied from the
VM's v4_h1 to path_b/data/outputs/v4_h1_vm), moving frames and rests from the trained reference's S3c report
(the md5-matched Mac run). Writes v4_h1_vm/episode_X/s4b2_trackability[_pre_k7]/<cand>/trackability.json and the
0.4 contact split per S3c kind (inverted candidate)."""
import sys, json, time
from pathlib import Path
REPO = Path('/Users/jzzzzz79/V2D/NVIDIA_V2D_Egocentric')
sys.path.insert(0, str(REPO / 'path_b'))
import numpy as np
from pathb import trackability as T
OUT = REPO / 'path_b/data/outputs'
VM = OUT / 'v4_h1_vm'
rows = [r for r in json.load(open(sys.argv[1])) if 'error' not in r]
split = {}
for r in rows:
    ep, cand = r['episode'], r['candidate']
    suffix = '_pre_k7' if r['folder'].endswith('_pre_k7') else ''
    run = r['playback_run']
    reports = {}
    for o in r['objects']:
        f = OUT / run / f'episode_{ep:06d}/s3c_invert/{o}/invert_report.json'
        reports[o] = json.loads(f.read_text()) if f.is_file() else None
    cand_dir = VM / f'episode_{ep:06d}/s5_task{suffix}/{cand}'
    t0 = time.time()
    f = T.features(cand_dir, cand, reports, s3a_root=OUT / run / f'episode_{ep:06d}')
    f.update(seconds=time.time() - t0, reference_run=run, trained_md5=r['trained_md5'], source=str(cand_dir.relative_to(OUT)))
    dest = VM / f'episode_{ep:06d}/s4b2_trackability{suffix}/{cand}/trackability.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(f, indent=1, default=float) + '\n')
    if cand == 'inverted':
        t = T.task(cand_dir)
        n = len(t['cells']['object_body_position'])
        acc = {k: [0, 0, 0] for k in ('static', 'grasp', 'bridge', 'other')}
        for k, o in enumerate(T.objects(t)):
            lab = T.labels(reports.get(o['name']), n)
            slots = T.contact_slots(t['cells'], k)
            static = np.zeros(n, bool)
            for a, b in lab['static']:
                static[a:b + 1] = True
            kinds = {'bridge': lab['bridged'], 'grasp': lab['grasp'] & ~lab['bridged'], 'static': static & ~lab['bridged'] & ~lab['grasp']}
            kinds['other'] = ~(kinds['bridge'] | kinds['grasp'] | kinds['static'])
            for kname, m in kinds.items():
                acc[kname][0] += int(m.sum()); acc[kname][1] += int((slots[m] >= 1).sum()); acc[kname][2] += int((slots[m] >= 2).sum())
        split[f'{ep}{suffix}'] = {k: {'frames': v[0], 'ge1': v[1] / v[0] if v[0] else None, 'ge2': v[2] / v[0] if v[0] else None} for k, v in acc.items()}
    rss = f['rss']
    print(f"ep{ep:>2}{suffix:7} {cand:8} {f['seconds']:5.1f}s moving={f['n_moving']:4d} ums={f['ums']:.2f} opp={f['ums_opp']:.2f} run={f['ums_run']:3d} "
          f"dhand={f['d_hand_cm'] if f['d_hand_cm'] is None else round(f['d_hand_cm'], 1)} rcc1={f['rcc1'] if f['rcc1'] is None else round(f['rcc1'], 2)} "
          f"rss0={rss['rss0_cm']:.1f}cm/{rss['rss0_deg']:.0f}d max={rss['rss_max_cm']:.1f}/{rss['rss_max_deg']:.0f} unsettled={rss['unsettled_rests']}/{len(rss['rests'])} "
          f"support={int(f['support_ok'])}", flush=True)
json.dump(split, open(sys.argv[2], 'w'), indent=1)
