# ep3 skeptic (white_pot COM): re-check the pot tipping census from the rollout parquets themselves.
# usage (Babel, flash_chord venv python): ep3com_census.py <bob1 scene dir> <out.json> <run_label=parquet> ...
import sys, glob, json, os, re, numpy as np, pyarrow.parquet as pq, trimesh
SC = sys.argv[1].rstrip('/'); OUT = sys.argv[2]; RUNS = [a.split('=', 1) for a in sys.argv[3:]]
proc = SC + '/assets/human_motion_data/ego_recon/processed/'
TH = np.arange(0.01, 0.10, 0.01); AX = np.linspace(0, 1, TH.size)
def auc(add): return float(np.trapezoid(np.array([np.mean(add < t) for t in TH]), x=AX))
def q2m(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
MESH = {}
for f in glob.glob(proc + '*_visual.stl'):
    short = 'white_pot' if 'white_pot' in f else 'wooden_spoon'
    MESH[short] = np.asarray(trimesh.load(f, force='mesh', process=True).vertices)
def load(p):
    t = pq.read_table(p).to_pydict(); out = {}
    names = [str(n) for n in t['object_body_name']]
    for nm in sorted(set(names)):
        idx = [i for i, n in enumerate(names) if n == nm]
        steps = np.array([t['step'][i] for i in idx]); o = np.argsort(steps); idx = [idx[i] for i in o]
        g = lambda kk: np.array([t[kk][i] for i in idx], float)
        short = 'white_pot' if 'white_pot' in nm else 'wooden_spoon'
        out[short] = dict(name=nm, frame=np.array([t['reference_frame'][i] for i in idx]),
                          ap=np.c_[g('achieved_position_x'), g('achieved_position_y'), g('achieved_position_z')],
                          aq=np.c_[g('achieved_quaternion_w'), g('achieved_quaternion_x'), g('achieved_quaternion_y'), g('achieved_quaternion_z')],
                          rp=np.c_[g('reference_position_x'), g('reference_position_y'), g('reference_position_z')],
                          rq=np.c_[g('reference_quaternion_w'), g('reference_quaternion_x'), g('reference_quaternion_y'), g('reference_quaternion_z')])
    return out
res = {}; ADDS = {}
for lab, p in RUNS:
    D = load(p); pot = D['white_pot']; T = len(pot['frame'])
    up = q2m(pot['rq'][0]).T @ np.array([0, 0, 1.0])
    rel = np.zeros(T); absl = np.zeros(T); rot = np.zeros(T)
    for f in range(T):
        Ra = q2m(pot['aq'][f]); Rr = q2m(pot['rq'][f])
        rel[f] = np.degrees(np.arccos(np.clip((Ra @ up) @ (Rr @ up), -1, 1)))
        absl[f] = np.degrees(np.arccos(np.clip((Ra @ up)[2], -1, 1)))
        rot[f] = np.degrees(np.arccos(np.clip((np.trace(Rr.T @ Ra) - 1) / 2, -1, 1)))
    first = int(np.argmax(rel > 10)) if (rel > 10).any() else None
    runs10 = []
    on = rel > 10; s = None
    for f in range(T + 1):
        v = on[f] if f < T else False
        if v and s is None: s = f
        if not v and s is not None: runs10.append([s, f - 1]); s = None
    lifted_roll = int((pot['ap'][:, 2] - pot['ap'][0, 2] > 0.02).sum()); lifted_ref = int((pot['rp'][:, 2] - pot['rp'][0, 2] > 0.02).sum())
    r = dict(parquet=p, frames=T, first_rel_tilt_gt10=first, tilt_gt10_runs=runs10, n_frames_tilt_gt10=int(on.sum()),
             max_rel_tilt_f0_50=round(float(rel[:51].max()), 1), argmax_f0_50=int(rel[:51].argmax()),
             rel_tilt_at=[round(float(rel[f]), 1) for f in (0, 2, 5, 10, 20, 35, 50)],
             final_rel_tilt=round(float(rel[-1]), 1), final_abs_tilt=round(float(absl[-1]), 1), ref_final_abs_tilt=None,
             mean_rel_tilt_f289_312=round(float(rel[289:].mean()), 1), lifted_roll=lifted_roll, lifted_ref=lifted_ref)
    Rr_end = q2m(pot['rq'][-1]); r['ref_final_abs_tilt'] = round(float(np.degrees(np.arccos(np.clip((Rr_end @ up)[2], -1, 1)))), 1)
    # own-ref ADD (mesh vertices) for the bob1-mesh runs
    r['pot_body_name'] = pot['name']
    if '_A1' in pot['name']:
        adds = {}
        for short in ('white_pot', 'wooden_spoon'):
            Dd = D[short]; V = MESH[short]; a = np.zeros(T)
            for f in range(T):
                a[f] = np.linalg.norm(V @ q2m(Dd['aq'][f]).T + Dd['ap'][f] - V @ q2m(Dd['rq'][f]).T - Dd['rp'][f], axis=1).mean()
            adds[short] = a
        ADDS[lab] = (adds, on)
        r['ownref_auc_pot'] = round(auc(adds['white_pot']), 4); r['ownref_auc_spoon'] = round(auc(adds['wooden_spoon']), 4)
        r['ownref_auc'] = round((r['ownref_auc_pot'] + r['ownref_auc_spoon']) / 2, 4)
        r['pot_ADD_cm_mean_tipped_frames'] = round(float(adds['white_pot'][on].mean()) * 100, 2) if on.any() else None
        r['pot_ADD_cm_mean_untipped_frames'] = round(float(adds['white_pot'][~on].mean()) * 100, 2)
        jp = os.path.join(os.path.dirname(p), 'episode_000003.json')
        if os.path.exists(jp):
            jm = json.load(open(jp))['metrics']; r['json_add_auc'] = round(jm['add_auc'], 4); r['json_per_body'] = [round(x, 4) for x in jm['add_auc_per_body']]
    res[lab] = r
    print(lab, json.dumps(r), flush=True)
# attribution: replace tipped-frame pot ADD by the submitted s42 ADD at the same frames
base = [l for l in ADDS if l.endswith('h5A1_s42')]
if base:
    b = ADDS[base[0]][0]['white_pot']
    for lab, (adds, on) in ADDS.items():
        a = adds['white_pot'].copy(); cf = a.copy(); cf[on] = b[on]
        res[lab]['tip_attrib_pot_auc_loss'] = round(auc(cf) - auc(a), 4)
        res[lab]['tip_attrib_episode_loss'] = round((auc(cf) - auc(a)) / 2, 4)
        res[lab]['episode_gap_to_s42_ownref'] = round(res[base[0]]['ownref_auc'] - res[lab]['ownref_auc'], 4)
        print('ATTRIB', lab, res[lab]['tip_attrib_episode_loss'], 'gap to s42', res[lab]['episode_gap_to_s42_ownref'], flush=True)
json.dump(res, open(OUT, 'w'), indent=1)
print('DONE', OUT)
