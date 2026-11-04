"""skb2b.py -- counterfactual 'support at the depth table' for the A1 rollouts: at rest frames put each rollout body
at reference z - SINK (sim sink ~0.5 cm, as observed), optionally also remove the start gap during the spoon pickup
(linear decay f36->f55). Kaggle proxy vs the A1 reference with AUC.py alignment. Reuses skb2.py functions."""
import sys, json
import numpy as np
sys.argv = [sys.argv[0], '/dev/null']
exec(open('skb2.py').read().split("res = {'support_top'")[0])
SINK = 0.005
REST = {0: list(range(0, 51)) + list(range(267, 313)), 1: list(range(0, 36)) + list(range(289, 313))}
out = {}
for lab in ('A1_s42_submitted', 'A1_s43', 'A1_s44'):
    P, Q, RP, RQ = rollout(RUNS[lab])
    for variant in ('rest_only', 'rest+pickup'):
        P2 = P.copy()
        for k, fr in REST.items():
            fr = np.array(fr)
            P2[fr, k, 2] = RP[fr, k, 2] - SINK
        if variant == 'rest+pickup':
            g = (RP[35, 1, 2] - SINK) - P[35, 1, 2]
            for t_ in range(36, 56):
                P2[t_, 1, 2] = P[t_, 1, 2] + g * (55 - t_) / 20.0
        a, rdeg, off = kaggle(P2, Q, REFP, REFQ, align=True)
        u, _, _ = kaggle(P2, Q, RP, RQ, align=False)
        base, _, _ = kaggle(P, Q, REFP, REFQ, align=True)
        r = dict(proxy=[round(auc(a[0]), 4), round(auc(a[1]), 4), round((auc(a[0]) + auc(a[1])) / 2, 4)],
                 base_proxy=[round(auc(base[0]), 4), round(auc(base[1]), 4), round((auc(base[0]) + auc(base[1])) / 2, 4)],
                 unaligned=[round(auc(u[0]), 4), round(auc(u[1]), 4), round((auc(u[0]) + auc(u[1])) / 2, 4)],
                 anchor_shift_cm=round(float(off[2] * 100), 3),
                 phase_add_kaggle={ph: [round(float(a[k][x:y + 1].mean() * 100), 2) for k in range(2)] for ph, x, y in PH})
        out[f'{lab}:{variant}'] = r
        print(lab, variant, json.dumps(r), flush=True)
json.dump(out, open('skb2b.json', 'w'), indent=1)
print('DONE')
