"""Per-episode hand/scene ledger: which scene should be trained on, with the evidence side by side.
Columns: objects version (lead_sprint_v1), hand sources available, robot-link contact on lifted frames (held hand,
<=1 cm fraction) for our control / rawexcl / kenexcl and for Ken's shipped fix scene, Edmond's bob1 source + tracking,
Ken's own-ref numbers, the gate decision and the recommended scene. Writes HAND_LEDGER.json / HAND_LEDGER.md."""
from pathlib import Path
import json, re
O=Path(__file__).resolve().parent
KEN=Path('/private/tmp/claude-501/-Volumes-E-Code-V2D/91f630bc-c49b-45dd-beda-9b4445d9a7be/scratchpad/ken_scenes/experiments/haisenk/hand_fixes_wentao_v1_20261103')
KSUM=json.loads((Path('/private/tmp/claude-501/-Volumes-E-Code-V2D/9e418383-d795-43bf-92a7-b5721bcd24f8/scratchpad/hf_head/experiments/haisenk/hand_fixes_wentao_v1_20261103/SUMMARY.json')).read_text())
html=Path('/Users/roboticssn/.claude/uploads/9e418383-d795-43bf-92a7-b5721bcd24f8/d78c5f8a-bob1_submission.html').read_text()
BOB={e['episode']:e for e in json.loads(re.search(r'const EPISODES = (\[.*?\]);',html).group(1))}
INV={int(r['episode']):r for r in json.loads(Path('/Volumes/E/Code/V2D/work/lead-sprint-release-20261103/stage/inventory.json').read_text())}
CHANGED={5:['wooden_spoon'],32:['wooden_spoon'],33:['blue_cup','wooden_spoon']}
TOL=0.05
def strip(k):
    for pre in ('control_','rawhands_','rawclean_','rawexcl_','kenclean_','kenexcl_'):
        if k.startswith(pre): return k[len(pre):]
    return k
def contact(path):
    if not path.exists(): return None
    d=json.loads(path.read_text())['objects']; out={}
    for k,v in d.items():
        name=strip(k); m=re.search(r'(white_pot_lid|white_pot|wooden_spoon|blue_cup|beige_cup|water_pitcher|plastic_dish_rack|mini_sweeper|mini_dust_pan|wooden_piece_1|wooden_piece_2)$',name)
        out[m.group(1) if m else name]=v
    return out
def held(c):
    """(object, hand) pairs that are in contact: within_1cm >= 0.3 in this scene."""
    return {(o,s):v[s]['within_1cm'] for o,v in c.items() for s in ('left','right') if v.get(s) and v[s]['within_1cm']>=0.3}
def gate(ctrl,cand):
    if ctrl is None or cand is None: return None
    worst=0.0
    for (o,s),cv in held(ctrl).items():
        ev=(cand.get(o,{}).get(s) or {}).get('within_1cm')
        if ev is None: continue
        worst=min(worst,ev-cv)
    return worst
rows=[]; md=['| 段 | 物体版本 | bob1 用谁 · tracking | Ken own-ref（修复 / 对照） | 接触 对照 | 接触 rawexcl | 接触 kenexcl | 接触 Ken 场景 | 门槛判定 | 建议训练用 |','|---|---|---|---|---|---|---|---|---|---|']
fmt=lambda c:'—' if c is None else '；'.join(f'{o}/{s[0]} {v:.2f}' for (o,s),v in sorted(held(c).items())) or '无接触手'
for ep in sorted(INV):
    c=contact(O/'scenes'/f'ep{ep:02d}_control'/'HAND_CONTACT.json'); rx=contact(O/'scenes'/f'ep{ep:02d}_rawexcl'/'HAND_CONTACT.json'); kx=contact(O/'scenes'/f'ep{ep:02d}_kenexcl'/'HAND_CONTACT.json'); ks=contact(KEN/f'episode_{ep:06d}'/'scene'/'HAND_CONTACT.json')
    g_rx=gate(c,rx); g_kx=gate(c,kx)
    k=KSUM.get(f'episode_{ep:06d}',{}); kfix=', '.join(f'{v:.2f}' for v in k.get('fix_runs',{}).values()) if k else ''; kctl=', '.join(f'{v:.2f}' for v in k.get('controls_as_delivered',{}).values()) if k else ''
    b=BOB.get(ep,{})
    objv='v1 改了 '+'/'.join(CHANGED[ep]) if ep in CHANGED else 'fb11 = v1'
    # recommendation
    if ep in CHANGED:
        rec='ours: rawexcl' if (g_rx is not None and g_rx>=-TOL) else 'ours: control（v1 物体 + 坐实）'; why='物体是我们改的，Ken 的手贴在旧轨迹上'
    elif k:
        rec='Ken 场景（有 policy 证据）'; why=f'物体相同；Ken own-ref {kfix} 对 {kctl}'
        if kx is not None and g_kx is not None and g_kx>=-TOL: rec+='；备选 ours: kenexcl（Ken 修正手 + 排他式贴 + 坐实）'
    else:
        best=max([(g,n) for g,n in [(g_rx,'rawexcl'),(g_kx,'kenexcl')] if g is not None],default=None)
        rec=('ours: '+best[1]) if best and best[0]>=-TOL and best[1]=='rawexcl' and rx else 'ours: control（fb11 手 + 坐实）'; why='无人修手；按接触门槛'
    rows.append(dict(episode=ep,objects=objv,bob1=dict(source=b.get('source'),tracking=b.get('tracking'),expected_kaggle=b.get('expected_kaggle')),ken_ownref=dict(fix=k.get('fix_runs'),control=k.get('controls_as_delivered'),scene=k.get('scene')) if k else None,
        contact=dict(control=c,rawexcl=rx,kenexcl=kx,ken_scene=ks),gate=dict(rawexcl=g_rx,kenexcl=g_kx,tol=TOL),recommendation=rec,why=why))
    md.append(f"| {ep} | {objv} | {b.get('source','—')} · {b.get('tracking','—')} | {(kfix+' / '+kctl) if k else '—'} | {fmt(c)} | {fmt(rx)} | {fmt(kx)} | {fmt(ks)} | rawexcl {('%+.2f'%g_rx) if g_rx is not None else '—'}；kenexcl {('%+.2f'%g_kx) if g_kx is not None else '—'} | {rec} |")
(O/'HAND_LEDGER.json').write_text(json.dumps(rows,indent=1)+'\n'); (O/'HAND_LEDGER.md').write_text('\n'.join(md)+'\n'); print('\n'.join(md))
