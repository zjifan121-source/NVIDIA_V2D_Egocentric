"""lead_v3: ONE self-consistent bundle per episode = lead_sprint_v1 objects (fb11 + the ep05/32/33 mesh+track replacements)
with the matched hands written INTO bundle/result.npz (no sub-variants), the seated support_v1, and the training scene
built from exactly that bundle. Hands per episode: the exclusive re-attach of the team raw hands when it kept every
contact hand within 0.05 of the delivered hands, otherwise the delivered (fb11) hands (ep05 by comparison; ep35/36/37/38
because HF has no raw hands for them). Objects/camera are asserted equal to v1 (world poses allclose 1e-6).
Writes stage/ + UPLOAD_DRY_RUN.json. Re-runnable."""
from pathlib import Path
import json, shutil, hashlib, datetime, sys, csv
import numpy as np
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'; L=OPS/'experiments_20261103'
W=Path(__file__).resolve().parent; S=W/'stage'; HAND=L/'lead_hand_reattach_v1'; SEAT=L/'lead_support_seating_v1'; V1S=ROOT/'work/lead-sprint-release-20261103/stage'
REPO='rose-tea-cafe/v2d-t3-recon'; PREFIX='experiments/wentacc/scored20_lead_v3_20261104'
V1_PREFIX='experiments/wentacc/scored20_lead_sprint_v1_20261103'; V1_COMMIT=json.loads((ROOT/'work/lead-sprint-release-20261103/UPLOAD_VERIFIED.json').read_text())['commit']
TOL=0.05
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
LOCAL=[str(HAND),str(SEAT),str(V1S),str(ROOT)]
def relocal(x):
    if isinstance(x,dict): return {k:relocal(v) for k,v in x.items()}
    if isinstance(x,list): return [relocal(v) for v in x]
    if isinstance(x,str):
        for r in LOCAL:
            if x.startswith(r): return '<local>'+x[len(r):]
    return x
def strip_tag(k):
    for pre in ('control_','rawhands_','rawclean_','rawexcl_','kenexcl_'):
        if k.startswith(pre): return k[len(pre):]
    return k
def contact(ep,mode,objects=None):
    f=HAND/'scenes'/f'ep{ep:02d}_{mode}'/'HAND_CONTACT.json'
    if not f.exists(): return None
    d={strip_tag(k):v for k,v in read(f)['objects'].items()}
    if objects and len(objects)==1 and len(d)==1: d={objects[0]:next(iter(d.values()))}  # single-object scenes are keyed by the mode tag only
    return d
if S.exists(): shutil.rmtree(S)
(S/'scenes').mkdir(parents=True)
inv=read(V1S/'inventory.json'); rows=[]; ledger=[]
for r in inv:
    ep=r['episode']; sd=S/'scenes'/f'episode_{ep:06d}'; v1b=V1S/f'scenes/episode_{ep:06d}/bundle'
    ctrl=contact(ep,'control',r['objects']); ex=contact(ep,'rawexcl',r['objects']); detail=[]; chosen='control'; reason='no raw hands on HF (ep35/36/37/38): delivered hands kept' if ex is None else ''
    if ex is not None:
        worst=0.0
        for obj,c in (ctrl or {}).items():
            for side in ('left','right'):
                cv=(c.get(side) or {}).get('within_1cm'); ev=((ex.get(obj) or {}).get(side) or {}).get('within_1cm')
                if cv is None or ev is None: continue
                detail.append(f'{obj}/{side} {cv:.2f}->{ev:.2f}')
                if cv>=0.3: worst=min(worst,ev-cv)
        chosen='rawexcl' if (ctrl is None or worst>=-TOL) else 'control'
        reason=('exclusive re-attach kept every contact hand (worst %.2f)'%worst if chosen=='rawexcl' else 'delivered hands better (worst %.2f)'%worst) if ctrl else 'no control contact data; re-attach used'
    srcb=HAND/'bundles'/f'ep{ep:02d}_{chosen}'; srcs=HAND/'scenes'/f'ep{ep:02d}_{chosen}'
    assert srcb.exists() and (srcs/'OUTPUT_CHECK.json').exists(),(ep,chosen)
    shutil.copytree(srcb,sd/'bundle',symlinks=False,ignore=shutil.ignore_patterns('HANDFIX_PREP.json','ATTACH_EXCLUSIVE.json'))
    for f in list(sd.rglob('*.json')): put(f,relocal(read(f)))
    # objects/camera identical to v1 (world poses), hands may differ
    a=np.load(sd/'bundle/result.npz',allow_pickle=False); b=np.load(v1b/'result.npz',allow_pickle=False)
    keys=[k for k in b.files if k.endswith('_to_world_transform') and not k.startswith('hand_')]
    assert all(np.allclose(a[k],b[k],atol=1e-6) for k in keys),(ep,'object/camera poses differ from v1')
    for m in [f for f in v1b.iterdir() if f.suffix in ('.obj','.mtl','.png')]: assert sha(sd/'bundle'/m.name)==sha(m),(ep,m.name,'mesh/material differs from v1')
    hands_changed=[k for k in b.files if k.startswith('hand_') and k in a.files and not np.array_equal(a[k],b[k])]
    shutil.copytree(SEAT/'out'/f'episode_{ep:06d}'/'support_v1',sd/'support_v1')
    spec=read(sd/'support_v1/support_in_bundle_world.json')
    for n in spec['bundle_sha256']:
        if n.endswith('.npz'): spec['bundle_sha256'][n]=sha(sd/'bundle'/n)  # hands changed the npz; objects did not (asserted above)
    put(sd/'support_v1/support_in_bundle_world.json',spec)
    shutil.copytree(srcs,sd/'scene',symlinks=False,ignore=shutil.ignore_patterns('*.generated_by_loader'))
    for f in list((sd/'scene').rglob('*.json')): put(f,relocal(read(f)))
    for f in list((sd/'scene').rglob('*.log')):
        t=f.read_text()
        for lr in LOCAL: t=t.replace(lr,'<local>')
        f.write_text(t)
    chk=read(srcs/'OUTPUT_CHECK.json'); ct=contact(ep,chosen,r['objects']) or {}
    holding={f'{o}/{s}':round(v[s]['within_1cm'],2) for o,v in ct.items() for s in ('left','right') if v.get(s) and v[s]['within_1cm']>=0.3}
    rec=dict(episode=ep,objects=r['objects'],objects_version='lead_sprint_v1 (= fb11 except the replaced meshes/tracks)' ,objects_changed_vs_fb11=r.get('changed_objects',[]),object_proxy=r.get('selected_full_proxy'),
             hands=dict(source='team raw hands + loader clean + exclusive re-attach' if chosen=='rawexcl' else 'fb11 delivered hands',rule=reason,contact_detail=detail,holding_hands_within_1cm=holding,hand_arrays_changed_vs_v1=hands_changed),
             support='seated (frame-0 penetration 0)',scene_check={k:chk[k] for k in ['task_error_cm','penetration_frames','frame0_seating_cm']},policy_tested=False)
    put(sd/'PROVENANCE.json',rec); ledger.append(rec)
    rows.append(dict(episode=ep,objects=';'.join(r['objects']),objects_changed=';'.join(r.get('changed_objects',[])),object_proxy=r.get('selected_full_proxy'),hands=chosen,holding_hands=' '.join(f'{k}={v}' for k,v in holding.items()),min_holding=min(holding.values()) if holding else None))
    print('V3',ep,chosen,holding,flush=True)
put(S/'inventory.json',rows)
with (S/'inventory.csv').open('w') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(S/'tools').mkdir();
for t in ['install_support.py']: shutil.copy2(V1S/'tools'/t,S/'tools'/t)
for t in ['relocate_scene.py']: shutil.copy2(ROOT/'work/lead-sprint-release-v2-20261103/stage/tools'/t,S/'tools'/t)
tab='| 段 | 物体（相对 fb11 的改动） | 物体代理 | 手 | 握物体的手贴合率 | 状态 |\n|---|---|---:|---|---|---|\n'
for x in rows:
    st='匹配' if x['min_holding'] is not None and x['min_holding']>=0.9 else ('基本匹配' if x['min_holding'] is not None and x['min_holding']>=0.8 else '手重建本身差')
    tab+=f"| {x['episode']} | {x['objects_changed'] or '同 fb11'} | {'—' if x['object_proxy'] is None else '%.3f'%x['object_proxy']} | {'重 attach' if x['hands']=='rawexcl' else 'fb11 原手'} | {x['holding_hands']} | {st} |\n"
readme=f'''# lead_v3：一段一个 bundle，物体 + 手 + 桌面自洽（2026-11-04）

**这是 Wentao 线当前唯一应该拿去训练/探针的版本。** 每段只有一个 `bundle/`（物体 = lead_sprint_v1，即 fb11 加上 ep05/32/33 换过的勺 mesh 与轨迹；手 = 与这些物体匹配过的手，已写进 `result.npz`；桌面 = 坐实后的 `support_v1/`），以及用这个 bundle 构建好的 `scene/`（团队 loader，含坐实桌面 USD）。没有子变体，不需要再选目录。

## 版本关系（为什么之前乱）

| 名字 | 内容 | 用途 |
|---|---|---|
| fb11（`scored20_contactshape_proxy_mix_v1_20261103`） | Wentao 旧交付：物体 + 贴在更旧轨迹上的手 + 未坐实桌面 | 历史基线；bob1 里"Wentao"的 6 段来自它 |
| lead_sprint_v1 | fb11 + ep05/32/33 换 mesh/轨迹；手、桌面未动 | 物体层更新 |
| lead_sprint_v2（+追加） | v1 物体 + 坐实桌面 + 重 attach 的手，但手在 `bundle_rawexcl/`、`scene/` 子目录里，`bundle/` 仍是旧手 | 过渡版，容易拿错 |
| **lead_v3（本目录）** | v1 物体 + 匹配的手写进 `bundle/` + 坐实桌面 + 对应 scene | **训练与探针用这个** |
| Ken / Haisen | Ken 自己的重建；Haisen 在 fb11 上修的 7 段手 | bob1 的其余 14 段来源 |
| bob1 | Edmond 按段混选（Ken 11、Haisen 3、Wentao fb11 6）的提交 | 当前 Kaggle 最好 0.752；探针的基线 |

## 逐段

{tab}
贴合率 = 物体被举起时重定向后的机器人手离物体 ≤1 cm 的帧占比（1.0 = 一直贴着）。"手重建本身差"的段（ep10/16/35/36/37）attach 贴不上，需要重新重建手，不在本版范围。

## 用法

```bash
python tools/relocate_scene.py scenes/episode_0000XX/scene   # 把 parquet/URDF 里的构建机路径指到你的副本
# 训练直接用 scene/；若要自己构建，用 bundle/ + tools/install_support.py 装 support_v1
```

物体与相机数组与 lead_sprint_v1（提交 {V1_COMMIT[:8]}）逐元素一致（构建脚本断言）；每段 `PROVENANCE.json` 记录手的来源、规则、贴合率与 loader 检查。无 GT、无训练、无官方分数。
'''
(S/'README.md').write_text(readme)
put(S/'SUMMARY.json',dict(state='LEAD_V3_SELF_CONSISTENT_BUNDLES',objects='lead_sprint_v1',hands='matched (exclusive re-attach or delivered, per PROVENANCE)',support='seated',episodes=20,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),ledger=ledger))
ALLOWED=['.json','.csv','.md','.npz','.obj','.mtl','.png','.jpg','.py','.usda','.parquet','.stl','.urdf','.log','.txt']
files=[p for p in sorted(S.rglob('*')) if p.is_file()]
for p in files:
    assert p.suffix.lower() in ALLOWED,p
    if p.suffix in ['.json','.md','.csv'] and 'scene' not in p.parts: assert not any(s in p.read_text() for s in ['/Volumes/','/Users/','185.216.']),('leak',p)
manifest=[dict(relative=str(p.relative_to(S)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
(S/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['relative']}\n" for r in manifest)); manifest.append(dict(relative='SHA256SUMS.txt',sha256=sha(S/'SHA256SUMS.txt'),bytes=(S/'SHA256SUMS.txt').stat().st_size))
put(W/'UPLOAD_DRY_RUN.json',dict(state='DRY_RUN_READY',repo=REPO,prefix=PREFIX,base_revision=V1_COMMIT,stage=str(S),files=manifest,bytes=sum(x['bytes'] for x in manifest),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print('DRY_RUN files',len(manifest),'bytes',sum(x['bytes'] for x in manifest))
