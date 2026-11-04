"""lead_v3 overlay for ep28: the white_pot mesh+track replaced by the v8 winner (ep05 donor mesh, continuity-aware
recovery, proxy 0.698 -> 0.713), the table re-seated to the NEW objects, the team raw hands exclusively re-attached to the
NEW pot (or the delivered hands if the re-attach loses contact), all written into ONE bundle/ + support_v1/ + scene/ in
the lead_v3 layout. Sources: the A100 mini stage pulled to a100_mini_stage/. Writes stage/ + UPLOAD_DRY_RUN.json."""
from pathlib import Path
import json, shutil, hashlib, datetime, csv
import numpy as np
ROOT=Path('/Volumes/E/Code/V2D'); W=Path(__file__).resolve().parent; S=W/'stage'; A=W/'a100_mini_stage'
V3=ROOT/'work/lead-v3-release-20261104'; V3S=V3/'stage'; V1S=ROOT/'work/lead-sprint-release-20261103/stage'
REPO='rose-tea-cafe/v2d-t3-recon'; PREFIX='experiments/wentacc/scored20_lead_v3_ep28pot_20261104'
V3_COMMIT=json.loads((V3/'UPLOAD_COMMIT.json').read_text())['commit']; V3_PREFIX=json.loads((V3/'UPLOAD_COMMIT.json').read_text())['prefix']
EP=28; OBJECTS=['plastic_dish_rack','white_pot']; SLOT=1; TOL=0.05
WIN=dict(case='ep28_white_pot__recovery_cont_donor05',package='lead_retrack_recovery_v8',donor_mesh='ep05 white_pot (fb11)',proxy_before=0.6981,proxy_after=0.7126,rule='strict full-frame proxy gain + symmetry-aware motion sanity; 0 re-registrations; no GT')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
LOCAL=['/root/v2d/lead_winners_v8/mini_stage','/root/v2d',str(ROOT)]
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
def contact(mode):
    f=A/'hand_stage/scenes'/f'ep{EP:02d}_{mode}'/'HAND_CONTACT.json'
    return {strip_tag(k):v for k,v in read(f)['objects'].items()} if f.exists() else None
if S.exists(): shutil.rmtree(S)
sd=S/'scenes'/f'episode_{EP:06d}'; sd.mkdir(parents=True)
ctrl=contact('control'); ex=contact('rawexcl'); assert ctrl and ex
worst=0.0; detail=[]
for obj,c in ctrl.items():
    for side in ('left','right'):
        cv=(c.get(side) or {}).get('within_1cm'); ev=((ex.get(obj) or {}).get(side) or {}).get('within_1cm')
        if cv is None or ev is None: continue
        detail.append(f'{obj}/{side} {cv:.2f}->{ev:.2f}')
        if cv>=0.3: worst=min(worst,ev-cv)
chosen='rawexcl' if worst>=-TOL else 'control'
reason=('exclusive re-attach to the NEW pot kept every contact hand (worst %.2f)'%worst) if chosen=='rawexcl' else ('delivered hands better on the NEW pot (worst %.2f)'%worst)
srcb=A/'hand_stage/bundles'/f'ep{EP:02d}_{chosen}'; srcs=A/'hand_stage/scenes'/f'ep{EP:02d}_{chosen}'; assert srcb.exists() and (srcs/'OUTPUT_CHECK.json').exists()
shutil.copytree(srcb,sd/'bundle',symlinks=False,ignore=shutil.ignore_patterns('HANDFIX_PREP.json','ATTACH_EXCLUSIVE.json'))
for f in list((sd/'bundle').rglob('*.json')): put(f,relocal(read(f)))
# objects: slot 1 (white_pot) differs from v3 by construction (new mesh + track); slot 0 and camera identical to v3; hands per chosen
a=np.load(sd/'bundle/result.npz',allow_pickle=False); b=np.load(V3S/f'scenes/episode_{EP:06d}/bundle/result.npz',allow_pickle=False); cand=np.load(A/f'scenes/episode_{EP:06d}/bundle/result.npz',allow_pickle=False)
assert np.allclose(a['camera_to_world_transform'],b['camera_to_world_transform'],atol=1e-6) and np.allclose(a['object_to_world_transform'],b['object_to_world_transform'],atol=1e-6),'camera/slot0 changed'
assert np.allclose(a['object_1_to_world_transform'],cand['object_1_to_world_transform'],atol=1e-6),'slot1 is not the winner track'
assert not np.allclose(a['object_1_to_world_transform'],b['object_1_to_world_transform'],atol=1e-3),'slot1 unchanged vs v3?'
assert sha(sd/'bundle/mesh_1.obj')==sha(A/f'scenes/episode_{EP:06d}/bundle/mesh_1.obj') and sha(sd/'bundle/mesh.obj')==sha(V3S/f'scenes/episode_{EP:06d}/bundle/mesh.obj')
hands_changed=[k for k in b.files if k.startswith('hand_') and k in a.files and not np.array_equal(a[k],b[k])]
shutil.copytree(A/'seated_out'/f'episode_{EP:06d}'/'support_v1',sd/'support_v1')
spec=read(sd/'support_v1/support_in_bundle_world.json')
for n in list(spec.get('bundle_sha256',{})):
    if n.endswith('.npz'): spec['bundle_sha256'][n]=sha(sd/'bundle'/n)
put(sd/'support_v1/support_in_bundle_world.json',relocal(spec))
shutil.copytree(srcs,sd/'scene',symlinks=False,ignore=shutil.ignore_patterns('*.generated_by_loader'))
for f in list((sd/'scene').rglob('*.json')): put(f,relocal(read(f)))
for f in list((sd/'scene').rglob('*.log')):
    t=f.read_text()
    for lr in LOCAL: t=t.replace(lr,'<local>')
    f.write_text(t)
chk=read(srcs/'OUTPUT_CHECK.json'); ct=contact(chosen)
holding={f'{o}/{s}':round(v[s]['within_1cm'],2) for o,v in ct.items() for s in ('left','right') if v.get(s) and v[s]['within_1cm']>=0.3}
seat=read(A/'seated_out/SEATING_SUMMARY.json') if (A/'seated_out/SEATING_SUMMARY.json').exists() else None
rec=dict(episode=EP,objects=OBJECTS,objects_version='lead_v3 objects with white_pot (slot 1) replaced by the v8 winner',objects_changed_vs_v3=['white_pot'],winner=WIN,
         hands=dict(source='team raw hands + loader clean + exclusive re-attach to the new pot' if chosen=='rawexcl' else 'fb11 delivered hands',rule=reason,contact_detail=detail,holding_hands_within_1cm=holding,hand_arrays_changed_vs_v3=hands_changed),
         support='re-seated to the new objects (frame-0 penetration 0)',seating=relocal(seat),scene_check={k:chk[k] for k in ['task_error_cm','penetration_frames','frame0_seating_cm']},policy_tested=False,
         parent=dict(prefix=V3_PREFIX,commit=V3_COMMIT,note='every other episode: use lead_v3 as is'))
put(sd/'PROVENANCE.json',rec); put(sd/'CONTROL_HAND_CONTACT.json',relocal(read(A/'hand_stage/scenes'/f'ep{EP:02d}_control'/'HAND_CONTACT.json'))); put(sd/'CONTROL_OUTPUT_CHECK.json',relocal(read(A/'hand_stage/scenes'/f'ep{EP:02d}_control'/'OUTPUT_CHECK.json')))
(S/'tools').mkdir(); [shutil.copy2(V3S/'tools'/t,S/'tools'/t) for t in ['install_support.py','relocate_scene.py']]
readme=f'''# lead_v3 覆盖：ep28 白锅换 mesh+轨迹（2026-11-04）

只有 ep28 一段，其余 19 段继续用 lead_v3（`{V3_PREFIX}`，提交 {V3_COMMIT[:8]}）。布局与 lead_v3 相同：一个 `bundle/`（物体 + 匹配的手已写进 `result.npz`）+ 重新坐实的 `support_v1/` + 对应 `scene/`。

- 物体：white_pot（slot 1）换成 v8 GPU 批次的赢家 `{WIN['case']}`：用 ep05 的白锅 mesh 在 ep28 上重新配准 + 连续性保护的 recovery 跟踪，全帧代理 {WIN['proxy_before']:.3f} → {WIN['proxy_after']:.3f}（+{WIN['proxy_after']-WIN['proxy_before']:.4f}，运动规则通过，0 次重配准）。plastic_dish_rack 与相机和 lead_v3 逐元素一致（脚本断言）。
- 桌面：按新物体重新坐实（规则同 v2：桌面 = 物体最低点 − 2 mm，只降不升）。
- 手：{'团队原始双手 → loader 清理平滑 → 排他式贴到新白锅' if chosen=='rawexcl' else 'fb11 原手（排他式重贴在新锅上接触变差，未采用）'}；握锅的手贴合率 {holding}；判定：{reason}。
- 代理指标无 GT；未训练、未探针（`policy_tested: false`）。校准上 +0.0145 的代理增益落在"≥0.01 → GT 同号率 ~100%（Public20 校准）"区间。

用法：`python tools/relocate_scene.py scenes/episode_000028/scene`，然后训练 `scene/`；或 `bundle/` + `tools/install_support.py`。
'''
(S/'README.md').write_text(readme)
put(S/'SUMMARY.json',dict(state='LEAD_V3_EP28_POT_OVERLAY',parent_prefix=V3_PREFIX,parent_commit=V3_COMMIT,episode=EP,hands=chosen,holding=holding,winner=WIN,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
ALLOWED=['.json','.csv','.md','.npz','.obj','.mtl','.png','.jpg','.py','.usda','.parquet','.stl','.urdf','.log','.txt']
files=[p for p in sorted(S.rglob('*')) if p.is_file()]
for p in files:
    assert p.suffix.lower() in ALLOWED,p
    if p.suffix in ['.json','.md','.csv'] and 'scene' not in p.parts: assert not any(s in p.read_text() for s in ['/Volumes/','/Users/','185.216.','223.109.','/root/']),('leak',p)
manifest=[dict(relative=str(p.relative_to(S)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
(S/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['relative']}\n" for r in manifest)); manifest.append(dict(relative='SHA256SUMS.txt',sha256=sha(S/'SHA256SUMS.txt'),bytes=(S/'SHA256SUMS.txt').stat().st_size))
put(W/'UPLOAD_DRY_RUN.json',dict(state='DRY_RUN_READY',repo=REPO,prefix=PREFIX,base_revision=V3_COMMIT,stage=str(S),files=manifest,bytes=sum(x['bytes'] for x in manifest),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
print('CHOSEN',chosen,reason,holding); print('DRY_RUN files',len(manifest),'bytes',sum(x['bytes'] for x in manifest))
