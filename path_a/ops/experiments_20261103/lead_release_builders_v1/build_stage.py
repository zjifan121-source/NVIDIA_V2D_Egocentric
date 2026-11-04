"""Build the Hugging Face stage for the lead sprint v1 release: a NEW independent prefix whose direct base is the
fb11 Scored20 delivery. Only ep05 / ep32 / ep33 change (4 objects, mesh+track as a pair); every other bundle file,
all hand/camera arrays and all support_v1 files are byte-identical to the fb11 upload receipt. No GT, no HF access here.
Writes stage/ and UPLOAD_DRY_RUN.json. Re-runnable (stage is rebuilt from scratch)."""
from pathlib import Path
import json, csv, shutil, hashlib, datetime, sys
import numpy as np
ROOT=Path('/Volumes/E/Code/V2D'); OPS=ROOT/'NVIDIA_V2D_Egocentric/path_a/ops'; L=OPS/'experiments_20261103'
W=Path(__file__).resolve().parent; S=W/'stage'
REL=ROOT/'work/scored11-fulltrack-20261103/release'; LSB=L/'lead_sprint_bundle_v1'; RPT=ROOT/'work/reports/20261103-lead-sprint'
SCRIPTS=ROOT/'work/scored20-pipeline-release-20261102/baseline/reconstruction/scripts'; sys.path.insert(0,str(SCRIPTS/'scored20_pipeline')); import inputs
REPO='rose-tea-cafe/v2d-t3-recon'; PREFIX='experiments/wentacc/scored20_lead_sprint_v1_20261103'
FB11_PREFIX='experiments/wentacc/scored20_contactshape_proxy_mix_v1_20261103'
FB11='fb11a3ceaf3404d9e8d9c5adf85e68b724d8cc7c'; EXPECTED_PARENT='041d56dc5021445b47f6751c6cba2523a8da74da'
HAISEN_PREFIX='experiments/haisenk/hand_fixes_wentao_v1_20261103'
# Delivery threshold after the Public20 calibration (METRIC_CALIBRATION.json): ship only full-frame proxy gains >= 0.01
# that also passed the motion gate and a human review of the full-sequence overlay; +0.0023 gains are below the noise floor.
DELIVER={5:['wooden_spoon'],32:['wooden_spoon'],33:['blue_cup','wooden_spoon']}
DROPPED={10:[('blue_cup','lead_retrack_recovery_v6/ep10_blue_cup__recovery_cont_donor20',0.0023)],
         19:[('mini_sweeper','lead_retrack_recovery_v3/ep19_mini_sweeper__recovery_cont_own',0.0023),('mini_dust_pan','lead_retrack_recovery_v4/ep19_mini_dust_pan__recovery_cont_own',0.0023)]}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest(); read=lambda p:json.loads(Path(p).read_text())
def put(p,x): Path(p).parent.mkdir(parents=True,exist_ok=True); Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def equal(a,b): return a.shape==b.shape and np.array_equal(a,b)
fb11_receipt=read(ROOT/'work/scored11-fulltrack-20261103/UPLOAD_VERIFIED.json'); fb11_sha={r['relative']:r['sha256'] for r in fb11_receipt['sha256']}
assert fb11_receipt['commit']==FB11
if S.exists(): shutil.rmtree(S)
(S/'scenes').mkdir(parents=True)
inv=read(REL/'inventory.json'); proxy=read(LSB/'PROXY_SUMMARY.json'); selsum=read(LSB/'SELECTION_SUMMARY.json'); consumer={r['episode']:r for r in read(LSB/'CONSUMER_RESULTS.json')}
cases={r['case']:r for r in csv.DictReader(open(LSB/'ALL_CASES.csv'))}
def case_detail(source):
    batch,name=source.split('/'); sel=read(L/batch/'run_v1/SELECTION.json')['cases'][name]; fb=next(c for c in read(L/batch/'FROZEN_BATCH.json')['cases'] if c['case']==name)
    donor=None; mesh_hf=None
    if fb.get('mesh_source'):
        de=int(Path(fb['mesh_source']).parts[-3].split('_')[-1]); donor=de; mesh_hf=f"{FB11_PREFIX}/scenes/episode_{de:06d}/bundle/mesh.obj"
    return dict(batch=batch,case=name,variant=fb['variant'],init=fb.get('init','provided_pose'),anchor_frame=fb['anchor_frame'],
        mesh=dict(source='fb11 bundle of another Scored episode of the same physical object' if donor else 'own fb11 mesh',donor_episode=donor,hf_path=mesh_hf,sha256=fb['mesh_sha256']),
        proxy_before=sel['current']['full_proxy'],proxy_after=sel['candidate']['full_proxy'],delta=sel['delta'],
        iou_mean_before=sel['current']['iou_mean'],iou_mean_after=sel['candidate']['iou_mean'],depth_median_mm_before=sel['current']['depth_median_mean_mm'],depth_median_mm_after=sel['candidate']['depth_median_mean_mm'],
        motion_before=sel['current']['motion'],motion_after=sel['candidate']['motion'],rule_passed=sel['passed'],rule_strict_passed=sel['passed_strict'],
        re_registrations=len(sel.get('recovery_frames',[])),gpu_seconds=sel.get('gpu_seconds'),
        confidence='A: proxy gain >= 0.10 (Public20 calibration: 100% GT sign agreement)' if sel['delta']>=0.1 else 'B: proxy gain 0.01-0.10 (calibration: majority but not certain); accepted on full-sequence overlay review + motion gate')
rows=[]; changed_eps=sorted(DELIVER); objects_changed=0; video_index={}
HANDS=lambda z:[k for k in z.files if k.startswith('hand_')]+['camera_to_world_transform']
for r in inv:
    ep=r['episode']; objects=r['objects']; sd=S/'scenes'/f'episode_{ep:06d}'; src_ep=REL/'scenes'/f'episode_{ep:06d}'
    changed=ep in DELIVER; key=f'ep{ep:02d}'
    if changed:
        shutil.copytree(LSB/'scenes'/f'episode_{ep:06d}'/'bundle',sd/'bundle'); inputs.validate_scene(sd/'bundle')
    else:
        shutil.copytree(src_ep/'bundle',sd/'bundle')
        for p in sd.rglob('*'):
            if p.is_file(): assert fb11_sha.get(str(p.relative_to(S)))==sha(p),('unchanged bundle must equal fb11 receipt',ep,p.name)
    if (src_ep/'support_v1').exists():
        shutil.copytree(src_ep/'support_v1',sd/'support_v1')
        for p in (sd/'support_v1').iterdir(): assert fb11_sha.get(str(p.relative_to(S)))==sha(p),('support must equal fb11',ep,p.name)
    old=np.load(src_ep/'bundle/result.npz',allow_pickle=False); new=np.load(sd/'bundle/result.npz',allow_pickle=False)
    for k in HANDS(old): assert k in new.files and equal(old[k],new[k]),('hands/camera must be exact',ep,k)
    changed_fields=[k for k in old.files if k in new.files and not equal(old[k],new[k])]; added_fields=sorted(set(new.files)-set(old.files))
    assert (changed_fields!=[])==changed and all(k.startswith('object') for k in changed_fields+added_fields),(ep,changed_fields,added_fields)
    rec=dict(episode=ep,frames=int(old['camera_to_world_transform'].shape[0]),objects=objects,base=dict(repo=REPO,prefix=FB11_PREFIX,revision=FB11),
             selection='lead_sprint_v1_object_update' if changed else 'fb11_preserved_byte_identical',
             previous_full_proxy=proxy['episodes'].get(key,{}).get('before'),selected_full_proxy=(proxy['episodes'][key]['after'] if changed else proxy['episodes'].get(key,{}).get('before')),
             hand_camera_arrays_exact=True,support_v1_exact=True,changed_npz_fields=changed_fields,added_npz_fields=added_fields,
             official_AUC=None,policy_score=None,hand_accuracy_verified=False,physics_or_policy_tested=False)
    rec['delta']=(rec['selected_full_proxy']-rec['previous_full_proxy']) if rec['previous_full_proxy'] is not None else None
    if changed:
        rec['changed']=[]
        for c in selsum['episodes'][str(ep)]['changed']:
            assert c['object'] in DELIVER[ep]; d=case_detail(c['source']); d.update(object=c['object'],slot=c['slot']); rec['changed'].append(d); objects_changed+=1
        assert {c['object'] for c in rec['changed']}==set(DELIVER[ep])
        cr=consumer[ep]; rec['consumer']={k:cr[k] for k in ['schema','loader_exit_code','retarget_exit_code','support_exit_code','frames','physics_or_policy_tested']}
        assert cr['schema']=='FULL_NATIVE_CONSUMER_PASS' and cr['frames']==rec['frames']
        vsrc=LSB/'videos'/f'episode_{ep:06d}'; vr=read(vsrc/'VIDEO_RECEIPT.json'); assert vr['frames']==rec['frames']
        (sd/'videos').mkdir()
        for name,v in vr['files'].items(): shutil.copy2(vsrc/name,sd/'videos'/name); assert sha(sd/'videos'/name)==v['sha256']
        shutil.copy2(vsrc/'VIDEO_RECEIPT.json',sd/'videos'/'VIDEO_RECEIPT.json')
        rec['videos']=dict(before_after='videos/before_after.mp4',selected='videos/selected_overlay.mp4',frames=vr['frames'],left=vr['left'],right=vr['right'],overlay=vr['overlay'])
        video_index[ep]=rec['videos']
    if ep in DROPPED:
        rec['not_delivered_candidates']=[dict(object=o,source=s,full_proxy_gain=g,reason='below calibrated noise floor (< 0.01); fb11 bundle retained') for o,s,g in DROPPED[ep]]
    rec['fb11_videos']=f'{FB11_PREFIX}/scenes/episode_{ep:06d}/videos/ (revision fe07f83c)'
    put(sd/'run.json',rec)
    rows.append(dict(episode=ep,frames=rec['frames'],objects=objects,selection=rec['selection'],previous_full_proxy=rec['previous_full_proxy'],selected_full_proxy=rec['selected_full_proxy'],delta=rec['delta'],
                     changed_objects=DELIVER.get(ep,[]),hands_source='fb11 (byte-identical)',support_source='fb11 support_v1 (byte-identical)' if (sd/'support_v1').exists() else 'none in fb11'))
    print('STAGED',ep,'CHANGED' if changed else 'fb11',rec['delta'],flush=True)
scored=[r for r in rows if r['previous_full_proxy'] is not None]; assert len(scored)==19
mean_prev=float(np.mean([r['previous_full_proxy'] for r in scored])); mean_sel=float(np.mean([r['selected_full_proxy'] for r in scored]))
put(S/'inventory.json',rows)
with (S/'inventory.csv').open('w') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
# top-level documents
shutil.copy2(LSB/'ALL_CASES.md',S/'ALL_CASES.md'); shutil.copy2(LSB/'ALL_CASES.csv',S/'ALL_CASES.csv')
wrec=read(LSB/'WINDOW_RECEIPT.json'); wrec['host']='cloud A6000 host (address withheld)'; wrec['disk']='see local receipt'; put(S/'WINDOW_RECEIPT.json',wrec)
cal=read(RPT/'METRIC_CALIBRATION.json'); cal['source']='proxy_validation_v1/candidate_details.csv (Public20, 68 candidates, GT only in scoring)'; cal.pop('pairs',None); put(S/'METRIC_CALIBRATION.json',cal)
shutil.copy2(RPT/'metric_calibration.png',S/'metric_calibration.png')
(S/'tools').mkdir(); shutil.copy2(REL/'tools/install_support.py',S/'tools/install_support.py')
c33=cal['results']['dense33/all']['at_least']; c33x=cal['results']['dense33/excl_ep21']['at_least']
selection=dict(rule='full-frame proxy strictly higher; rot step max <= 90 deg; steps > 30 deg on <= 10% of frames; translation step p95 <= 50 mm and max <= 120 mm; no invalid frames; cups: axis tilt for rotation, centroid for translation (FROZEN_BATCH amendments of lead_retrack_recovery_v1..v7)',
    delivery_threshold='full-frame proxy gain >= 0.01 AND rule passed AND full-sequence overlay reviewed by a human; gains < 0.01 are not shipped (Public20 calibration)',
    delivered={str(k):v for k,v in DELIVER.items()},not_delivered={str(k):[dict(object=o,source=s,gain=g) for o,s,g in v] for k,v in DROPPED.items()},
    cases_evaluated=len(cases),cases_passed_rule=sum(1 for c in cases.values() if c['passed']=='True'),
    calibration_public20=dict(pairs_all=cal['results']['dense33/all']['pairs'],pairs_excl_ep21=cal['results']['dense33/excl_ep21']['pairs'],
        sign_agreement_at_least_0p1=c33['0.1']['sign_agreement'],sign_agreement_0p01_all=c33['0.01']['sign_agreement'],sign_agreement_0p01_excl_ep21=c33x['0.01']['sign_agreement'],
        harm_rate_0p01_excl_ep21=c33x['0.01']['harm_rate_auc_drop_gt_0p02'],harm_rate_0p05_excl_ep21=c33x['0.05']['harm_rate_auc_drop_gt_0p02'],spearman_all_candidates=cal['results']['dense33/spearman_all_candidates']))
put(S/'SELECTION.json',selection)
summary=dict(state='LEAD_SPRINT_V1_OBJECT_UPDATE_OVER_FB11_NOT_POLICY_VALIDATED',episodes=20,objects=36,changed_episodes=changed_eps,changed_objects=objects_changed,
    retained_episodes=[r['episode'] for r in rows if r['episode'] not in DELIVER],scored_episodes=19,unscored_episodes=[38],mean_previous=mean_prev,mean_selected=mean_sel,
    baseline=dict(repo=REPO,prefix=FB11_PREFIX,revision=FB11,note='fe07f83c added videos/docs only; bundle bytes unchanged'),
    hands='byte-identical to fb11 in all 20 episodes; NOT re-attached to the new object tracks of ep05/ep32/ep33',support_v1='byte-identical to fb11; table seating issue reported by haisenk applies unchanged',
    related=dict(haisenk_hand_fixes=dict(prefix=HAISEN_PREFIX,revision=EXPECTED_PARENT,episodes=[1,3,6,10,14,20,32],note='object arrays identical to fb11 except ep3 pot shifted 0.5 cm; only hand arrays differ')),
    selection_basis='same-observation full-trajectory fixed_observation_proxy_v1 with motion gate and calibrated delivery threshold; no GT',
    gpu=dict(container_hours=wrec['container_wall_hours'],cases=len(cases),failed_final=wrec['failed_final']),official_AUC=None,policy_score=None,
    limitations=['Proxy is not the official metric; Public20 calibration: reliable only for gains >= 0.1, majority-correct for 0.01-0.1, noise below 0.01.',
                 'Hands and camera retained exactly; hand-object consistency during handling changes with the new spoon tracks and must be re-attached downstream.',
                 'ep05/ep33 spoon meshes come from ep36, ep32 from ep37 (same physical object, own reconstructions): CD-O inputs for those scenes change accordingly.',
                 'Seven fb11 bundles (ep01/03/06/14/15/19/35) reference a material file that is absent on HF; retained as delivered (downstream loader repairs it).'],
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
put(S/'SUMMARY.json',summary)
# README
def pct(x): return f'{x:.6f}'
table=''
for r in rows:
    ep=r['episode']; q=r['previous_full_proxy']
    if ep in DELIVER:
        objs='、'.join(DELIVER[ep]); table+=f"| {ep} | {pct(q)} | {pct(r['selected_full_proxy'])} | {r['delta']:+.6f} | 采用（{objs}） | [完整对照](scenes/episode_{ep:06d}/videos/before_after.mp4) · [交付版overlay](scenes/episode_{ep:06d}/videos/selected_overlay.mp4) |\n"
    elif ep in DROPPED:
        table+=f"| {ep} | {pct(q)} | {pct(q)} | 0 | 保留 fb11（有 +0.0023 的通过 case，低于噪声下限，不交付） | — |\n"
    else:
        table+=f"| {ep} | {pct(q) if q is not None else '未评分'} | {pct(q) if q is not None else '未评分'} | 0 | 保留 fb11 | — |\n"
det={(d['object'],ep):d for ep in DELIVER for d in read(S/'scenes'/f'episode_{ep:06d}'/'run.json')['changed']}
def fmt(obj,ep): d=det[(obj,ep)]; return f"{d['proxy_before']:.4f} → {d['proxy_after']:.4f} ({d['delta']:+.4f})"
readme=f'''# Scored20 lead sprint v1：ep05 / ep32 / ep33 的勺 mesh 替换与重跟踪（基于 fb11）

本目录是**新的独立前缀，只新增文件**。直接基线是同仓库 `{FB11_PREFIX}` 在提交 **{FB11[:8]}** 的完整 20 集交付（后续 fe07f83c 只补了视频和文档，bundle 字节未变）。fb11 目录原样保留。

本次只改 **3 集里 4 个物体的 mesh + 轨迹（成套）**。其余 17 集的 bundle、全部 20 集的双手/相机数组、所有 `support_v1/` 与 fb11 上传回执逐字节相同（构建脚本逐文件断言；每集 `run.json` 记录 `changed_npz_fields`）。

## 改了什么

| 集 | 物体 | 改法 | mesh 来源 | 物体全帧代理 |
|---|---|---|---|---:|
| 5 | wooden_spoon | 换 ep36 的勺 mesh，frame 0 注册后用连续性 recovery 跟踪 | fb11 ep36 `mesh.obj` | {fmt('wooden_spoon',5)} |
| 32 | wooden_spoon | 换 ep37 的勺 mesh，frame 1 注册后同上（13 次受限重注册） | fb11 ep37 `mesh.obj` | {fmt('wooden_spoon',32)} |
| 33 | blue_cup | 自家 mesh，从末帧反向 recovery 跟踪 | fb11 ep33 `mesh.obj` | {fmt('blue_cup',33)} |
| 33 | wooden_spoon | 换 ep36 的勺 mesh，frame 0 注册后连续性 recovery 跟踪。原 Ken 勺 mesh 主轴 278 mm，比其他段的同一把勺短 12%，被手拿起后即丢失 | fb11 ep36 `mesh.obj` | {fmt('wooden_spoon',33)} |

19 个可计分集的同口径代理均分：**{mean_prev:.8f} → {mean_sel:.8f}**（{mean_sel-mean_prev:+.6f}）。ep38 无兼容观测不计分，保留 fb11。

**为什么只有这四个。** 一个 GPU 窗口（A6000，{wrec['container_wall_hours']} 容器小时）跑了 7 个批次、{len(cases)} 个评价 case（`ALL_CASES.md` / `ALL_CASES.csv`）：plain recovery、粒子滤波、连续性 recovery、末帧反向、donor mesh 用在杯类和 ep03/06/15/16/35/37 的勺上，都是 ±0.003 的噪声或退步；实质增益只来自替换本身有问题的勺 mesh。手-物刚性先验（腕位姿 × 常量抓握偏移）在 ep33 上只到 0.14 IoU，否定。ep10 蓝杯与 ep19 扫帚/簸箕各有 +0.0023 的通过 case，**低于代理噪声下限，不交付**。ep10 蓝杯 0.57、ep28 锅 0.70、ep19 扫帚 0.72 仍未突破。

| Episode | fb11 | 本次 | Δ | 最终选择 | 视频 |
|---|---:|---:|---:|---|---|
{table}
## 选择规则与代理校准

规则（预注册，修订两次并记录在各批次 FROZEN_BATCH amendments）：全帧代理严格更高；最大旋转步 ≤ 90°；>30° 的步 ≤ 10% 帧；平移步 p95 ≤ 50 mm、最大 ≤ 120 mm；无无效帧；杯类按轴倾斜计旋转、按质心计平移。{len(cases)} 个 case 中 {selection['cases_passed_rule']} 个通过，每个物体取代理最高的通过 case。无 GT。

代理仍是冻结的 `fixed_observation_proxy_v1 / proxy_auc_101`：q = 0.65 × mask IoU + 0.35 × exp(−重叠处深度中位残差 / 0.03 m)，101 阈值积分，全帧、同集物体等权、19 集等权。**它不是官方 AUC。**

用 Public20 的 68 个历史完整候选（有 GT，仅用于评分）做了成对校准（`METRIC_CALIBRATION.json`、`metric_calibration.png`）：同集两个候选的代理差 **≥ 0.1** 时 GT AUC 方向 **100% 一致**（{c33['0.1']['n']}/{c33['0.1']['n']} 对）；差在 **0.01–0.1** 时，排除 ep21（一集 18 个实验变体）后一致率 {c33x['0.01']['sign_agreement']*100:.0f}–{c33x['0.05']['sign_agreement']*100:.0f}%、GT 退步 >0.02 的概率 {c33x['0.05']['harm_rate_auc_drop_gt_0p02']*100:.0f}–{c33x['0.01']['harm_rate_auc_drop_gt_0p02']*100:.0f}%，含 ep21 则只有 {c33['0.01']['sign_agreement']*100:.0f}–{c33['0.05']['sign_agreement']*100:.0f}%；差 **< 0.01** 无法区分。候选间 Spearman 仅 {cal['results']['dense33/spearman_all_candidates']:.2f}。因此：ep33 勺（+0.20）在可靠区；ep05/ep32 勺与 ep33 杯（+0.018–0.046）在"多半对"区，额外依据是全段叠加视频人工复核与运动边界；+0.0023 的不交付。交付门槛由此定为 0.01。

## 与 haisenk 手部修复的关系（重要）

`{HAISEN_PREFIX}`（提交 {EXPECTED_PARENT[:8]}）在 fb11 上重做了 ep1/3/6/10/14/20/32 的双手，并指出 fb11 的双手是贴在旧物体轨迹上的。本目录**没有动双手**：

- ep05 / ep32 / ep33 的勺轨迹变了，这三集的双手需要用他们的工具（其前缀 `tools/wprep.py`、`tools/wbuild2.sh`）对新物体轨迹重新 attach，再建 scene；ep32 在他们的 7 集里，但他们的 ep32 双手是贴在 fb11 旧勺轨迹上的。
- 其余 6 集（1/3/6/10/14/20）本目录物体与 fb11 相同，直接用他们的 scene 即可（他们的 ep3 锅轨迹自己平移了 0.5 cm）。
- 他们还指出 fb11 的 `support_v1` 桌面在多集 frame 0 陷入或悬空物体 1–5 cm（ep19 5.3、ep30 3.6、ep20 3.3、ep32 2.6、ep14 1.6、ep6 1.5、ep28 1.4、ep5 1.4、ep37 1.2 cm）。本目录的 support_v1 与 fb11 相同；**训练前建议按他们的 `wseat.py` 坐实桌面**。

## 下载与使用

每集 `scenes/episode_XXXXXX/bundle/` 是完整原生场景：`result.npz` 及匹配的米制 OBJ/MTL/纹理。**不要把新 mesh 和旧 pose 混用**（ep05/ep32/ep33 的 `mesh_1.obj` 与 `object_1_*` 轨迹成套；ep33 的 `mesh.obj` 与 `object_*` 轨迹成套）。保留整个 bundle 目录，使用现有 loader，`result_subpath=.`；保持默认 1.2 手型映射及原训练配置做公平对照。ep32 的 npz 比 fb11 多了 `object_1_to_camera_transform` / `object_1_is_valid` 两个字段（旧 schema 补全，数值与世界系轨迹一致）。

有 `support_v1/` 的集，安装方法与 fb11 相同：

```bash
python tools/install_support.py \\
  --bundle scenes/episode_000005/bundle \\
  --parquet /your/processed/sequence_id=YOUR_ID/robot_name=sharpa_wave/YOUR_FILE.parquet \\
  --spec scenes/episode_000005/support_v1/support_in_bundle_world.json
```

提交 CD-O 时注意：ep05/ep33 的勺 mesh 现在是 ep36 的自家重建，ep32 的是 ep37 的（同一实物）。

## 核验

- 改动三集均通过本地原生 loader/retarget（退出码 0），support 重建退出 1 与此前交付一致；未做物理/policy 测试，双手精度未验证。
- 3 个完整前后对照 + 3 个交付版 overlay，H.264，逐帧数核对（各 `videos/VIDEO_RECEIPT.json`）。左 fb11、右本次；橙/蓝为两个物体的预测，绿色为改动物体的观测 mask；RGB 中的手是观测，不渲染重建手，不是物理回放。fb11 全部 20 集的视频在其前缀 `scenes/episode_XXXXXX/videos/`（fe07f83c）。
- GPU 窗口回执 `WINDOW_RECEIPT.json`（{wrec['container_wall_hours']} 容器小时；4 个最终失败 case 已记录）。
- `sha256sum -c SHA256SUMS.txt` 核对所有发布文件（自身除外）。
- 不是正式比赛提交；无官方分数、无 GT、无训练结果。
'''
(S/'README.md').write_text(readme)
files=[p for p in sorted(S.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt']
for p in files:
    if p.suffix in ['.json','.csv','.md','.py']:
        t=p.read_text(); assert not any(s in t for s in ['/Volumes/','/Users/','/mnt/v2d','185.216.']),('local path leak',p)
    assert p.suffix.lower() in ['.json','.csv','.md','.npz','.obj','.mtl','.png','.jpg','.jpeg','.py','.usda','.mp4'],p
manifest=[dict(relative=str(p.relative_to(S)),sha256=sha(p),bytes=p.stat().st_size) for p in files]
(S/'SHA256SUMS.txt').write_text(''.join(f"{r['sha256']}  {r['relative']}\n" for r in manifest))
manifest.append(dict(relative='SHA256SUMS.txt',sha256=sha(S/'SHA256SUMS.txt'),bytes=(S/'SHA256SUMS.txt').stat().st_size))
plan=dict(state='DRY_RUN_READY',repo=REPO,prefix=PREFIX,expected_parent=EXPECTED_PARENT,base_revision=FB11,stage=str(S),files=manifest,bytes=sum(x['bytes'] for x in manifest),
          changed_episodes=changed_eps,mean_previous=mean_prev,mean_selected=mean_sel,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
put(W/'UPLOAD_DRY_RUN.json',plan)
print('DRY_RUN files',len(manifest),'bytes',plan['bytes'],'mean',mean_prev,'->',mean_sel,'changed',changed_eps,flush=True)
