"""Single source of truth for 'which scene and which rollout is ours for each scored episode'.
Writes CURRENT_MANIFEST.json + VERSION_LEDGER.md from verified facts only. Re-run whenever a result arrives."""
import json, datetime
from pathlib import Path
W=Path(__file__).resolve().parent
V3='experiments/wentacc/scored20_lead_v3_20261104'; V3C='5e7601d1'
P28='experiments/wentacc/scored20_lead_v3_ep28pot_20261104'
KEN='experiments/haisenk/hand_fixes_wentao_v1_20261103'; KENC='041d56dc'
V2='experiments/wentacc/scored20_lead_sprint_v2_20261103'
facts=json.loads((W/'facts.json').read_text())
BOB1=facts['bob1']; PROBES=facts['probes']; EPS=facts['episodes']
def delta(ep):
    rows=[p for p in PROBES if p['episode']==ep]
    return rows
rows=[]
for e in EPS:
    ep=e['episode']; pr=delta(ep)
    rows.append(dict(episode=ep,scene_path=e.get('scene_path'),candidates=e.get('candidates'),bob1_source=BOB1[str(ep)]['who'],bob1_scene=BOB1[str(ep)]['scene'],bob1_own_ref=BOB1[str(ep)]['own_ref'],
        objects_vs_bob1=e['objects_vs_bob1'],current_scene=e['current_scene'],hands=e['hands'],status=e['status'],probes=pr,best_true_delta=max([p['true_delta'] for p in pr],default=None)))
man=dict(schema='lead_current_manifest_v1',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    rule='Read the highest-numbered manifest, not prefix names. Each episode has one current scene, except where candidates are listed: those stay open until a Kaggle probe decides. Old prefixes stay on HF untouched.', known_issues=dict(lead_v3='experiments/wentacc/scored20_lead_v3_20261104/KNOWN_ISSUES.md', ep28_overlay='experiments/wentacc/scored20_lead_v3_ep28pot_20261104/KNOWN_ISSUE_left_hand_false_grasp.md', ep28_overlay_commit='1a121667', r3='experiments/wentacc/scored20_lead_v3_hands_r1_20261104/R3_NOTES.md'),
    objects_definition='lead_sprint_v1: fb11 objects with the ep05, ep32 and ep33 mesh and track replacements',
    base_submission=dict(name='bob1',kaggle_auc=0.75166,hf='experiments/edmond/bob1_submission_20261103'),current_base=dict(name='bob1 + ep1 T2 + ep3 T5 + ep5 T3 + ep8 T4',kaggle_auc=0.76057,refs=dict(auc=57975024,sp_sr=57975349,mp_sr=57975482,rpe=57975355,mppe=57975359,cd_o=57975364),next='ep8 T6 won on rank-sum on 2026-11-04; base with T6 was being submitted when this manifest was written'),episodes=rows)
(W/'CURRENT_MANIFEST.json').write_text(json.dumps(man,indent=1,ensure_ascii=False)+'\n')
L=[]; a=L.append
a('# 版本账本（Wentao 线）\n'); a(f'更新：{man["updated_utc"]}。机器可读版：`CURRENT_MANIFEST.json`。\n')
a('**规则：不再靠前缀名判断版本，只看这份账本。每段只有一个当前 scene；有新结果就更新账本，旧前缀在 HF 上原样保留。**\n')
a('## 版本谱系\n'); a('| 代号 | HF 前缀 | 内容 | 上过 Kaggle 吗 |\n|---|---|---|---|')
for r in facts['versions']: a('| '+' | '.join(r)+' |')
a('\n**Ken 和 Haisen 是同一个人**（Haisen Kang，HF 前缀 `haisenk`）。之前的文档把两者写成两个人，是错的。\n')
a('## 勘误\n')
for x in facts['errata']: a(f'- {x}')
a('\n## 逐段\n'); a('真分差 = 把 bob1 这一段换成该 rollout 后，整体分变化 × 20，也就是这一段自身的真实 AUC 变化。\n')
a('HF 上的确切路径见 `CURRENT_MANIFEST_r3.json` 的 `scene_path`；ep28、ep33 有两个候选，看 `candidates`。\n'); a('| 段 | bob1 用谁 | bob1 own-ref | v1 物体与 bob1 这段的关系 | 当前 scene | 真分差（相对 bob1） | 状态 |\n|---|---|---:|---|---|---|---|')
for r in rows:
    pd='；'.join(f"{p['true_delta']:+.3f}（{p['label']}）" for p in r['probes']) or '—'
    a(f"| {r['episode']} | {r['bob1_source']} | {r['bob1_own_ref']:.3f} | {r['objects_vs_bob1']} | {r['current_scene']} | {pd} | {r['status']} |")
(W/'VERSION_LEDGER.md').write_text('\n'.join(L)+'\n')
print('rows',len(rows))
