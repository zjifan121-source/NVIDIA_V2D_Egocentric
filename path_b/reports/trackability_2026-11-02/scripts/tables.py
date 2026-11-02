"""Markdown tables for facts.md from facts02.json (pairs) and contacts_split.json (0.4).

    python tables.py facts02.json contacts_split.json
"""
import json
import sys

M = ("AUC", "SP-SR", "MP-SR", "MPPE cm", "RPE cm", "CD-O cm")


def six(s):
    return " / ".join(f"{s[k]:.3f}" if k in ("AUC",) else f"{s[k]:.2f}" for k in M)


rows = [r for r in json.load(open(sys.argv[1])) if "error" not in r]
split = json.load(open(sys.argv[2]))
print("### 0.1 Pairs\n")
print("Six metrics in the order AUC / SP-SR / MP-SR / MPPE cm / RPE cm / CD-O cm. `trained ref` = the Mac run whose "
      "`s4b_reference/<object>/<candidate>/result.npz` has the md5 of the VM's (S5's input); `k7` = that md5 is "
      "`v4_k7`'s.\n")
print("| ep | candidate | trained ref (md5 per object) | k7 | playback | rollout | tax | GT pick | GT-free pick |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    ep = f"{r['episode']}" + (" (`_pre_k7`)" if r["folder"].endswith("_pre_k7") else "")
    md5 = ", ".join(f"{o} `{h}`" for o, h in r["trained_md5"].items())
    runs = " = ".join(f"`{m}`" for m in r["md5_runs"] if m in ("v4_k7", "v4_j10")) or f"`{r['playback_run']}`"
    print(f"| {ep} | {r['candidate']} | {runs}: {md5} | {'yes' if r['is_k7'] else '**no**'} | "
          f"{six(r['playback'])} | {six(r['rollout'])} | {r['tax']:+.3f} | {r['pick']} | {r['gt_free_pick']} |")
print("\n### 0.3 Evaluator record per rollout\n")
print("| ep | candidate | failure_cause_fraction object / wrist | non_finite_worlds | steps | first frame > 10 cm: vs reference | vs GT |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    f = r["failure_cause_fraction"]
    ep = f"{r['episode']}" + (" (`_pre_k7`)" if r["folder"].endswith("_pre_k7") else "")
    print(f"| {ep} | {r['candidate']} | {f.get('object'):.0f} / {f.get('wrist'):.0f} | {len(r['non_finite_worlds'])} | {r['steps']} | "
          f"{r['first_ref_err10_frame'] if r['first_ref_err10_frame'] is not None else '—'} | "
          f"{r['first_gt_err10_frame'] if r['first_gt_err10_frame'] is not None else '—'} |")
print("\n### 0.4 Contact slots per S3c kind (inverted candidate; share of frames with >= 1 / >= 2 filled slots)\n")
print("| ep | static: frames, >= 1, >= 2 | grasp | bridge | other |")
print("|---|---|---|---|---|")
for ep, d in split.items():
    cells = []
    for k in ("static", "grasp", "bridge", "other"):
        v = d[k]
        cells.append(f"{v['frames']}, {v['ge1']:.2f}, {v['ge2']:.2f}" if v["frames"] else "0")
    print(f"| {ep} | " + " | ".join(cells) + " |")
