"""Phase-1 review (pathb/review.py): the submitted candidate per episode and the fix ranking."""
import json

from pathb import review


def scores(auc, mppe=5.0):
    return {"AUC": auc, "SP-SR": 0.0, "MP-SR": 0.0, "MPPE cm": mppe, "RPE cm": 0.0, "CD-O cm": 0.5}


def episode(run, n, wf, sel=None):
    root = run / f"episode_{n:06d}"
    (root / "s6_select").mkdir(parents=True)
    (root / "waterfall.json").write_text(json.dumps({"scores": wf}))
    if sel:
        (root / "s6_select/selection.json").write_text(json.dumps(sel))
    return root


def test_chosen_and_ranking(tmp_path):
    wf = {"raw": scores(0.40), "raw@scan": scores(0.45), "fused": scores(0.50), "ref_keyframe": scores(0.55),
          "ref_still": scores(0.35), "rollout_keyframe": scores(0.45), "rollout_still": scores(0.30)}
    sel = {"candidate": "keyframe", "evaluation_rule_pick": "still",
           "scores": {"keyframe": scores(0.45), "still": scores(0.30)},
           "costs": {"keyframe": {"cost_m": 0.04, "frames": 100, "termination_frames": 3}}}
    episode(tmp_path, 40, wf, sel)  # one object, S6 ran
    episode(tmp_path, 41, {"raw": scores(0.30), "fused": scores(0.35), "ref_keyframe": scores(0.38), "ref_still": scores(0.36)})
    pub = [review.public_episode(tmp_path, n, [], deep=False) for n in (40, 41)]
    c40, c41 = pub[0]["choice"], pub[1]["choice"]
    assert c40["rollout"] and c40["candidate"] == "keyframe" and c40["scores"]["AUC"] == 0.45 and c40["still"]["AUC"] == 0.30
    assert not c41["rollout"] and c41["column"] == "ref_keyframe"  # no S6: the best waterfall column
    assert pub[0]["flaw"]["stage"] == "P1c camera + S3b track" and abs(pub[0]["flaw"]["loss"] - 0.60) < 1e-9
    ranking = review.fix_ranking(pub)
    assert ranking[0]["stage"] == "P1c camera + S3b track" and abs(ranking[0]["auc_at_stake"] - (0.60 + 0.70) / 2) < 1e-9
    s6 = next(r for r in ranking if r["stage"] == "S6 rollout")  # ref 0.55 -> rollout 0.45 on ep40
    assert s6["episodes"] == [40] and abs(s6["auc_at_stake"] - 0.05) < 1e-9
    rule = next(r for r in ranking if r["stage"] == "S6 evaluation selection rule")
    assert rule["kind"].startswith("new") and abs(rule["auc_at_stake"] - 0.15 / 2) < 1e-9
