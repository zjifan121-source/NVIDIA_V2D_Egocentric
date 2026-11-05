# `ep6_d2/`: episode 6 in-pot spoon de-smoothing (D2), used in FINAL_B

**Fix.** The bob1 ep6 scene smooths the spoon track while the spoon is inside the pot, which hides real fast motion. D2 removes
that over-smoothing using the raw FoundationPose residual and keeps bob1's meshes unchanged. The resulting scene is
`rf_ep6_d2x`.

**Run.** `x_ep6_d2_K_s44`, KEN/STRICT recipe. The submitted rollout is the checkpoint-sweep evaluation at step 85004288, with
own-ref 0.9446. It was not re-drawn. The sweep over 70M-100M gave:

| Step | 70M | 75M | 80M | 85M | 90M | 95M | 100M |
|---|---|---|---|---|---|---|---|
| Own-ref | 0.402 | 0.414 | 0.939 | 0.945 | 0.941 | 0.408 | 0.348 |

**Evidence.**
- D2 failed one pre-registered GT-free gate, G6(c): the attach rule re-picked the grip, a +0.6 cm shift against a 0.5 cm limit.
  The record is `FAILED.json`.
- It was trained anyway and checked on the leaderboard. Adding it to FINAL_A lowered MPPE from 3.28710 to 3.28089 (Kaggle
  57998873), and the result became FINAL_B.
- An earlier seed, s43 at 80M, was neutral on RPE (3.87791 against 3.87982) and was not used.

## Files

- `scripts/d2.py`: the de-smoothing.
- `scripts/ep6_fixpatch.py`: scene patch.
- `scripts/build_d2.sh`, `scripts/run_d2.sh`, `scripts/setup.sh`: build and install.
- `scripts/g6.py`, `scripts/run_g6*.sh`: gate G6 (attach and grip check).
- `babel_out/work/interpen.py`: interpenetration check.
- `gates.json`: the pre-registered gates.
- `notes.md`: working notes.
