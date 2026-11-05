# `ep3_f3w/`: episode 3 spoon-track fix (F3w), used in FINAL_B

**Fix.** In the stirring windows ("F3"), the spoon track uses the Path A' (Wentao) centroid and long axis. Its twist about the
long axis comes from our own raw FoundationPose track, which is steadier relative to the wrist. Everything else is unchanged
from the adopted T5 scene (`sc_ep3`: table support at the depth-measured height, pot centre of mass at its shell centroid).
The resulting scene is `rf_ep3_f3w`.

**Run.** `rf_ep3_f3w_K_s42` at step 100003840, KEN/STRICT recipe. The submitted rollout is the best of 48 re-evaluations
(draw a7, own-ref 0.9547; see `../redraw/`).

**Evidence.**
- The pre-registered GT-free gates G1-G8 passed. They are in `gates.json` and `gate_results.json`.
- A later local judge rejected the fix (`FAILED.json`), so it was first left out of the local-best file.
- On the leaderboard it was positive. Adding it to an otherwise identical file moved AUC from 0.77661 to 0.77726 and MPPE from
  3.32849 to 3.32007 (Kaggle 57997940 and 57997944). It was then kept in FINAL_B.

## Files

- `build_f3w.py`: builds the scene from `sc_ep3` and the two tracks.
- `babel/d3.py`, `babel/d3_run.sh`: the depth and track diagnostics run on Babel; their result is `fromBabel/d3.json`.
- `build/rf_ep3_f3w/*.json`: build reports.
- `notes.md`: working notes.
