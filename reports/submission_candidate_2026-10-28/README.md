# Moving-reference tracking candidate — current status

Latest verified Kaggle AUC: **0.65379 (65.379%)**, submission57729024, immutable `packed/candidate_0807696.parquet`. Prior submission57726860 scored0.64008.

Current validated own-reference tracking: **0.811621 across20 scored episodes**. This is tracking against reconstructed references, not the hidden score. `plan.json` and `tracking_inventory.json` contain current selections; the uploaded candidate is an earlier immutable selection.

Episode25 selects `ep25_assembly2_OT1M_s42/early_90005504`: **0.827072**, per board0.859199/0.794944. Its actual mesh-bottom lifts are **12.42cm /10.82cm**, versus reference14.49cm/11.56cm. Both boards genuinely lift; selection now requires each actual board mesh bottom to rise at least5cm, rejecting hold-still and table-pivot failures. Assembly is still imperfect: board2 final bottom is5.76cm above its reference. See `ep25_assembly2_90m_lift_audit.json`. The uploaded75M checkpoint also genuinely lifts both boards14.60cm/11.53cm.

`ACTIVE_WORK.md` records experiment history. Refresh with `tools/.venv/bin/python reports/submission_candidate_2026-10-28/refresh_tracking.py`. Goal remains active until full deterministic unassisted tracking reaches90%+ across20 episodes with successful episode25 manipulation. Historical updates below retain their original numbers and may describe superseded trials.

## Confirmed submission score,02:18 EDT

AUC-only Kaggle submission **57726860 COMPLETE0.64008**. This meets the original60–70% submission target. Previous v2 was0.42455. Receipt packed/candidate_0792945.receipt.json. Latest selection now0.798492 after ep3_grip_s43/early90M0.716354 and ep25_grip2_s42final0.655460 (body1.81952,body2.49140). Current plan is newer than immutable submitted79.2945% candidate. Goal remains active for90% tracking; do not mark complete merely from the64% leaderboard score.

Board2 bottom-lift audit confirms old s44 policy only0.10cm maximum clearance although origin rises6.46cm; it rotates ontable. Actual reference bottom lifts11.32cm. New robot-grasp R005 counterpart job10964392 queued; weights-unchanged reference-transfer diagnostic10964404 queued (script transfer_pinch_checkpoint.py rewrites only config/provenance header, asserts binary tensor payload identical).

Exact board collision LP diagnostic ep25_exact_convex_intersections.json finds positive2.65mm Chebyshev-radius overlap from~210 onward (226LPs afterAABBculling). Prior hull-vertex sampled test was inconclusive! Investigating whether actual slot too narrow or CoACD fills it. Original OBJ texture seams need merging before boolean intersection; raw processFalse.is_watertight isFalse, not automatically broken solid. Do not modify slots before verifying source mesh and decomposition.

## Latest verified result,03:27 EDT

Tracking **80.7696%**; episode25 **79.6963%** fromep25_assembly2_OT1M_s42/early75005952, bothboardsabout80%. Actualbottomlifts14.60/11.53cm, confirmingbothpickups. NewAUCsubmission57729024accepted,pending;previous57726860scored64.008%. Immutablecandidate0807696filesandreceiptareunderpacked/. Latecheckpointmonitor10964996running;newwider-explorationandearly-firstframeresettrialsareactive. SeeACTIVE_WORK.mdlatestsectionforauthoritativetrials/evidence.
