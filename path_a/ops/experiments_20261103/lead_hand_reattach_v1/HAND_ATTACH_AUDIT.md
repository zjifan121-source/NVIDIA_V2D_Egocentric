# Exclusive attach audit (2026-11-04)

The contact metric in HAND_LEDGER.md and HAND_REATTACH_COMPARISON.md is measured after attach, and attach moves the hand, so it is circular for any hand that attach moved. The rule "ship rawexcl only if every contact hand >= control - 0.05" inherited that flaw.

This audit compares the wrist in each rawexcl bundle with the same wrist in its rawclean bundle, on the A100 (no GT). Scripts: /root/probes/hold_audit.py, /root/probes/hold_audit2.py. Outputs: /root/probes/hold_audit.json, /root/probes/hold_audit2.json. A dragged frame is one where attach moved the wrist more than 5 cm.

| Episode | Hand | Dragged frames / total | Median drag cm | p95 drag cm | Median nearest-object distance before attach cm |
|---|---|---|---|---|---|
| 15 | right | 112 / 166 | 51.6 | 55.1 | 19.0 |
| 28 | left | 166 / 220 | 50.2 | 61.6 | 18.8 |
| 16 | left | 127 / 173 | 42.8 | 49.9 | 14.8 |
| 14 | left | 69 / 158 | 23.7 | 47.8 | 11.4 |
| 15 | left | 100 / 166 | 22.5 | 42.3 | 16.6 |
| 19 | left, needs review | 116 / 235 | 11.1 | 37.0 | 13.5 |

Episodes 1, 3, 5, 6, 8, 10, 20, 25, 30, 32, 33: median drag at most 15.1 cm with the hand already near the object. Episodes 35-38 have no raw hands here and were not audited; another session's test flags ep36 left.

Cause: attach_exclusive.py holds a hand for a whole lifted segment if it comes within 12 cm once. Fix: per-frame hold, /root/v2d/lead_v3/handspin/attach_hold.py (Edmond's de66fea no-mask rule). Scenes: old pot /root/v2d/lead_v3/handspin/scenes/ep28_hold, new pot /root/v2d/lead_v3/handspin28/scenes/ep28_hold (object tracks verified identical to the v8 pot).

Affected HF data, which stays as is because HF is append-only: scored20_lead_v3_ep28pot_20261104 (ep28 left hand) and the rawexcl scenes for ep14, 15, 16 and 28 in the v2 addenda and lead_v3 wherever those episodes shipped rawexcl.
