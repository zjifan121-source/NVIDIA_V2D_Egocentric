# Trackability, steps 1-3: validation (2026-11-02)

**Data**
- **Pairs:** the 39 F2 pairs whose trained reference is verified (facts.md 0.1): 19 inverted, 1 keyframe and 19
  still, over 19 public episodes.
- **Tax** = the reference's playback AUC − F2's rollout AUC (official metric code).
- **Features:** `pathb/trackability.py` on F2's own S5 tasks (`v4_h1_vm/.../s4b2_trackability*/<cand>/trackability.json`),
  with moving frames and rests from the md5-matched run's S3c report. CPU only (Mac).
  - Cost: 0.5-18 s per candidate, so ≤ 35 s per episode (the bar is 2 min); RSS reuses S3a's CoACD pieces.
- **Scripts:** `scripts/step1.py`, `scripts/step2_pairs.py`, `scripts/validate.py`.

**Definitions**
- **UMS:** the share of moving frames (posed-mesh centroid > 1 cm per frame, or inside an S3c grasp or bridge) on
  which no hand has ≥ 2 fingertip sites within δ = max(1.5 cm, ½ × the mesh's smallest oriented extent).
- **RCC:** the share of moving frames with ≥ 1 / ≥ 2 filled contact slots.
- **RSS:** each object dropped for 0.5 s from frame 0 and from every static segment's first frame, in S5's scene
  (S3a's CoACD pieces, S5's support disks, ground at z = 0, no hands). Supports first, as S3d: objects whose bottom is
  lower stay fixed at the reference's pose, objects above are left out. `rss_flag` = rss0 > 1 cm or rss_max > 10°.
- **Still candidate:** UMS 0 by definition (no moving frames).

## Verdict: REJECT (3.4). Every bar fails, so step 4 is not done and nothing is pushed.

| Model (LOO by episode) | ρ_s (predicted rollout vs rollout) | LOO MAE | pick, 12 F2 episodes | pick, all 19 |
|---|---|---|---|---|
| **tax̂ = 0.097 + 0.276·ums + 0.124·rss_flag** (all-pair fit) | **+0.694** (bar ≥ 0.8) | **0.133** (bar ≤ 0.05) | **9 / 12** (bar ≥ 11) | 10 / 19 |
| ums alone | +0.692 | 0.133 | 8 / 12 | 9 / 19 |
| bridged share alone | +0.706 | 0.125 | 7 / 12 | 10 / 19 |
| constant tax (no feature) | +0.658 | 0.146 | 9 / 12 | 10 / 19 |
| playback alone (no tax) | — | 0.193 | 9 / 12 | 10 / 19 |
| GT-free observation rule (F2's own pick) | — | — | 11 / 12 | 17 / 19 |

- **Best single feature by LOO MAE:** the bridged share (0.125). It barely beats a constant tax (0.146).
- **Effect of the features:** they lift ρ_s by 0.03-0.05 over the constant tax.
- **Picks:** no feature model picks better than playback alone. F2's GT-free rule picks better still.

## Step 1: bridged share (S3c's labels only)
On each trained inverted reference (the md5-matched Mac run): the share of moving frames (centroid > 1 cm per frame,
or inside a grasp or bridge) that are bridged, failed or unstable. Compared with the inverted tax (scipy):

| Pairs | bridged ρ_s (τ) | failed | unstable | any of the three |
|---|---|---|---|---|
| all 19 inverted | +0.207 (+0.123) | +0.176 (+0.142) | +0.156 (+0.099) | +0.207 (+0.123) |
| 13 named (G-H's 11 + G3's 11, 23) | +0.214 (+0.154) | +0.013 (+0.035) | +0.154 (+0.128) | +0.214 (+0.154) |

ρ_s < 0.6, so the hypothesis does not stand on the solver's labels alone. Steps 2-3 ran anyway, as the task says.

## 3.1 Features against F2's rollouts

**All 39 pairs**

| Feature | ρ_s vs rollout AUC | τ | ρ_s vs tax | τ |
|---|---|---|---|---|
| ums | -0.080 | -0.062 | +0.518 | +0.395 |
| ums_opp | +0.063 | +0.046 | +0.487 | +0.355 |
| ums_run | -0.052 | -0.037 | +0.578 | +0.431 |
| d_hand_cm | +0.041 | +0.031 | +0.481 | +0.340 |
| rcc1 | -0.037 | -0.022 | -0.461 | -0.349 |
| rcc2 | -0.038 | -0.025 | -0.466 | -0.352 |
| rss0_cm | -0.187 | -0.131 | +0.218 | +0.151 |
| rss_max_cm | -0.143 | -0.108 | +0.329 | +0.228 |
| unsettled rests | -0.265 | -0.211 | +0.451 | +0.349 |
| rss_flag | -0.221 | -0.182 | +0.427 | +0.355 |
| support_ok | +0.442 | +0.365 | -0.303 | -0.252 |
| bridged share | +0.046 | +0.029 | +0.545 | +0.382 |

The still pairs make these ranks look better than they are. The still candidate has no moving frames (UMS, RCC and the
bridged share are 0 or undefined) and mostly a small tax.

**The 19 inverted pairs alone**
- Against the tax: ums +0.225, ums_run +0.357, rss_flag +0.310, bridged +0.207, rcc1 -0.140.
- Against the relative tax (tax / playback): ums +0.392, ums_run +0.421, rss_flag +0.434.
- On the 15 inverted pairs with playback ≥ 0.6: ums +0.520, ums_run +0.550, rss_flag +0.378.

**The task's two ranking requirements**
- **UMS ranks 42, 45, 22, 27, 2 above 40, 41, 44, 26: yes, by 0.01.**

  | | 42 | 45 | 22 | 27 | 2 | 40 | 41 | 44 | 26 |
  |---|---|---|---|---|---|---|---|---|---|
  | UMS | 0.16 | 0.03 | 0.25 | 0.31 | 0.05 | 0.02 | 0.00 | 0.00 | 0.00 |
  | RCC ≥ 1 | 0.57 | 0.97 | 0.71 | 0.63 | 0.77 | 0.97 | 0.99 | 0.92 | 0.91 |

  RCC fails: ep45 keeps 97 % contact coverage.
- **RSS flags 18 and ep26's pot, none of 40, 41, 44: yes at episode level. On ep18 it flags the wrong object.**
  - **ep18 inverted:** flagged through the dust pan (1.99 cm, 14.3° at frame 0). The sweeper, which F2's rollouts tip
    (BUILD_LOG 2026-10-29: 33° by step 7; here 93° inverted / 42° still within 2 s), stands in RSS (0.1 cm, 1.2°).
    The still scene is not flagged at all.
  - **ep26:** the pot leaves its rack (66 cm).
  - **Not flagged, as required:** ep40 (0.0 cm), ep41 (0.25 cm, worst rest 0.86 cm), ep44 (0.80 cm).

**What UMS cannot see.** Four large taxes have the hand on the object nearly the whole time:
- ep11: tax 0.625, UMS 0.02
- ep45: tax 0.372, UMS 0.03
- ep2: tax 0.385, UMS 0.05
- ep23: tax 0.484, UMS 0.08

In the G-H row (BUILD_LOG 2026-10-31), ep45's loss is the carrier wrist's reference speed (p95 1.93 m/s, max 5.8):
a hand on the object, but one the policy cannot follow.

## 3.1b RSS against the rollout's own first 2 s
Each object's maximum error against its reference over frames 0-40 of F2's rollout, worst object per pair.
- ρ_s(rss0 cm, drift cm) = +0.684; ρ_s(rss0 deg, drift deg) = +0.386.
- The frame-0 flag (rss0 > 1 cm or > 10°) against a drift > 1 cm or > 10°: agree on 30 / 39.
  - Never a flag without drift.
  - 9 drifts without a flag: the policy moves the object early (ep11 inverted 21 cm, ep40 inverted 4.9 cm / 12°,
    ep27 inverted 6.4 cm) or tips a sweeper RSS keeps standing (ep18 still).
- **Magnitudes are overstated on stacked rests:**
  - ep26's pot in its rack: 66 cm vs 2.5 cm.
  - ep0 / ep2 lids on their pots: 13-32 cm vs 4.3-10.6 cm.
  - The reason: these rests start interpenetrating (ep0's lid 0.8 cm inside the pot), and S3d's settle, with
    S3d's stiff contacts, reports the same unsettled lid (13.3 cm, 34°).
- **Missing supports are exact:** ep12 / ep13 still at 102 / 99 cm against 105 / 102 cm in the rollout.

## 3.2 Fits (least squares, leave-one-episode-out)

| Fit | Coefficients (all pairs) | LOO MAE | ρ_s(pred, rollout) | pick 12 | pick all |
|---|---|---|---|---|---|
| ums + rss_flag | 0.097, 0.276, 0.124 | 0.133 | +0.694 | 9 / 12 | 10 / 19 |
| ums alone | 0.155, 0.405 | 0.133 | +0.692 | 8 / 12 | 9 / 19 |
| bridged share alone | 0.117, 0.285 | 0.125 | +0.706 | 7 / 12 | 10 / 19 |
| rss_flag alone | 0.109, 0.147 | 0.133 | +0.693 | 9 / 12 | 10 / 19 |
| ums_opp + rss_flag | 0.077, 0.227, 0.114 | 0.129 | +0.707 | 8 / 12 | 9 / 19 |
| rcc1 + rss_flag | 0.342, -0.253, 0.128 | 0.135 | +0.690 | 8 / 12 | 9 / 19 |
| d_hand + rss_flag | 0.093, 0.024, 0.126 | 0.135 | +0.695 | 9 / 12 | 10 / 19 |
| constant | 0.192 | 0.146 | +0.658 | 9 / 12 | 10 / 19 |

- **Pairs:** 39 pairs in 19 episodes. "pick 12" means F2's rollouts 1-12: 40 26 18 41 39 44 27 22 13 2 42 45, where
  the current rule is 11 of 12.
- **Pick rule:** argmax over an episode's candidates of the predicted rollout AUC, against the GT-best rollout.
- **Single features** (LOO MAE, ρ_s): ums_opp 0.129 / +0.690; rcc2 0.131 / +0.688; unsettled 0.132 / +0.683;
  support_ok 0.134 / +0.709; ums_run 0.136 / +0.712.

## 3.3 Predicted six metrics (main fit, LOO)

| Metric | Prediction | Error | Playback as the prediction |
|---|---|---|---|
| AUC | playback − tax̂ | MAE 0.133, ρ_s +0.694 | MAE 0.193 |
| SP-SR | 0 when ums_run > 60 frames | agrees 27 / 39 | 24 / 39 |
| MP-SR | 0 when rss_flag or ums > c_ums | agrees 39 / 39 | 32 / 39 |
| MPPE / RPE | playback / (1 − tax̂ / playback) | MAE 59.6 / 60.3 cm | 12.2 / 13.3 cm |
| CD-O | unchanged | MAE 0.005 cm | 0.005 cm |

- **MP-SR's 39 / 39 is trivial.** c_ums = 0.00 was chosen in-sample, and every F2 rollout but ep44's keeps MP-SR 0.
- **MPPE / RPE.**
  - The task's "scaled by (1 − tax̂ / playback)" is read here as division, so the error grows when the AUC falls.
  - Multiplying would shrink it, the wrong direction.
  - Division blows up where tax̂ approaches the playback (the ratio is clipped at 0.05), so these metrics are not
    usable as built.

## What this does not change
- **Selection:** S6 still selects by the observation cost (F2: right on 11 of 12, 17 of 19).
- **The compare tool:** it keeps the per-type submitted estimate (0.22 single-object, 0.29 two-object).
- **The stage:** `s4b2_trackability` stays in the chain as a health check: frame-0 support, RSS and UMS per S5
  candidate, ≤ 35 s per episode.
- **What it catches before training:** a missing support (ep12 / ep13 still, ep21 / ep23 / ep24 inverted).

## Wrist-speed term (James, 2026-11-02: "Test wrist speed"): still REJECT

**What was added**
- **Features** (`pathb/trackability.py`), over the moving frames, from S5's `robot_{side}_wrist_*` at the task's fps:
  the carrier wrist's (the hand with the nearer fingertip) reference speed p95 and max, the share faster than
  1.5 m/s (the G-H row's bar), acceleration p95 and angular speed p95. The still candidate's hands stand still (all 0).
- **They reproduce the G-H row** (BUILD_LOG 2026-10-31):
  - ep45: acceleration p95 104 m/s² (G-H: 86), max 5.9 m/s (5.8), angular p95 695°/s (777)
  - ep39: max 10.4 m/s (10.4)
  - ep42: acceleration p95 43 m/s² (32)
- **Scripts:** `scripts/step2_wrist.py` recomputes every feature but RSS with the current module and keeps step 2's RSS
  (its code did not change); `scripts/validate_wrist.py`.

**Ranks against the tax (ρ_s)**

| Feature | all 39 | inverted 19 | inverted, playback ≥ 0.6 (15) |
|---|---|---|---|
| wrist p95 | +0.496 | +0.140 | +0.286 |
| wrist max | +0.493 | +0.049 | +0.271 |
| share > 1.5 m/s | +0.441 | +0.327 | +0.428 |
| acceleration p95 | +0.529 | +0.270 | +0.396 |
| angular p95 | +0.472 | -0.084 | -0.011 |
| (UMS) | +0.518 | +0.225 | +0.520 |
| (RSS flag) | +0.427 | +0.310 | +0.378 |

**Fits:** 39 models of at most two features (each wrist term alone, with UMS, with the RSS flag, with `ums_run`, and
pairs of wrist terms), LOO by episode. The best 5 by MAE:

| Model | Coefficients (all pairs) | LOO MAE | ρ_s | pick 12 | pick all |
|---|---|---|---|---|---|
| RSS flag + share > 1.5 m/s | 0.049, 0.163, 9.18 | 0.111 | +0.794 | 8 / 12 | 11 / 19 |
| RSS flag + acceleration p95 | 0.060, 0.168, 0.0043 /(m/s²) | 0.120 | +0.752 | 8 / 12 | 9 / 19 |
| `ums_run` + share > 1.5 m/s | 0.110, 0.0029 /frame, 7.66 | 0.120 | +0.738 | 8 / 12 | 10 / 19 |
| RSS flag + wrist p95 | 0.046, 0.119, 0.30 /(m/s) | 0.121 | +0.735 | 7 / 12 | 10 / 19 |
| UMS + share > 1.5 m/s | 0.119, 0.307, 7.97 | 0.123 | +0.740 | 8 / 12 | 9 / 19 |

**Verdict:** no model passes 3.4.
- **The best** (RSS flag + the share of moving frames with the carrier wrist faster than 1.5 m/s) cuts LOO MAE from
  0.133 to 0.111 (bar 0.05) and lifts ρ_s from +0.694 to +0.794 (bar 0.8). It is the best of 39 models on the same 39
  pairs, so it is optimistic.
- **Picks get worse:** 8 of 12 (bar 11); no model passes 9 of 12.
- **Still unexplained:** ep2 (tax 0.385: UMS 0.05, wrist max 1.39 m/s, no fast frame) and ep23 (0.484: UMS 0.08, max 1.12 m/s).
  Neither where the reference hand is nor how fast it moves explains these.
