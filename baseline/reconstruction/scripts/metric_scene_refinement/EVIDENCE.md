# Measured scope and extraction provenance

This release is an experimental, opt-in implementation. It has not been run across
Scored20, does not estimate Scored AUC, and does not publish bundles. Runtime paths,
weights, observed images, scans and experiment caches are not part of this commit.

## Local Public results already measured before this release

| Case / comparison | Scene AUC | Inter-object RPE (cm) |
| --- | ---: | ---: |
| ep27, 246 frames, same-front-end old asset A | 0.266514 | 41.408610 |
| ep27, input-selected R7/R8 C1 | 0.912729 | 1.599919 |
| ep21, 186 frames, retained support-FP baseline | 0.733535 | 3.763363 |
| ep21, contact-shape FP before temporal correction | 0.752184 | 2.069690 |
| ep21, same mesh + full temporal pose correction | 0.840054 | 1.879144 |
| ep22, 189 frames, original bundle | 0.624835 | 2.749806 |
| ep22, shared part/body shape + fresh FP | 0.651290 | 2.726690 |

These are Public **local reconstruction** results. The local metric implementation
uses the existing approximate scan-centroid/body-registration/symmetry adapter;
these are not organizer-confirmed exact hidden scores or policy scores. Public GT
has been exposed during development. Candidate predictions were frozen before the
corresponding new score, but research direction and retained combinations were
GT-informed: this is not a fresh blind benchmark.

Ep21 pitcher position improved from 4.184387 to 3.146753 cm against its retained
baseline; orientation changed from 7.246726 to 7.831908 degrees (worse). Ep22 pitcher
position improved from 5.939554 to 5.575297 cm, orientation from 6.231947 to 6.302870
degrees (worse). Report these tradeoffs, not just the scene AUC. A partner's score
can change through common-scene registration even when its raw prediction is fixed.

The two extracted shape fits did not pass their original input-only gates. Ep21's
source shape p90 warning remains; ep22's bounded previous endpoint was investigated
as an exploratory branch after a shape gate failure. Their downstream gains do not
retroactively make those gates pass. The CLI now explicitly returns failure for
these artifacts and never automatically continues to tracking. Temporal correction
requires a separately supplied, reviewed full bundle and calibrated measurements.

The current development combination across 20 Public episodes has arithmetic mean
AUC 0.907555 and mean RPE 1.798315 cm across the 13 paired episodes. It combines
historical methods/candidates and GT-informed retention; it is **not** one fixed
new pipeline run on all 20 episodes. Input proxies have also failed to select better
candidates reliably. No public-to-scored score conversion is justified.

## CPU extraction checks

- R7/R8 is synchronized from the existing `main_buffer` implementation, rather than
  rewritten from operations scripts. Its tests exercise matching mesh/pose ownership,
  independent support SE(3), one shared scale, heldout checks and explicit failures.
- On saved ep22 input, the extracted geodesic basis, shape coefficients, all support
  poses, faces and vertices exactly match the historical fit (max array difference 0).
- On saved ep21 contact-shape input, graph controls, basis, coefficients, contact
  target, auxiliary poses, faces and vertices exactly match (max difference 0).
- On saved ep21 temporal input, all 186 camera/world transforms, local landmarks,
  correction vectors and **every** native bundle field exactly match (max difference 0).
- The contact-shape and part-shape gates still fail, as originally recorded. Their
  results and failure receipts are preserved; this is expected parity, not a new gain.
- Refactored FP contact propagation is CPU contract-tested; no new inference,
  reconstruction, GPU tracking, training, official scoring or HF write occurred.

CPU replay used Python 3.11.15, NumPy 2.3.5, SciPy 1.17.1, OpenCV 4.10.0,
trimesh 5.1.1 and numba 0.68.1. Other platforms may have numerical differences.
The tests do not promise kernel-level GPU parity.

## Still outside this implementation

Input-only optimal-view selection is reviewed, not automated. Observed landmark
identity/triangulation, plane fitting, contact-phase annotation, and full-scene final
acceptance remain upstream/downstream interfaces. The release does not fix hands,
validate policy contact, solve all shape/pose ambiguity, or select Scored candidates
from GT. Failed recent cup/lid experiments are not included as accepted methods.
