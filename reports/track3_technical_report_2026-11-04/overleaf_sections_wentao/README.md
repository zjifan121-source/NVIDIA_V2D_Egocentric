# Overleaf sources: the reconstruction line's part of the team technical report (2026-11-04)

The team report is written in the shared Overleaf project "V2D Technical Report - Gravity Aligned"
(`main.tex`, NVIDIA extended-abstract class, 2-page main-text limit). These files are the exact text
inserted there on 2026-11-04, second round after the lead's review (main.tex sha256 after the edit:
`e549b11b43a7ed3872d5385b23a99d3ed3b5ef685e6cd1a78a0b451e4eea2986`):

- `patha_prime_section.tex` — Sec. 2.3 "Path A′: metric scene refinement (fixed-mesh tracking)" (`sec:wentao`): the teammates' technical sentences kept, the `\todo` replaced by the front-end stage sentence, the public-split 0.908 result, the scene-hygiene list (marked `\cut`), the ep28 results and the ep33 recipe control.
- `selection_two_check_paragraph.tex` — one paragraph appended to Sec. 2.7 "Ground-truth-free selection": board emulator + camera proxy, two-check rule and its validation on 14 swaps (their ρ-weighted rule untouched).
- `compliance_code_container.tex` — the Compliance paragraph (plain statement, no TODO) and the "Code and container" line (commit de66fea3…, PR #24, container after 12 Nov, Apache 2.0 on request).
- `appendix_patha_prime.tex` — Appendix "Path A′ details": stage list, defect/repair table (`tab:hygiene`), two-check decisions, artefact provenance.
- `appendix_provenance.tex` — Appendix "Provenance of the submitted file" (`app:provenance`, `tab:provenance`): final file sha, kit provenance column hashes, per-episode checkpoint and row hashes from `../table_episodes.tex`.

Other edits in teammates' text, all requested by the lead's review: author block (`Wentao Cao` third,
wentaoc@andrew.cmu.edu); abstract and conclusion score placeholders -> 0.765 (team final; 0.752 before the four single-episode swaps);
leaderboard table final row ("Team final (bob1 + 4 swaps)") and the Sec. 3.5 placeholder -> the team final's six-board numbers; recipes table
caption + "on ep33, 0.637 (DEF) vs. 0.827 (KEN)". No scan-related wording anywhere in our text.

Numbers come from `../REPORT.tex` (team draft), `../../scene_consistency_2026-11-04/PITFALLS_AND_LESSONS.md`,
`work/reports/20261103-lead-sprint/` and `work/reports/20261030-eight-episode-comparison/`; the kit provenance
hashes were read from the submitted file's columns (`eval_script_sha256`, `eval_code_sha256`,
`packer_script_sha256`, `uploader_script_sha256`, `checkpoint_sha256`, `code_commit_url`).

## State at the close (2026-11-04, about 23:00 UTC)

The teammates replaced `main.tex` with a 2-page rewrite (921 to 381 lines) between 21:45 and 22:45 UTC, so most blocks above no longer appear verbatim in Overleaf; the files here are the record of what this line wrote. The sha256 `e549b11b...` above is the last version with these blocks intact. `appendix_provenance.tex` was then updated at the lead's request to separate the counted scores (best public submission per board at the close: AUC 0.77970, SP-SR 0.95, MP-SR 0.65, RPE 3.86589, MPPE 3.28089, CD-O 0.32243, rank sum 81) from the common base file 57978563 that the table describes; in Overleaf that table also carries two columns the teammates added. Overleaf `main.tex` after that edit: `a704243d4380909cf1aba395f0bee28ec0392215afbbbdcf842b67858210df67`. Five corrections requested by the lead then went into the rewritten text: the two hygiene-table cells whose claims were never measured (joint limits, impulse ejection), the ep27 sentence, and the stream labels of ep1 (T2) and ep5 (T8) in the provenance table, whose objects are ours. `main.tex` after those: `d3b104e2fb20cd4fa1c22503b0cffcaa8dbb29dd11cdd722b122305f5ad49a60`. The provenance paragraph was then narrowed (the rows of base file 57978563 are shared only for episodes the final uploads did not replace) and the ep5 row's recipe corrected to "Re-level + STRICT" (run `r_ep5_rl_K_s43`): `264f6fa2b39f4cf39d0dd48e8186ab58fc2657636b4c71f276b007658ccf1ab4`.
