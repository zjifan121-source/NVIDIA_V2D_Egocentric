# Overleaf sources: the reconstruction line's part of the team technical report (2026-11-04)

The team report is written in the shared Overleaf project "V2D Technical Report - Gravity Aligned"
(`main.tex`, NVIDIA extended-abstract class, 2-page main-text limit). These files are the exact text
inserted there on 2026-11-04, second round after the lead's review (main.tex sha256 after the edit:
`07dbdcb948f7447a35e43bd85105b101ec69d1c54de20e4b5ebd14cf485d6c2e`):

- `patha_prime_section.tex` — Sec. 2.3 "Path A′: metric scene refinement (fixed-mesh tracking)" (`sec:wentao`): the teammates' technical sentences kept, the `\todo` replaced by the front-end stage sentence, the public-split 0.908 result, the scene-hygiene list (marked `\cut`), the ep28 results and the ep33 recipe control.
- `selection_two_check_paragraph.tex` — one paragraph appended to Sec. 2.7 "Ground-truth-free selection": board emulator + camera proxy, two-check rule and its validation on 14 swaps (their ρ-weighted rule untouched).
- `compliance_code_container.tex` — the Compliance paragraph (plain statement, no TODO) and the "Code and container" line (commit de66fea3…, PR #24, container after 12 Nov, Apache 2.0 on request).
- `appendix_patha_prime.tex` — Appendix "Path A′ details": stage list, defect/repair table (`tab:hygiene`), two-check decisions, artefact provenance.
- `appendix_provenance.tex` — Appendix "Provenance of the submitted file" (`app:provenance`, `tab:provenance`): final file sha, kit provenance column hashes, per-episode checkpoint and row hashes from `../table_episodes.tex`.

Other edits in teammates' text, all requested by the lead's review: author block (`Wentao Cao` third,
wentaoc@andrew.cmu.edu); abstract and conclusion score placeholders -> 0.765 (team final; 0.752 own file);
leaderboard table final row and the Sec. 3.5 placeholder -> the team final's six-board numbers; recipes table
caption + "on ep33, 0.637 (DEF) vs. 0.827 (KEN)". No scan-related wording anywhere in our text.

Numbers come from `../REPORT.tex` (team draft), `../../scene_consistency_2026-11-04/PITFALLS_AND_LESSONS.md`,
`work/reports/20261103-lead-sprint/` and `work/reports/20261030-eight-episode-comparison/`; the kit provenance
hashes were read from the submitted file's columns (`eval_script_sha256`, `eval_code_sha256`,
`packer_script_sha256`, `uploader_script_sha256`, `checkpoint_sha256`, `code_commit_url`).
