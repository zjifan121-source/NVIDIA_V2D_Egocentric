"""Splice one episode's packed rows (from pack_one.sh) into a full Track 3 submission (e.g. bob1), keeping every other
row byte-identical. The five uniform provenance columns must match the base exactly (Kaggle rejects mixed values);
--adopt-commit-url copies the base's code_commit_url onto the new rows when only that field differs.
usage: python swap_episode.py BASE.parquet ONE_EPISODE.parquet EPISODE OUT.parquet [--adopt-commit-url]"""
import sys, pandas as pd
base_p, one_p, ep, out_p = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
adopt = "--adopt-commit-url" in sys.argv
base, one = pd.read_parquet(base_p), pd.read_parquet(one_p)
pre = f"track_3/e{ep:06d}/"
assert list(base.columns) == list(one.columns), f"column mismatch: {set(base.columns) ^ set(one.columns)}"
assert one.row_id.str.startswith(pre).all(), "ONE_EPISODE file holds rows of another episode"
old = base[base.row_id.str.startswith(pre)]
assert len(old) == len(one) and set(old.row_id) == set(one.row_id), f"row grid differs: base {len(old)} vs new {len(one)}"
uniform = ["eval_script_sha256", "eval_code_sha256", "packer_script_sha256", "uploader_script_sha256", "code_commit_url"]
diff = {c: (base[c].astype(str).unique().tolist(), one[c].astype(str).unique().tolist()) for c in uniform
        if set(base[c].astype(str).unique()) != set(one[c].astype(str).unique())}
if "code_commit_url" in diff and adopt:
    one = one.copy(); one["code_commit_url"] = base["code_commit_url"].iloc[0]; diff.pop("code_commit_url")
if diff:
    sys.exit(f"REFUSED: uniform provenance differs from the base: {diff}")
order = {r: i for i, r in enumerate(base.row_id)}
new = pd.concat([base[~base.row_id.str.startswith(pre)], one]).sort_values("row_id", key=lambda s: s.map(order)).reset_index(drop=True)
assert len(new) == len(base) and (new.row_id.values == base.row_id.values).all()
keep = new[~new.row_id.str.startswith(pre)]; ref = base[~base.row_id.str.startswith(pre)]
assert keep.reset_index(drop=True).equals(ref.reset_index(drop=True)), "other episodes changed"
new.to_parquet(out_p, index=False)
print(f"wrote {out_p}: {len(new)} rows; episode {ep}: {len(one)} rows replaced; other {len(keep)} rows identical to base")
print("checkpoint_sha256 old -> new:", old.checkpoint_sha256.iloc[0][:12], "->", one.checkpoint_sha256.iloc[0][:12])
