import sys,pyarrow.parquet as pq
t=pq.read_table(sys.argv[1]); print(t.num_rows,t.num_columns,t.column_names[:8])
df=t.select(['eval_script_sha256','eval_code_sha256','packer_script_sha256','uploader_script_sha256','code_commit_url']).to_pandas()
print({c:[str(x)[:16] if c!='code_commit_url' else x for x in df[c].unique()][:2] for c in df.columns})
