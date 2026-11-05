# download the 14 teammate episodes' bob1 policy + scene from Edmond's HF package (read-only)
import time
from huggingface_hub import snapshot_download
base = "experiments/edmond/bob1_submission_20261103"
eps = {14: 75005952, 15: 250003456, 16: 130007040, 19: 95006720, 20: 100003840, 25: 140005376, 28: 75005952,
       30: 95006720, 32: 100003840, 33: 135008256, 35: 100003840, 36: 250003456, 37: 100003840, 38: 100003840}
pats = []
for ep, st in eps.items():
    e = f"{base}/episode_{ep:06d}"
    pats += [f"{e}/scene/*", f"{e}/policy/*.json", f"{e}/policy/*.yaml", f"{e}/policy/*.txt", f"{e}/policy/*.md",
             f"{e}/policy/results/*", f"{e}/policy/policy_{st}.safetensors", f"{e}/policy/train/policy_{st}.safetensors",
             f"{e}/policy/train/*.yaml", f"{e}/policy/train/*.json", f"{e}/policy/train/.hydra/*", f"{e}/policy/*/results/*"]
t = time.time()
p = snapshot_download("rose-tea-cafe/v2d-t3-recon", repo_type="dataset", allow_patterns=pats,
                      local_dir="/data/user_data/haisenk/v2d/push1104/taskA/hf", max_workers=16)
print("DL_DONE", p, round(time.time() - t), "s", flush=True)
