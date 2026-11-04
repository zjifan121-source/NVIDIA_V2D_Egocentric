import json,re,subprocess,sys,time
from pathlib import Path
F=Path("/root/probes/five_board"); n=sys.argv[1]
src=(F/"submit_list.sh").read_text()
m=re.search(rf"^\s*{n}\)\s+f=(\S+?);.*?m=\"([^\"]+)\"",src,re.M); f,msg=m.group(1),m.group(2)
p=F/f"{f}.receipt.json"
if not p.exists(): print("NO_RECEIPT",n); raise SystemExit
r=json.loads(p.read_text())["submissions"]
from kaggle.api.kaggle_api_extended import KaggleApi
a=KaggleApi(); a.authenticate()
for b,s in r.items():
    if s.get("state")=="COMPLETE": print(n,b,s.get("ref"),s.get("public_score")); continue
    print(n,b,"NOT_COMPLETE",s.get("state"),str(s.get("error"))[:120])
    for t in range(1,5):
        try:
            on=[x for x in a.competition_submissions(s["slug"])[:6] if getattr(x,"description","")==msg]
        except Exception as e:
            print("board read failed",str(e)[:100]); time.sleep(20); continue
        if on:
            x=on[0]; print(n,b,"ALREADY_ON_BOARD",getattr(x,"ref",None),getattr(x,"public_score",None)); break
        q=subprocess.run(["/root/venv-kit/bin/python","submit/v2d_submit.py",str(F/f"{f}.parquet"),"--track","track_3","--only",b,"--wait","-m",msg,"--receipt",str(F/f"{f}.{b}.retry{t}.receipt.json")],cwd="/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit",capture_output=True,text=True)
        print("\n".join(q.stdout.splitlines()[-3:])[:400])
        if " COMPLETE " in q.stdout: break
        time.sleep(25)
