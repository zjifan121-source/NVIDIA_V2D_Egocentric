import json,subprocess,sys,time
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
f,msg,rc=sys.argv[1:4]
p=Path(rc)
if not p.exists(): print("NO_RECEIPT",rc); raise SystemExit(1)
r=json.loads(p.read_text())["submissions"]
a=KaggleApi(); a.authenticate()
for b,s in r.items():
    if s.get("state")=="COMPLETE": print(b,s.get("ref"),s.get("public_score")); continue
    print(b,"NOT_COMPLETE",s.get("state"),str(s.get("error"))[:120])
    for t in range(1,6):
        try:
            on=[x for x in a.competition_submissions(s["slug"])[:6] if getattr(x,"description","")==msg]
        except Exception as e:
            print("board read failed",str(e)[:100]); time.sleep(25); continue
        if on:
            x=on[0]; print(b,"ALREADY_ON_BOARD",getattr(x,"ref",None),getattr(x,"public_score",None)); break
        q=subprocess.run(["/root/venv-kit/bin/python","submit/v2d_submit.py",f,"--track","track_3","--only",b,"--wait","-m",msg,"--receipt",rc.replace(".receipt.json",f".{b}.retry{t}.receipt.json")],cwd="/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit",capture_output=True,text=True)
        print("\n".join(q.stdout.splitlines()[-3:])[:400])
        if " COMPLETE " in q.stdout: break
        time.sleep(25)
