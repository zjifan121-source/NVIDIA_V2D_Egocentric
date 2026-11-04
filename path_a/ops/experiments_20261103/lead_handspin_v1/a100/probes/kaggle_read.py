# Read-only: lists recent submissions and quota on each Track 3 board. No upload calls.
import sys, json
sys.path.insert(0,'/root/v2d/NVIDIA_V2D_Egocentric/assets/data/v2d_submission_kit')
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
slugs=['v2d-challenge-track3-'+m for m in ['auc','sp-sr','mp-sr','rpe','mppe','cd-o']]
for s in slugs:
    try:
        l=api.competition_get_submission_limits(s); lim=(getattr(l,'num_allowed_now',None),getattr(l,'num_today',None))
    except Exception as e: lim=('ERR',str(e)[:80])
    print(s.replace('v2d-challenge-track3-',''),'allowed_now,used_today=',lim,flush=True)
    try:
        subs=api.competition_submissions(s,page_size=(6 if s.endswith('auc') else 2)) or []
        for x in subs:
            print('   ',str(getattr(x,'date',''))[:19],getattr(x,'submitted_by_ref',None),getattr(x,'team_name',None),'pub=',getattr(x,'public_score',None),'status=',str(getattr(x,'status',''))[-12:],'|',(getattr(x,'description','') or '')[:70],flush=True)
    except Exception as e: print('    ERR',type(e).__name__,str(e)[:120],flush=True)
