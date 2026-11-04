# Read-only: full recent AUC history. No upload calls.
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
s='v2d-challenge-track3-auc'
for x in (api.competition_submissions(s,page_size=20) or []):
    print(str(getattr(x,'date',''))[5:16],getattr(x,'submitted_by_ref',None),getattr(x,'public_score',None),'|',(getattr(x,'description','') or '')[:150],flush=True)
