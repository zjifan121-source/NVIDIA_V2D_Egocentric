from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
for s in api.competition_submissions("v2d-challenge-track3-auc")[:2]: print(s.ref if hasattr(s,"ref") else "",str(s.date)[5:16],s.public_score,s.description)
r=api.competition_leaderboard_view("v2d-challenge-track3-auc") if hasattr(api,"competition_leaderboard_view") else None
if r:
    rows=[(i+1,getattr(x,"team_name",None),getattr(x,"score",None)) for i,x in enumerate(r)]
    print([x for x in rows if x[1]=="Gravity Aligned"], rows[:3])
