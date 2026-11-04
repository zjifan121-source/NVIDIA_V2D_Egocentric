from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
for b in ["auc","sp-sr","mp-sr","rpe","mppe","cd-o"]:
    c="v2d-challenge-track3-"+b
    r=api.competition_leaderboard_view(c)
    rows=[(i+1,getattr(x,"team_name",None),getattr(x,"score",None)) for i,x in enumerate(r)]
    print(b,[x for x in rows if x[1]=="Gravity Aligned"],"of",len(rows),"top",rows[0][2])
