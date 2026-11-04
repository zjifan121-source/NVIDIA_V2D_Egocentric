from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
for b in ["sp-sr","mp-sr","rpe","mppe","cd-o"]:
    c="v2d-challenge-track3-"+b; l=api.competition_get_submission_limits(c); s=api.competition_submissions(c)[:1]
    print(b,"today",getattr(l,"num_today",None),[(str(x.date)[5:16],getattr(x,"submitted_by_ref",None),x.public_score,(x.description or "")[:48]) for x in s])
