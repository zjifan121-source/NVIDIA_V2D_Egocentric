from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
c="v2d-challenge-track3-auc"; lim=api.competition_get_submission_limits(c)
print("allowed_now",getattr(lim,"num_allowed_now",None),"today",getattr(lim,"num_today",None))
for s in api.competition_submissions(c)[:5]: print(str(s.date)[5:16],getattr(s,"submitted_by_ref",None),s.public_score,(s.description or "")[:90])
