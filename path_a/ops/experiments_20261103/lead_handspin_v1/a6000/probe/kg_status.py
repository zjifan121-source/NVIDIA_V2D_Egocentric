from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi(); api.authenticate()
for b in ['auc','sp-sr','mp-sr','rpe','mppe','cd-o']:
    c=f'v2d-challenge-track3-{b}'
    subs=api.competition_submissions(c); lim=api.competition_get_submission_limits(c)
    print(b,'allowed_now',getattr(lim,'num_allowed_now',None),'today',getattr(lim,'num_today',None),'| latest:',[(str(s.date)[5:16],getattr(s,'submitted_by_ref',None),s.public_score,(s.description or '')[:60]) for s in subs[:2]])
