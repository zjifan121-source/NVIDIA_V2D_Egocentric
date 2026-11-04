import json, glob, sys, re
for pattern in sys.argv[1:]:
    for f in glob.glob(pattern):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n = f.split('/')[-2]
        m = re.match(r'ep(\d+)', n)
        if not m:
            continue
        ep = int(m.group(1))
        limit = 70e6 if ep in (25, 28, 32, 33) else 1e12
        s = [x for x in d if x['step'] >= limit]
        if s:
            b = max(s, key=lambda x: x['auc'])
            print(n, 'best', round(b['auc'], 3), 'at', b['step'] // 1000000, 'M', [round(v, 2) for v in b['per_body']])
