"""Generate all design decisions without running confirmation directions."""
from core import *
from datetime import datetime, timezone

if __name__ == '__main__':
    if (ROOT/'plan.json').exists():
        raise FileExistsError('Plan already exists')
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), n=20,
                methods=['bank','random_probe','greedy_probe','random_full'], moduli={}, jobs=[])
    for p, start, nv in [(17,2000,6),(23,3000,8)]:
        rng = np.random.default_rng(np.random.SeedSequence([20260909,p,81]))
        v = sorted(int(a) for a in rng.permutation(p)[:nv])
        plan['moduli'][str(p)] = dict(p=p, validation=v, test=[a for a in range(p) if a not in v],
                                     seeds=list(range(start,start+20)), budget=budget(p))
        for seed in range(start,start+20):
            rng = np.random.default_rng(np.random.SeedSequence([20260909,p,seed,82]))
            for method in rng.permutation(plan['methods']):
                plan['jobs'].append(dict(p=p,seed=seed,method=str(method),name=f'p{p}-s{seed}-{method}'))
    write_json(ROOT/'plan.json',plan)
    print(json.dumps(plan['moduli']),flush=True)
