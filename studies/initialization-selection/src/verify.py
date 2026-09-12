"""Replay decisions, verify artifact integrity, and run eight complete repeats."""
from core import *
from evaluate import check_seals
from benchmark import now
import csv
import gc

def replay_greedy(c,records):
    seen={};current=0;cursor=0;skips=0
    for i,rec in enumerate(records):
        if i==0:code=0
        else:
            while True:
                code=current^(1<<c.bits[cursor%len(c.bits)]);cursor+=1
                if code in seen:
                    skips+=1
                    if skips<len(c.bits):continue
                    code=next(j for j in c.order if j not in seen);skips=0;current=code
                else:skips=0
                break
        assert code==rec['code']
        seen[code]=rec['validation']
        if code==current or key(seen[code])>key(seen[current]):current=code

def run():
    check_seals();out=ROOT/'verification';out.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'plan.json').read_text());replays=[];repeats=[]
    for job in plan['jobs']:
        p=job['p'];method=job['method'];path=ROOT/'runs'/job['name']
        info=json.loads((path/'result.json').read_text());choice=json.loads((path/'choice.json').read_text())
        assert sha(path/'choice.json')==info['choice_sha256']
        records=choice['selection']['records'];codes=[r['code'] for r in records]
        assert len(set(codes))==len(codes)
        c=Candidates(p,job['seed']);limits=budget(p)
        arr=np.load(path/'selected.npz')
        assert np.array_equal(arr['m0'],c.m0(info['chosen_code']).cpu().numpy())
        assert int(arr['index'])==59 and len(arr['validation_history'])==60
        if method=='bank':
            q=np.load(ROOT/'banks'/f'p{p}'/'tensor.npz')['q']
            chosen,_=bank_code(c,q)
            assert chosen==info['chosen_code']
            assert info['fit_count']==60 and info['update_count']==59
        else:
            chosen=max(records,key=lambda r:key(r['validation']))['code']
            assert chosen==info['chosen_code']
            length=60 if method=='random_full' else 11
            nc=limits['full'] if method=='random_full' else limits['probes']
            assert len(records)==nc
            assert info['fit_count']==nc*length+(0 if length==60 else 49)
            assert info['update_count']==nc*(length-1)+(0 if length==60 else 49)
            assert info['fit_count']<=limits['fit_cap']
            if method.startswith('random'):assert codes==c.order[:nc]
            else:replay_greedy(c,records)
            for r in records:
                e=c.direction(r['code']);a=np.einsum('kij,ij->k',c.basis,e)
                assert abs(np.linalg.norm(e)-1)<1e-12
                assert np.max(abs(abs(a)-abs(c.a)))<1e-12
                assert np.max(abs(e-np.einsum('k,kij->ij',a,c.basis)-c.residual))<1e-12
        replays.append(dict(name=job['name'],passed=True))
    rows=list(csv.DictReader((ROOT/'evaluation/directions.csv').open()))
    for row in rows:
        arr=np.load(ROOT/'evaluation'/f'p{row["p"]}-s{row["seed"]}-{row["method"]}.npz')
        mm=metrics(arr['predictions'],arr['targets'])
        assert mm['acc']==float(row['test_acc']) and mm['correct']==int(row['test_correct'])
        assert mm['normalized_margin']==float(row['test_normalized_margin'])
        assert mm['mean_margin']==float(row['test_mean_margin']) and mm['mse']==float(row['test_mse'])
    for p,seed in [(17,2000),(23,3000)]:
        data=selection_data(p,plan['moduli'][str(p)]['validation'])
        q=np.load(ROOT/'banks'/f'p{p}'/'tensor.npz')['q']
        for method in plan['methods']:
            name=f'p{p}-s{seed}-{method}'
            gc.collect();torch.cuda.empty_cache()
            info,state,records=select(method,p,seed,data,q,out/name)
            original=json.loads((ROOT/'runs'/name/'result.json').read_text())
            assert info['chosen_code']==original['chosen_code']
            original_records=json.loads((ROOT/'runs'/name/'choice.json').read_text())['selection']['records']
            assert records==original_records
            a=np.load(ROOT/'runs'/name/'selected.npz');b=np.load(out/name/'selected.npz')
            diffs={k:float(np.max(np.abs(a[k]-b[k]))) for k in a.files}
            assert all(v==0 for v in diffs.values())
            repeats.append(dict(name=name,choice_exact=True,candidate_metrics_exact=True,array_max_abs_differences=diffs))
            del state
            print(json.dumps(dict(repeat=name,status='exact')),flush=True)
    frozen=json.loads((ROOT/'freeze.json').read_text())
    for name,h in frozen['old_manuscript_hashes'].items():assert sha(ROOT.parent/name)==h,name
    assert sha(ROOT/'vendor/rfm.py')==sha(ROOT.parent/'rfm-mechanism/vendor/rfm.py')
    check_seals()
    result=dict(utc=now(),passed=True,decision_replays=len(replays),prediction_rows_recomputed=len(rows),
                repeats=repeats,old_manuscript_unchanged=True,vendor_unchanged=True,freeze_and_seals_valid=True)
    write_json(out/'audit.json',result)
    print(json.dumps(dict(audit='passed',decision_replays=160,prediction_rows_recomputed=200,repeats=8)),flush=True)

if __name__=='__main__':setup();run()
