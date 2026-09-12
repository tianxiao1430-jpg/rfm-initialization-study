"""Post-review endpoint stratification and saved-state coverage; no model training."""
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'rfm-mechanism'

def write(name,rows):
    with (HERE/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    with (HERE/'run_summary.csv').open(encoding='utf-8') as f:
        runs=[r for r in csv.DictReader(f) if r['phase']=='confirmation']
    with (HERE/'sequence_summary.csv').open(encoding='utf-8') as f:
        saved={(r['name'],int(r['test_a'])):r for r in csv.DictReader(f)
               if r['phase']=='confirmation' and r['view']=='point'}
    points=[];coverage=[];inputs=[]
    for run in runs:
        name=run['name'];p=int(run['p']);folder=SOURCE/'results/confirmation'/name
        for filename in ['final.npz','states.npz']:
            path=folder/filename
            inputs.append({'path':path.relative_to(HERE.parent.parent).as_posix(),
                           'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        with np.load(folder/'final.npz',allow_pickle=False) as z:h=z['history']
        with np.load(folder/'states.npz',allow_pickle=False) as z:
            steps=z['steps'].tolist()
            assert len(steps)==len(z['matrices'])
        coverage.append(dict(name=name,p=p,saved_steps='|'.join(map(str,steps)),
                             has_step8=int(8 in steps),has_step9=int(9 in steps)))
        for a in range(p):
            s=h[:,a,:];truth=2*a%p
            margin=s[:,truth]-np.delete(s,truth,axis=1).max(axis=1)
            tolerance=np.maximum(1e-8*np.std(s,axis=1),64*np.finfo(float).eps*np.abs(s).max(axis=1))
            state=np.where(margin>tolerance,1,np.where(margin < -tolerance,-1,0))
            definite=state[state!=0]
            wc=int(np.sum((definite[:-1]==-1)&(definite[1:]==1)))
            cw=int(np.sum((definite[:-1]==1)&(definite[1:]==-1)))
            old=saved[name,a]
            assert wc==int(old['wrong_to_correct']) and cw==int(old['correct_to_wrong'])
            assert state[-1]==int(old['final_correct_state'])
            points.append(dict(name=name,p=p,test_a=a,initial_state=int(state[0]),final_state=int(state[-1]),
                ever_correct=int(np.any(state==1)),ever_ambiguous=int(np.any(state==0)),
                wrong_to_correct=wc,correct_to_wrong=cw,correctness_flips=wc+cw,
                first_correct_step=int(np.flatnonzero(state==1)[0]) if np.any(state==1) else ''))
    assert len(points)==1540 and all(r['initial_state']==-1 for r in points)
    summaries=[]
    for p in [17,23,'all']:
        for end in [1,-1,0]:
            rr=[r for r in points if (p=='all' or r['p']==p) and r['final_state']==end]
            summaries.append(dict(p=p,final_state=end,points=len(rr),
                trajectories=len({r['name'] for r in rr}),initially_wrong=sum(r['initial_state']==-1 for r in rr),
                ever_correct=sum(r['ever_correct'] for r in rr),
                zero_flips=sum(r['correctness_flips']==0 for r in rr),
                one_flip=sum(r['correctness_flips']==1 for r in rr),
                multiple_flips=sum(r['correctness_flips']>1 for r in rr),
                wrong_to_correct=sum(r['wrong_to_correct'] for r in rr),
                correct_to_wrong=sum(r['correct_to_wrong'] for r in rr)))
    correct=next(r for r in summaries if r['p']=='all' and r['final_state']==1)
    wrong=next(r for r in summaries if r['p']=='all' and r['final_state']==-1)
    assert correct['points']==correct['one_flip']==correct['ever_correct']==1254
    assert wrong['points']==wrong['zero_flips']==286 and wrong['ever_correct']==0
    assert all(r['multiple_flips']==0 for r in summaries)
    assert all(r['saved_steps']=='0|1|2|5|10|59' for r in coverage)
    write('endpoint_point_history.csv',points)
    write('endpoint_stratification.csv',summaries)
    write('saved_metric_coverage.csv',coverage)
    (HERE/'review_input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf-8')
    facts={'passed':True,'primary_points':1540,'all_initially_wrong':True,
        'final_correct_points':1254,'final_correct_each_exactly_one_flip':True,
        'final_wrong_points':286,'final_wrong_ever_clearly_correct':0,'points_with_multiple_correctness_flips':0,
        'primary_runs':80,'metric_snapshots_per_run':[0,1,2,5,10,59],
        'runs_with_step8_or_step9_metric_snapshot':0,
        'limits':'Post hoc endpoint strata are descriptive, not independent groups or causal explanations. Correctness flips differ from wrong-class leader replacements. No step-8/9 metric snapshots exist in this archive.'}
    (HERE/'review_addendum_facts.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(facts))

if __name__=='__main__':main()
