import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
def profile(pred):
    p=len(pred);v=np.array([np.roll(pred[a],-2*a) for a in range(p)]).mean(0);return v-v.mean()
def interval(x,seed=20260906):
    x=np.asarray(x);rng=np.random.default_rng(seed);means=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(means,[.025,.975]).tolist())
def main():
    forecasts=json.loads((ROOT/'frozen/forecasts.json').read_text());predictions=np.load(ROOT/'frozen/forecast_profiles.npz');rows=[]
    for r in forecasts:
        name=f"p{r['p']}_s{r['seed']}_cross";path=ROOT/'results/confirmation'/name
        s=json.loads((path/'summary.json').read_text());assert s['status']=='completed';a=np.load(path/'final.npz');obs=profile(a['pred']);pred=predictions[name]
        acc=float(np.mean(a['pred'].argmax(1)==a['y'].argmax(1)));early=(a['history'].argmax(2)==a['y'].argmax(1)).mean(1)
        rows.append(dict(**r,actual_acc=acc,absolute_error=abs(acc-r['predicted_acc']),constant_error=abs(acc-r['constant_acc']),
                         profile_cosine=float(np.dot(pred,obs)/(np.linalg.norm(pred)*np.linalg.norm(obs))),
                         profile_relative_l2=float(np.linalg.norm(pred-obs)/np.linalg.norm(obs)),raw_step5_acc=float(early[5]),raw_step10_acc=float(early[10])))
    metrics={}
    for p in [17,23]:
        rr=[r for r in rows if r['p']==p];y=np.array([r['actual_acc'] for r in rr]);a=np.array([r['absolute_error'] for r in rr]);b=np.array([r['constant_error'] for r in rr])
        metrics[str(p)]=dict(n=len(rr),measured_accuracy=interval(y),mae=interval(a),constant_mae=interval(b),mae_improvement=interval(b-a),
                             mse=float(np.mean(a*a)),constant_mse=float(np.mean(b*b)),
                             spearman=float(spearmanr([r['signal'] for r in rr],y).statistic),
                             success_count=int((y>=.9).sum()),failure_count=int((y<=.1).sum()),
                             success_agreement=float(np.mean([(r['predicted_acc']>=.9)==(r['actual_acc']>=.9) for r in rr])),
                             profile_cosine_range=[float(min(r['profile_cosine'] for r in rr)),float(max(r['profile_cosine'] for r in rr))],
                             profile_relative_l2_median=float(np.median([r['profile_relative_l2'] for r in rr])),
                             early_step5_mae=float(np.mean([abs(r['raw_step5_acc']-r['actual_acc']) for r in rr])),
                             early_step10_mae=float(np.mean([abs(r['raw_step10_acc']-r['actual_acc']) for r in rr])))
    with (ROOT/'results/confirmation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'results/confirmation_analysis.json').write_text(json.dumps(dict(metrics=metrics,rows=rows),indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
