"""Post-hoc manuscript revision audit. No training, refitting, or forecast changes.

Extract once: python revision_analysis.py --extract ../rfm-mechanism
Recompute:    python revision_analysis.py --inputs revision-analysis/inputs
Requires numpy, scipy, matplotlib; writes only to --output.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse, csv, hashlib, itertools, json, shutil
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
def dump(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False), encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p, rows):
    with p.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def csvread(p):
    return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(p.open())]
def profile(pred):
    v=np.mean([np.roll(row,-2*a) for a,row in enumerate(pred)],axis=0)
    return v-v.mean()
def margin(v): return float(v[0]-max(v[1:]))
def rms(v): return float(np.sqrt(np.mean(v*v)))
def cosine(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
def z(v): return margin(v)/rms(v)
def stats(x):
    a=np.asarray(x);return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
        min=float(a.min()),q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),
        q90=float(np.quantile(a,.9)),max=float(a.max()))
def boot(rows, key):
    # Exploratory paired cluster bootstrap; same seed carries both modulus rows.
    seeds=sorted({r['seed'] for r in rows});rng=np.random.default_rng(20260907)
    sums=np.array([sum(r[key] for r in rows if r['seed']==s) for s in seeds])
    sizes=np.array([sum(r['seed']==s for r in rows) for s in seeds])
    ix=rng.integers(len(seeds),size=(10000,len(seeds)))
    return np.quantile(sums[ix].sum(1)/sizes[ix].sum(1),[.025,.975]).tolist()
def extract(root, dst):
    dst.mkdir(parents=True,exist_ok=True); arrays={}; manifest={}
    def track(path): manifest[path.relative_to(root).as_posix()]=sha(path)
    for p in [17,23]:
        path=root/f'frozen/quadratic_p{p}.npz';track(path)
        with np.load(path) as qd:
            for k in qd.files: arrays[f'p{p}_{k}']=qd[k]
    for name in ['confirmation.csv','intervention.csv','intervention_analysis.json','quadratic_discovery.json']:
        path=root/'results'/name;track(path);shutil.copy2(path,dst/name)
    path=root/'frozen/calibration_candidate.json';track(path);shutil.copy2(path,dst/path.name)
    specs=[]
    for p in [17,23]:
        for seed in range(100,150 if p==17 else 130):
            specs.append(('confirmation',f'p{p}_s{seed}_cross',p,False))
        for seed in range(100,120 if p==17 else 110):
            for family in ['phase_best','phase_worst','remove']:
                for suffix in ['','_control0','_control1','_control2']:
                    specs.append(('intervention',f'p{p}_s{seed}_{family}{suffix}',p,False))
        for seed in range(100,110):
            for variant in ['eps001','eps03']:
                specs.append(('boundary',f'p{p}_s{seed}_{variant}',p,False))
        for name in ['none','mode1','mode1_half']:
            specs.append(('operator',f'p{p}_{name}',p,True))
    for phase,name,p,history in specs:
        path=root/f'results/{phase}/{name}/final.npz';track(path)
        with np.load(path) as data:
            arrays[name+'_pred']=data['pred']
            if phase in ['confirmation','intervention']:
                e=(data['m0']-np.eye(2*p))/(.01*np.sqrt(2*p))
                arrays[name+'_a']=np.einsum('kij,ij->k',arrays[f'p{p}_basis'],e)
            if history: arrays[name+'_history']=data['history']
    np.savez_compressed(dst/'scores_and_coordinates.npz',**arrays)
    dump(dst/'provenance.json',dict(status='post hoc extraction from completed, frozen study; no new training',
        transformations='pred and history copied verbatim; a_k=<basis_k,(m0-I)/(.01 sqrt(2p))>; fitted tensors copied unchanged',
        original_files_sha256=manifest,extracted_array_sha256=sha(dst/'scores_and_coordinates.npz')))

def analyze(inputs, out):
    out.mkdir(parents=True,exist_ok=True)
    arr=np.load(inputs/'scores_and_coordinates.npz'); cr=csvread(inputs/'confirmation.csv'); ir=csvread(inputs/'intervention.csv')
    cal=json.loads((inputs/'calibration_candidate.json').read_text()); primary=json.loads((inputs/'intervention_analysis.json').read_text())
    def forecast(p,a): return .01**2*np.einsum('i,cij,j->c',a,arr[f'p{p}_q'],a)
    def predacc(v): return float(np.interp(z(v),cal['x'],cal['y']))
    confirmation=[]
    for r in cr:
        p,s=int(r['p']),int(r['seed']);name=f'p{p}_s{s}_cross'
        scores=arr[name+'_pred'];v=profile(scores);hat=forecast(p,arr[name+'_a'])
        aligned=np.array([np.roll(row,-2*a) for a,row in enumerate(scores)])
        pointmargin=aligned[:,0]-aligned[:,1:].max(1)
        assert abs(predacc(hat)-r['predicted_acc'])<1e-12
        confirmation.append(dict(p=p,seed=s,actual_acc=r['actual_acc'],predicted_acc=r['predicted_acc'],
            accuracy_error_pp=100*r['absolute_error'],error_test_quanta=p*r['absolute_error'],
            observed_profile_margin=margin(v),predicted_profile_margin=margin(hat),
            observed_normalized_profile_margin=z(v),predicted_normalized_profile_margin=z(hat),
            mean_pointwise_margin=float(pointmargin.mean()),profile_rms=rms(v),
            profile_margin_error=margin(hat)-margin(v)))
    table(out/'confirmation_continuous.csv',confirmation)
    confirm={};threshold=[]
    for p in [17,23]:
        rows=[r for r in confirmation if r['p']==p]
        obs=np.array([r['observed_profile_margin'] for r in rows]);hat=np.array([r['predicted_profile_margin'] for r in rows])
        point=np.array([r['mean_pointwise_margin'] for r in rows])
        error=np.array([r['error_test_quanta'] for r in rows])
        confirm[str(p)]=dict(error_pp=stats([r['accuracy_error_pp'] for r in rows]),
            error_quanta=stats(error),within_one_quantum=int(sum(error<=1+1e-12)),
            profile_margin_mae=float(abs(obs-hat).mean()),profile_margin_rmse=float(np.sqrt(np.mean((obs-hat)**2))),
            profile_margin_spearman=float(spearmanr(obs,hat).statistic),
            normalized_margin_mae=float(np.mean([abs(r['observed_normalized_profile_margin']-r['predicted_normalized_profile_margin']) for r in rows])),
            pointwise_vs_profile_gap=stats(obs-point),
            profile_margin_sign_agreement=int(sum((obs>0)==(hat>0))))
        for t in [.8,.85,.9,.95]:
            actual=np.array([r['actual_acc']>=t for r in rows]);pred=np.array([r['predicted_acc']>=t for r in rows])
            threshold.append(dict(p=p,threshold=t,required_correct=int(np.ceil(p*t-1e-12)),n=len(rows),
                agreement=int(sum(actual==pred)),always_success_agreement=int(sum(actual)),
                false_positive=int(sum(pred&~actual)),false_negative=int(sum(~pred&actual))))
    table(out/'threshold_sensitivity.csv',threshold)
    controls=[]; adverse=[]
    for r in ir:
        p,s=int(r['p']),int(r['seed']);prefix=f'p{p}_s{s}_'
        va=profile(arr[prefix+'cross_pred']);vb=profile(arr[prefix+'phase_best_pred']);vw=profile(arr[prefix+'phase_worst_pred'])
        adverse.append(dict(p=p,seed=s,worst_acc=r['phase_worst'],uniform_chance=1/p,
            base_margin=margin(va),best_margin=margin(vb),worst_margin=margin(vw),
            best_z=z(vb),worst_z=z(vw),worst_rms=rms(vw),best_rms=rms(vb),
            worst_to_best_rms=rms(vw)/rms(vb),worst_vs_negative_best_cosine=cosine(vw,-vb),
            worst_correct_centered_score=float(vw[0]),worst_top_offset=int(vw.argmax())))
        a=arr[prefix+'cross_a'];basepred=predacc(forecast(p,a));basez=z(forecast(p,a))
        for family in ['phase_best','phase_worst']:
            b=arr[prefix+family+'_a'];preds=[];zs=[]
            for j in range(3):
                v=forecast(p,arr[prefix+family+f'_control{j}_a']);preds.append(predacc(v));zs.append(z(v))
            controls.append(dict(p=p,seed=s,family=family,base_acc=r['base_acc'],
                control_acc=r[family+'_control_mean'],control_gain=r[family+'_control_mean']-r['base_acc'],
                intervention_acc=r[family],net_effect=r[family+'_effect'],
                matched_angle_degrees=float(np.degrees(np.arccos(np.clip(a@b/(a@a),-1,1)))),
                base_predicted_acc=basepred,control_predicted_acc=float(np.mean(preds)),
                predicted_control_gain=float(np.mean(preds))-basepred,
                base_z=basez,control_mean_z=float(np.mean(zs))))
    table(out/'control_gains.csv',controls);table(out/'adverse_profiles.csv',adverse)
    controlsummary={}
    for family in ['phase_best','phase_worst']:
        controlsummary[family]={}
        for group in ['all',17,23]:
            rows=[r for r in controls if r['family']==family and (group=='all' or r['p']==group)]
            controlsummary[family][str(group)]=dict(n=len(rows),
                gain=stats([r['control_gain'] for r in rows]),gain_posthoc_cluster_ci95=boot(rows,'control_gain'),
                improved=sum(r['control_gain']>1e-12 for r in rows),worsened=sum(r['control_gain']< -1e-12 for r in rows),
                angle=stats([r['matched_angle_degrees'] for r in rows]),
                predicted_gain=stats([r['predicted_control_gain'] for r in rows]),
                predicted_gain_mae=float(np.mean([abs(r['predicted_control_gain']-r['control_gain']) for r in rows])),
                gain_spearman=float(spearmanr([r['control_gain'] for r in rows],[r['predicted_control_gain'] for r in rows]).statistic))
    ads={str(p):dict(worst_acc=stats([r['worst_acc'] for r in adverse if r['p']==p]),
        wrong_profile_max=sum(r['worst_margin']<0 for r in adverse if r['p']==p),
        worst_z=stats([r['worst_z'] for r in adverse if r['p']==p]),
        amplitude_ratio=stats([r['worst_to_best_rms'] for r in adverse if r['p']==p]),
        negative_best_cosine=stats([r['worst_vs_negative_best_cosine'] for r in adverse if r['p']==p])) for p in [17,23]}
    # A unit u relabels frequency k to k/u, folding negative sine modes with sign.
    structure={};sr=[];eigarrays={};covrows=[]
    for p in [17,23]:
        q=arr[f'p{p}_q'];basis=arr[f'p{p}_basis'];d=q.shape[1];errors=[];basis_errors=[]
        for u in range(1,p):
            U=np.zeros((d,d))
            for k in range(1,d+1):
                j=k*pow(u,-1,p)%p;U[(j if j<=d else p-j)-1,k-1]=1 if j<=d else -1
            indices=pow(u,-1,p)*np.arange(p)%p;indices=np.r_[indices,p+indices]
            transformed=basis[:,indices][:,:,indices]
            represented=np.einsum('ik,imn->kmn',U,basis)
            basis_error=float(np.max(abs(transformed-represented)))
            assert basis_error<1e-12,(p,u,basis_error)
            basis_errors.append(basis_error)
            diff=np.einsum('ij,cjk,lk->cil',U,q,U)-q[(u*np.arange(p))%p]
            err=float(np.linalg.norm(diff)/np.linalg.norm(q));errors.append(err)
            covrows.append(dict(p=p,unit=u,relative_tensor_covariance_error=err))
        for c in range(p):
            vals,vecs=np.linalg.eigh(q[c]);order=np.argsort(abs(vals))[::-1];vals,vecs=vals[order],vecs[:,order]
            energy=vals**2;cum=np.cumsum(energy)/energy.sum();lead=vecs[:,0]**2
            eigarrays[f'p{p}_c{c}_eigenvalues']=vals;eigarrays[f'p{p}_c{c}_eigenvectors']=vecs
            sr.append(dict(p=p,c=c,rank_relative_1e8=int(sum(abs(vals)>max(abs(vals))*1e-8)),
                rank_energy90=int(np.searchsorted(cum,.9)+1),rank_energy95=int(np.searchsorted(cum,.95)+1),
                rank_energy99=int(np.searchsorted(cum,.99)+1),stable_rank=float(energy.sum()/energy.max()),
                top_eigenvalue=float(vals[0]),lead_largest_additive_frequency_weight=float(max(lead)),
                lead_participation=float(1/sum(lead**2))))
        ss=np.linalg.svd(q.reshape(p,-1),compute_uv=False);energy=ss**2
        # Q0 should commute with unit actions if Q were an exact equivariant Hessian.
        structure[str(p)]=dict(d=d,reflection_relative_error=float(np.linalg.norm(q-q[(-np.arange(p))%p])/np.linalg.norm(q)),
            signed_permutation_basis_max_error=max(basis_errors),
            output_flatten_rank_relative_1e8=int(sum(ss>max(ss)*1e-8)),
            unit_covariance_error=stats(errors),diagonal_energy_fraction=float(np.sum(np.einsum('cii->ci',q)**2)/np.sum(q*q)),
            output_flatten_rank95=int(np.searchsorted(np.cumsum(energy)/energy.sum(),.95)+1),
            output_flatten_singular_values=ss.tolist(),
            rank95_range=[min(r['rank_energy95'] for r in sr if r['p']==p),max(r['rank_energy95'] for r in sr if r['p']==p)],
            leading_participation_range=[min(r['lead_participation'] for r in sr if r['p']==p),max(r['lead_participation'] for r in sr if r['p']==p)])
    table(out/'tensor_spectra.csv',sr);table(out/'tensor_covariance.csv',covrows)
    np.savez_compressed(out/'tensor_eigensystems.npz',**eigarrays)
    amplitude=[]
    for p in [17,23]:
        for t in range(60):
            base=profile(arr[f'p{p}_none_history'][t]);v=profile(arr[f'p{p}_mode1_history'][t])-base
            half=profile(arr[f'p{p}_mode1_half_history'][t])-base
            amplitude.append(dict(p=p,kind='pure_mode1',seed=-1,iteration=t,epsilon=.001,
                scaled_profile_relative_error=float(np.linalg.norm(4*half-v)/max(np.linalg.norm(v),1e-300)),reference_profile_norm=float(np.linalg.norm(v))))
        for s in range(100,110):
            base=profile(arr[f'p{p}_s{s}_cross_pred'])
            for tag,eps in [('eps001',.001),('eps03',.03)]:
                v=profile(arr[f'p{p}_s{s}_{tag}_pred'])
                amplitude.append(dict(p=p,kind='general_cross',seed=s,iteration=59,epsilon=eps,
                    scaled_profile_relative_error=float(np.linalg.norm(v*(.01/eps)**2-base)/np.linalg.norm(base)),reference_profile_norm=float(np.linalg.norm(base))))
    table(out/'amplitude_checks.csv',amplitude)
    amps={str(p):{str(eps):stats([r['scaled_profile_relative_error'] for r in amplitude if r['p']==p and r['kind']=='general_cross' and r['epsilon']==eps]) for eps in [.001,.03]} for p in [17,23]}
    scaling=[];reference=37*(17*16)**3
    for p in [17,23,29,41,61,97]:
        d=(p-1)//2;n=p*(p-1);b=1+d*(d+1)//2
        scaling.append(dict(p=p,d=d,train_points=n,bank_trajectories=b,kernel_evaluations=60*b,
            dense_factor_work_relative_p17=b*n**3/reference,single_fp64_kernel_gib=8*n*n/2**30,
            exhaustive_relative_signs=2**(d-1)))
    table(out/'cost_extrapolation.csv',scaling)
    summary=dict(status='Post-hoc analyses added after confirmation and primary comparisons; no new training or fitting',
        bootstrap='10000 paired integer-seed cluster resamples, RNG 20260907; exploratory, unadjusted intervals',
        confirmation=confirm,controls=controlsummary,adverse=ads,tensor=structure,amplitude=amps,
        primary_intervention_metrics=primary['metrics'],
        inputs_sha256={p.name:sha(p) for p in sorted(inputs.iterdir()) if p.is_file()})
    dump(out/'analysis.json',summary)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,grid=plt.subplots(2,2,figsize=(8.6,6.8),layout='constrained');axes=grid.ravel()
    colors={17:'#0072B2',23:'#D55E00'}
    for p in [17,23]:
        rows=[r for r in confirmation if r['p']==p];err=np.sort([r['error_test_quanta'] for r in rows])
        axes[0].step(err,np.arange(1,len(err)+1)/len(err),where='post',label=f'p={p}',color=colors[p])
        axes[1].scatter([r['observed_profile_margin'] for r in rows],[r['predicted_profile_margin'] for r in rows],s=18,alpha=.7,color=colors[p])
    axes[0].axvline(1,ls=':',color='gray');axes[0].set(xlabel='Absolute accuracy error / (1/p)',ylabel='Fraction of directions');axes[0].legend()
    lims=[min(axes[1].get_xlim()[0],axes[1].get_ylim()[0]),max(axes[1].get_xlim()[1],axes[1].get_ylim()[1])]
    axes[1].plot(lims,lims,'--',color='gray');axes[1].set(xlabel='Measured mean-profile margin',ylabel='Predicted mean-profile margin')
    axes[2].semilogy([r['p'] for r in scaling],[r['dense_factor_work_relative_p17'] for r in scaling],'o-',color='#009E73')
    axes[2].set(xlabel='Modulus p',ylabel='Dense-factor work / p=17',title='Analytic extrapolation; not timing')
    for p in [17,23]:
        rows=[r for r in amplitude if r['p']==p and r['kind']=='pure_mode1']
        axes[3].plot([r['iteration'] for r in rows],[100*r['scaled_profile_relative_error'] for r in rows],color=colors[p],label=f'p={p}')
    axes[3].set(xlabel='Evaluation index t',ylabel='Half-amplitude scaling error (%)');axes[3].legend()
    for ax,label in zip(axes,['(a)','(b)','(c)','(d)']):ax.text(0,1.04,label,transform=ax.transAxes)
    fig.savefig(out/'06_revision_diagnostics.png',dpi=220);plt.close(fig)
    print(json.dumps({k:summary[k] for k in ['confirmation','controls','adverse','tensor','amplitude']},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--extract',type=Path);ap.add_argument('--inputs',type=Path,default=HERE/'revision-analysis/inputs');ap.add_argument('--output',type=Path,default=HERE/'revision-analysis')
    args=ap.parse_args()
    if args.extract: extract(args.extract.resolve(),args.inputs.resolve())
    analyze(args.inputs.resolve(),args.output.resolve())
