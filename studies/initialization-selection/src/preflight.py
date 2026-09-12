"""Independent old-engine and exact saved-checkpoint continuation checks."""
from core import *
import importlib.util

if __name__ == '__main__':
    setup()
    out=ROOT/'preflight'
    out.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'plan.json').read_text())
    checks=[]
    for p in [17,23]:
        c=Candidates(p,900000+p)
        errors=dict(norm=0.,residual=0.,magnitudes=0.,symmetry=0.)
        for code in range(c.size):
            e=c.direction(code)
            a=np.einsum('kij,ij->k',c.basis,e)
            r=e-np.einsum('k,kij->ij',a,c.basis)
            errors['norm']=max(errors['norm'],abs(np.linalg.norm(e)-1))
            errors['residual']=max(errors['residual'],float(np.max(np.abs(r-c.residual))))
            errors['magnitudes']=max(errors['magnitudes'],float(np.max(np.abs(abs(a)-abs(c.a)))))
            errors['symmetry']=max(errors['symmetry'],float(np.max(np.abs(e-e.T))))
            assert np.linalg.eigvalsh(np.eye(2*p)+.01*np.sqrt(2*p)*e).min()>0
        assert max(errors.values())<1e-12
        data=selection_data(p,plan['moduli'][str(p)]['validation'])
        state=State(c.m0(1)); hist,_=advance(state,60,data)
        short=State(c.m0(1)); early,_=advance(short,11,data)
        save_state(out/f'p{p}-checkpoint.npz',short)
        loaded=load_state(out/f'p{p}-checkpoint.npz')
        late,_=advance(loaded,60,data)
        assert np.array_equal(hist,np.concatenate([early,late]))
        assert torch.equal(state.m,loaded.m) and torch.equal(state.alpha,loaded.alpha)
        checks.append(dict(p=p,invariant_errors=errors,resume_exact=True,candidates=c.size))
    oldroot=ROOT.parent/'rfm-mechanism'
    sys.path.insert(0,str(oldroot/'src'))
    spec=importlib.util.spec_from_file_location('old_experiment',oldroot/'src/experiment_v3.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    result=old.run(dict(p=17,seed=900017,kind='cross'),out/'old-engine')
    assert result['status']=='completed'
    arr=np.load(out/'old-engine/final.npz')
    # Identical validation width to old engine isolates implementation equality.
    data=selection_data(17,list(range(17)))
    state=State(Candidates(17,900017).m0(0)); hist,_=advance(state,60,data)
    assert np.array_equal(hist,arr['history']) and np.array_equal(state.m.cpu().numpy(),arr['m'])
    assert sha(ROOT/'vendor/rfm.py')==sha(oldroot/'vendor/rfm.py')
    write_json(out/'checks.json',dict(passed=True,checks=checks,old_engine_exact=True,
                                     vendor_exact=True,environment=rfm.environment()))
    print(json.dumps(dict(preflight='passed',checks=checks)),flush=True)
