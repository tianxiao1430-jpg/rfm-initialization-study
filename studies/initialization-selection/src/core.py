"""Numerical engine and selectors. No held-out test objects enter this module.

The GPL-3.0 kernel and batch-of-two AGOP order are inherited from the previous
study. Scores used for selection are explicitly validation-only.
"""
from __future__ import annotations
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
extra = Path(os.environ.get('RFM_EXTRA_DEPS', str(ROOT.parents[1] / 'work/python-deps')))
if extra.is_dir():
    sys.path.insert(0, str(extra))
sys.path.insert(0, str(ROOT / 'vendor'))
import rfm
import numpy as np
import torch
import hashlib
import json
import time
from dataclasses import dataclass


def setup():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('This measured protocol requires CUDA')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def inputs(p, pairs):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    eye = np.eye(p)
    x = np.concatenate([eye[pairs[:, 0]], eye[pairs[:, 1]]], axis=1)
    y = eye[(pairs[:, 0] + pairs[:, 1]) % p]
    return torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()


def selection_data(p, val_indices):
    x, y = inputs(p, [(a, b) for a in range(p) for b in range(p) if a != b])
    xv, yv = inputs(p, [(a, a) for a in val_indices])
    return x, y, xv, yv


def metrics(scores, targets):
    pred = np.asarray(scores)
    truth = np.asarray(targets).argmax(1)
    centered = pred - pred.mean(1, keepdims=True)
    rms = np.sqrt((centered**2).mean(1))
    other = pred.copy()
    other[np.arange(len(truth)), truth] = -np.inf
    gap = pred[np.arange(len(truth)), truth] - other.max(1)
    correct = pred.argmax(1) == truth
    normalized = gap / np.maximum(rms, np.finfo(np.float64).tiny)
    return dict(acc=float(correct.mean()), correct=int(correct.sum()), n=len(truth),
                mean_margin=float(gap.mean()), normalized_margin=float(normalized.mean()),
                mse=float(((pred-targets)**2).mean()))


def key(row):
    return row['acc'], row['normalized_margin']


def basis(p):
    ix = np.arange(p)
    out = []
    for k in range(1, (p+1)//2):
        b = np.sin(2*np.pi*k*(ix[None, :] - ix[:, None])/p)
        e = np.block([[np.zeros((p, p)), b], [b.T, np.zeros((p, p))]])
        out.append(e / np.linalg.norm(e))
    return np.stack(out)


def cross_direction(p, seed):
    e = rfm.perturbation(p, seed, 'break')
    e[:p, :p] = 0
    e[p:, p:] = 0
    e /= torch.linalg.norm(e)
    return e.numpy()


class Candidates:
    def __init__(self, p, seed):
        self.p, self.seed = p, seed
        self.basis = basis(p)
        self.original = cross_direction(p, seed)
        self.a = np.einsum('kij,ij->k', self.basis, self.original)
        self.residual = self.original - np.einsum('k,kij->ij', self.a, self.basis)
        self.size = 2**(len(self.a)-1)
        rng = np.random.default_rng(np.random.SeedSequence([20260909, p, seed, 71]))
        self.order = [0] + [int(x) for x in rng.permutation(np.arange(1, self.size))]
        rng = np.random.default_rng(np.random.SeedSequence([20260909, p, seed, 72]))
        self.bits = [int(x) for x in rng.permutation(len(self.a)-1)]

    def coords(self, code):
        s = np.ones(len(self.a))
        for bit in range(len(self.a)-1):
            if (code >> bit) & 1:
                s[bit+1] = -1
        return self.a*s

    def direction(self, code):
        if code == 0:
            return self.original.copy()
        return self.residual + np.einsum('k,kij->ij', self.coords(code), self.basis)

    def m0(self, code):
        return initializer(self.p, self.direction(code), .01)


def initializer(p, direction=None, epsilon=.01):
    m = torch.eye(2*p)
    if direction is not None:
        m += epsilon*(2*p)**.5*torch.from_numpy(direction)
    if float(torch.linalg.eigvalsh(m).min()) <= 0:
        raise ValueError('Initializer not positive definite')
    return m.cuda()


def step_update(x, m, alpha, k):
    g = rfm.gradients(x, m, alpha, k, 2.5, centering=True)
    agop = torch.zeros_like(m)
    for batch in torch.split(g, 2):
        agop += torch.sum(batch.transpose(1, 2) @ batch, dim=0)
    agop /= len(g)
    ev, q = torch.linalg.eigh(agop)
    ev = torch.where(ev > 0., ev, 0.)
    return q @ torch.diag(ev.sqrt()) @ q.T


@dataclass
class State:
    m: torch.Tensor
    alpha: torch.Tensor | None = None
    k: torch.Tensor | None = None
    index: int = -1


def advance(state, end, data):
    """Exactly `end-(index+1)` new fits; resume includes the deferred update."""
    x, y, xv, yv = data
    targets = yv.cpu().numpy()
    history = []
    start = state.index
    for t in range(start+1, end):
        if t > 0:
            state.m = step_update(x, state.m, state.alpha, state.k)
        state.k = rfm.kernel(x, x, state.m, 2.5)
        state.alpha = torch.linalg.solve(state.k, y)
        pv = rfm.kernel(x, xv, state.m, 2.5).T @ state.alpha
        if not bool(torch.isfinite(state.m).all()) or not bool(torch.isfinite(pv).all()):
            raise FloatingPointError('Non-finite trajectory')
        history.append(pv.cpu().numpy())
        state.index = t
    predictions = np.stack(history)
    row = metrics(predictions[-1], targets)
    row['relative_residual'] = float(torch.linalg.norm(state.k@state.alpha-y)/torch.linalg.norm(y))
    return predictions, row


def save_state(path, state, **arrays):
    np.savez_compressed(path, m=state.m.cpu().numpy(), alpha=state.alpha.cpu().numpy(),
                        k=state.k.cpu().numpy(), index=state.index, **arrays)


def load_state(path):
    a = np.load(path)
    return State(torch.from_numpy(a['m']).cuda(), torch.from_numpy(a['alpha']).cuda(),
                 torch.from_numpy(a['k']).cuda(), int(a['index']))


def budget(p, n=20):
    d = (p-1)//2
    b = 1+d*(d+1)//2
    cap = 60+60*b/n
    return dict(bank_trajectories=b, fit_cap=cap, probes=int((cap-49)//11), full=int(cap//60))


def profile(pred, val_indices, p):
    v = np.array([np.roll(row, -2*a) for row, a in zip(pred, val_indices)]).mean(0)
    return v-v.mean()


def bank_code(candidates, q):
    aa = np.stack([candidates.coords(code) for code in range(candidates.size)])
    v = np.einsum('ki,cij,kj->kc', aa, q, aa)
    z = (v[:, 0]-v[:, 1:].max(1)) / np.maximum(np.sqrt(((v-v.mean(1, keepdims=True))**2).mean(1)), 1e-300)
    return int(z.argmax()), z


def select(method, p, seed, data, q=None, out=None, n=20):
    """Write candidate validation logs and seal the choice BEFORE continuation."""
    if method not in ('bank', 'random_probe', 'greedy_probe', 'random_full'):
        raise ValueError(method)
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    candidates = Candidates(p, seed)
    limits = budget(p, n)
    records, states, best, current = [], {}, None, 0
    fits, updates = 0, 0
    if method == 'bank':
        code, zz = bank_code(candidates, q)
        records.append(dict(code=code, predicted_normalized_profile_margin=float(zz[code]),
                            candidates_scored=len(zz), fits=0))
        chosen = code
        selection = dict(code=chosen, criterion='frozen tensor normalized profile margin', records=records)
    else:
        limit = limits['full'] if method == 'random_full' else limits['probes']
        length = 60 if method == 'random_full' else 11
        bit_cursor, visited_skips = 0, 0
        while len(records) < limit:
            if len(records) == 0:
                code = 0
            elif method in ('random_full', 'random_probe'):
                code = candidates.order[len(records)]
            else:
                code = current ^ (1 << candidates.bits[bit_cursor % len(candidates.bits)])
                bit_cursor += 1
                if code in states:
                    visited_skips += 1
                    if visited_skips < len(candidates.bits):
                        continue
                    code = next(c for c in candidates.order if c not in states)
                    visited_skips = 0
                    current = code
                else:
                    visited_skips = 0
            state = State(candidates.m0(code))
            hist, val = advance(state, length, data)
            fits += length
            updates += length-1
            rec = dict(code=code, evaluation=len(records), fits=length, validation=val)
            records.append(rec)
            states[code] = (state, hist, val)
            if best is None or key(val) > key(states[best][2]):
                best = code
            if method == 'greedy_probe' and (code == current or key(val) > key(states[current][2])):
                current = code
            if out is not None:
                with (out/'candidates.jsonl').open('a') as f:
                    f.write(json.dumps(rec, allow_nan=False)+'\n')
            if method == 'random_full' and code == 0:
                if out is not None:
                    save_state(out/'unmodified.npz', state, validation_predictions=hist[-1], m0=candidates.m0(0).cpu().numpy())
        chosen = best
        selection = dict(code=chosen, criterion='validation accuracy then normalized pointwise margin', records=records)
    torch.cuda.synchronize()
    selection_s = time.perf_counter()-started
    choice = dict(p=p, seed=seed, method=method, selection=selection,
                  fits_at_selection=fits, fit_cap=limits['fit_cap'], selection_seconds=selection_s)
    if out is not None:
        write_json(out/'choice.json', choice)
        choice_hash = sha(out/'choice.json')
    else:
        choice_hash = hashlib.sha256(json.dumps(choice, sort_keys=True).encode()).hexdigest()
    if method == 'bank':
        state = State(candidates.m0(chosen))
        hist, val = advance(state, 60, data)
        fits, updates = 60, 59
    elif method != 'random_full':
        state, early, _ = states[chosen]
        if out is not None:
            save_state(out/'selected_probe.npz', state, m0=candidates.m0(chosen).cpu().numpy())
        hist_late, val = advance(state, 60, data)
        hist = np.concatenate([early, hist_late])
        fits += 49
        updates += 49
    else:
        state, hist, val = states[chosen]
    if out is not None:
        save_state(out/'selected.npz', state, validation_history=hist, m0=candidates.m0(chosen).cpu().numpy())
    torch.cuda.synchronize()
    total_s = time.perf_counter()-started
    info = dict(p=p, seed=seed, method=method, chosen_code=chosen,
                fit_count=fits, update_count=updates, unique_candidates=len(records),
                selection_seconds=selection_s, online_seconds=total_s,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                final_validation=val, choice_sha256=choice_hash,
                final_index=state.index, status='complete')
    if out is not None:
        write_json(out/'result.json', info)
    return info, state, records
