"""RFM arithmetic experiment, adapted from the GPL-3.0 upstream implementation.

Upstream: marceltomas/breaking-data-symmetries @ 311273bfc08adcde344c8b021fc5aa8d9970ad98.
The original kernel and batch-of-two AGOP accumulation order are preserved.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

# Use the established eager CUDA kernels; the optional native Triton route
# requires a C compiler absent on this host. Record this environment choice.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
import numpy as np
import torch
import torch.nn.functional as F


def dataset(p, split="symmetric", seed=0, dtype=torch.float64, device="cpu"):
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    train = pairs[:, 0] != pairs[:, 1]
    gen = torch.Generator().manual_seed(seed)
    off = torch.where(train)[0]
    removed = []
    if split in ("remove_one", "remove_pair"):
        idx = off[torch.randperm(len(off), generator=gen)[0]]
        removed = [int(idx)]
        train[idx] = False
        if split == "remove_pair":
            a, b = pairs[idx].tolist()
            idx2 = b * p + a
            removed.append(idx2)
            train[idx2] = False
    elif split == "random_half":
        train[:] = False
        train[torch.randperm(p*p, generator=gen)[:p*p//2]] = True
    elif split != "symmetric":
        raise ValueError(split)
    x = F.one_hot(pairs, num_classes=p).reshape(p*p, 2*p).to(dtype=dtype, device=device)
    y = F.one_hot((pairs[:,0]+pairs[:,1]) % p, num_classes=p).to(dtype=dtype, device=device)
    fixed = pairs[:,0] == pairs[:,1]
    return x[train], y[train], x[~train], y[~train], x[fixed], y[fixed], {
        "pairs": pairs.numpy(), "train_mask": train.numpy(), "removed": removed,
        "train_hash": hashlib.sha256(pairs[train].numpy().tobytes()).hexdigest(),
    }


def kernel(x, z, m, bandwidth=2.5):
    xn = ((x @ m) * x).sum(1, keepdim=True)
    zn = xn if x is z else ((z @ m) * z).sum(1, keepdim=True)
    dist = x.mm(m @ z.T)
    dist.mul_(-2)
    dist.add_(xn)
    dist.add_(zn.reshape(1, -1))
    dist.clamp_(min=0)
    dist.mul_(-1.0 / (2 * bandwidth**2))
    return dist.exp_()


def gradients(x, m, alpha, k, bandwidth=2.5, centering=True):
    n, d = x.shape
    c = alpha.shape[1]
    step1 = (alpha.reshape(n, c, 1) @ (x @ m).reshape(n, 1, d)).reshape(n, c*d)
    step2 = (k.T @ step1).reshape(n, c, d)
    step3 = ((alpha.T @ k).T.reshape(n, c, 1) @ (x @ m).reshape(n, 1, d))
    # Upstream uses the negative of the input gradient; sign cancels in G.T G.
    g = (step2 - step3) * -1 / (bandwidth**2)
    return g - g.mean(0) if centering else g


def update(x, m, alpha, k, bandwidth=2.5):
    g = gradients(x, m, alpha, k, bandwidth)
    agop = torch.zeros_like(m)
    for batch in torch.split(g, 2):
        agop += torch.sum(batch.transpose(1, 2) @ batch, dim=0)
    agop /= len(g)
    ev, q = torch.linalg.eigh(agop)
    minimum = float(ev.min().item())
    negative_fraction = float((-ev.clamp(max=0).sum() / ev.abs().sum().clamp_min(torch.finfo(ev.dtype).tiny)).item())
    new_m = q @ torch.diag(ev.clamp(min=0).sqrt()) @ q.T
    return new_m, minimum, negative_fraction


def swapped(m):
    p = m.shape[0] // 2
    ix = torch.cat((torch.arange(p, 2*p, device=m.device), torch.arange(p, device=m.device)))
    return m[ix][:, ix]


def project(m):
    return (m + swapped(m)) / 2


def perturbation(p, seed, kind):
    gen = torch.Generator().manual_seed(seed)
    e = torch.randn(2*p, 2*p, generator=gen, dtype=torch.float64)
    e = (e + e.T) / 2
    if kind == "preserve":
        e = (e + swapped(e)) / 2
    elif kind == "break":
        e = (e - swapped(e)) / 2
    elif kind != "generic":
        raise ValueError(kind)
    return e / torch.linalg.norm(e)


def score(pred, y):
    truth = y.argmax(1)
    correct = pred.gather(1, truth[:, None]).squeeze(1)
    others = pred.clone()
    others.scatter_(1, truth[:, None], -torch.inf)
    margin = correct - others.max(1).values
    return {"acc": float((pred.argmax(1) == truth).double().mean().item()),
            "mse": float((pred-y).square().mean().item()),
            "mean_margin": float(margin.mean().item()),
            "min_margin": float(margin.min().item())}


def environment():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(), "threads": torch.get_num_threads(),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "TORCH_DISABLE_NATIVE_JIT":os.environ.get("TORCH_DISABLE_NATIVE_JIT"),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def run(config, out):
    cfg = {"p":29, "split":"symmetric", "seed":0, "dtype":"float64", "device":"cuda",
           "iterations":60, "ridge":0.0, "bandwidth":2.5, "perturb_kind":"none",
           "epsilon":0.0, "project":False, "threads":1, "timeout":600, **config}
    torch.set_num_threads(cfg["threads"])
    torch.set_default_dtype(getattr(torch, cfg["dtype"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg["seed"])
    dt = getattr(torch, cfg["dtype"])
    path = Path(out)
    path.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (path/"config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (path/"environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")
    rows = []
    status = "completed"
    error = None
    try:
        x, y, xt, yt, xf, yf, data = dataset(cfg["p"], cfg["split"], cfg["seed"], dt, cfg["device"])
        m0 = torch.eye(2*cfg["p"], dtype=torch.float64)
        if cfg["perturb_kind"] != "none":
            m0 += cfg["epsilon"] * torch.linalg.norm(m0) * perturbation(cfg["p"], cfg["seed"], cfg["perturb_kind"])
        if torch.linalg.eigvalsh(m0).min() <= 0:
            raise ValueError("Initial perturbation is not positive definite")
        m = m0.to(dtype=dt, device=cfg["device"])
        actual_eps = float((torch.linalg.norm(m-torch.eye(len(m), dtype=dt, device=m.device)) / len(m)**0.5).item())
        np.savez_compressed(path/"data.npz", **data)
        metadata = {"n_train":len(x), "n_test":len(xt), "n_fixed":len(xf),
                    "actual_initial_epsilon":actual_eps, "train_hash":data["train_hash"],
                    "removed":data["removed"]}
        (path/"data_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        previous_min_eig = 0.0
        previous_negative_fraction = 0.0
        for t in range(cfg["iterations"]):
            if time.monotonic() - started > cfg["timeout"]:
                raise TimeoutError("Per-run hard time limit")
            k = kernel(x, x, m, cfg["bandwidth"])
            kr = k + cfg["ridge"] * torch.eye(len(x), dtype=dt, device=x.device)
            alpha = torch.linalg.solve(kr, y)
            pred = kernel(x, xt, m, cfg["bandwidth"]).T @ alpha
            fixed_pred = kernel(x, xf, m, cfg["bandwidth"]).T @ alpha
            train_pred = k.T @ alpha
            if not all(bool(torch.isfinite(z).all()) for z in (m, k, alpha, pred)):
                raise FloatingPointError("Non-finite intermediate")
            row = {"iteration":t, "elapsed_s":time.monotonic()-started,
                   "relative_residual":float((torch.linalg.norm(kr@alpha-y)/torch.linalg.norm(y)).item()),
                   "symmetry_defect":float((torch.linalg.norm(m-swapped(m))/torch.linalg.norm(m)).item()),
                   "m_norm":float(torch.linalg.norm(m).item()),
                   "previous_agop_min_eigenvalue":previous_min_eig,
                   "previous_agop_negative_mass_fraction":previous_negative_fraction}
            for label, pr, yy in (("train",train_pred,y),("test",pred,yt),("fixed",fixed_pred,yf)):
                row.update({f"{label}_{key}":v for key,v in score(pr,yy).items()})
            rows.append(row)
            with (path/"metrics.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False)+"\n")
            if t % 20 == 0 or t == cfg["iterations"]-1:
                print(json.dumps({"run":path.name, **{k:row[k] for k in ("iteration","fixed_acc","relative_residual","symmetry_defect","elapsed_s")}}), flush=True)
            if t != cfg["iterations"]-1:
                m, previous_min_eig, previous_negative_fraction = update(x,m,alpha,k,cfg["bandwidth"])
                if cfg["project"]:
                    m = project(m)
        np.savez_compressed(path/"final.npz", m=m.cpu().numpy(), pred=pred.cpu().numpy(),
                            fixed_pred=fixed_pred.cpu().numpy(), fixed_y=yf.cpu().numpy())
    except Exception:
        status = "failed"
        error = traceback.format_exc()
        (path/"error.txt").write_text(error, encoding="utf-8")
        print(error, flush=True)
    summary = {"status":status, "config":cfg, "elapsed_s":time.monotonic()-started,
               "n_rows":len(rows), "final":rows[-1] if rows else None, "error":error}
    if rows:
        summary["max_relative_residual"] = max(r["relative_residual"] for r in rows)
        summary["max_fixed_acc"] = max(r["fixed_acc"] for r in rows)
        summary["last10_fixed_acc_mean"] = sum(r["fixed_acc"] for r in rows[-10:])/len(rows[-10:])
    (path/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text(encoding="utf-8"))
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    failures=0
    for job in plan:
        name=job.pop("name")
        if (root/name/"summary.json").exists():
            print("Already recorded, skipping:",name,flush=True)
            continue
        s=run(job,root/name)
        failures += s["status"] != "completed"
    print(json.dumps({"suite_finished":True,"failed_runs":failures}),flush=True)


if __name__ == "__main__":
    main()
