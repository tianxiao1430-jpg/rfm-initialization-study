"""Versioned extension: controlled Fourier directions, no changes to v1 runs."""
import numpy as np
import torch
from components import cross_matrix
from experiment import initial_direction as original_direction

def mode(p,k):
    idx=np.arange(p)
    b=np.sin(2*np.pi*k*(idx[None,:]-idx[:,None])/p)
    e=cross_matrix(b);return e/np.linalg.norm(e)

def initial_direction(p,seed,kind,amount=None):
    if kind=='mode':e=mode(p,seed)
    elif kind=='mode_pair':e=(mode(p,seed)+mode(p,int(amount)))/np.sqrt(2)
    elif kind=='mode_difference':e=(mode(p,seed)-mode(p,int(amount)))/np.sqrt(2)
    else:return original_direction(p,seed,kind,amount)
    return torch.from_numpy(e)
