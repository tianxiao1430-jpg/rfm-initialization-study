"""Load immutable initial matrices listed in the pre-outcome freeze manifest."""
from pathlib import Path
import numpy as np
import torch
from directions_v2 import initial_direction as previous_direction
ROOT=Path(__file__).resolve().parents[1]
def initial_direction(p,seed,kind,amount=None):
    if kind!='custom':return previous_direction(p,seed,kind,amount)
    if Path(amount).name!=amount:raise ValueError('initializer must be a filename')
    e=np.load(ROOT/'initializations'/amount)['direction']
    assert e.shape==(2*p,2*p) and np.linalg.norm(e-e.T)<1e-12 and abs(np.linalg.norm(e)-1)<1e-12
    return torch.from_numpy(e)
