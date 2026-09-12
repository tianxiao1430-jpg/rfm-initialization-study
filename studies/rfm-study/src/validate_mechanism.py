"""Independent local derivative checks; no confirmation outcome selection."""
import json
from pathlib import Path
import numpy as np
import torch
from probe_modes import first_derivative
from experiment import initial_direction
from components import circulant_projection
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in [17,23,29]:
    for seed in [20,21]:
        for kind in ['circ','residual','diag']:
            e=initial_direction(p,seed,kind);dm,meta=first_derivative(p,e)
            b=dm[:p,p:].numpy();b=(b-b.T)/2;c=circulant_projection(b)
            rows.append(dict(p=p,seed=seed,kind=kind,derivative_norm=float(torch.linalg.norm(dm)),
                             circulant_cross_output_norm=float(np.sqrt(2)*np.linalg.norm(c)),**meta))
(ROOT/'results/linear_checks.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
