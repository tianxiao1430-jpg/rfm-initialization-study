"""Orthogonal circulant decomposition and sign-invariant initial features."""
import numpy as np

def circulant_projection(b):
    p=len(b);idx=np.arange(p)
    c=np.array([b[idx,(idx+k)%p].mean() for k in range(p)])
    return c[(idx[None,:]-idx[:,None])%p]

def features(b):
    p=len(b);c=circulant_projection(b);energy=float(np.sum(b*b))
    spectrum=np.linalg.svd(b,compute_uv=False)**2
    fourier=np.abs(np.fft.fft(c[0]))**2
    fourier=fourier[1:(p+1)//2]
    mass=fourier.sum()
    return dict(circulant_fraction=float(np.sum(c*c)/energy),
                frequency_concentration=float(np.max(fourier)/mass) if mass>1e-30 else 0.,
                spectral_fraction=float(spectrum[0]/spectrum.sum()),
                effective_rank=float(spectrum.sum()**2/np.sum(spectrum*spectrum)),
                centered_energy_fraction=float(np.sum(((np.eye(p)-np.ones((p,p))/p)@b@(np.eye(p)-np.ones((p,p))/p))**2)/energy))

def cross_matrix(b):
    p=len(b);e=np.zeros((2*p,2*p));e[:p,p:]=b;e[p:,:p]=b.T
    return e

def split_odd(e):
    p=len(e)//2;d=e.copy();d[:p,p:]=0;d[p:,:p]=0
    c=e-d
    return d,c
