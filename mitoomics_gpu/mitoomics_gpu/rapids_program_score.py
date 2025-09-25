#!/usr/bin/env python3
"""
rapids_program_score.py
GPU-accelerated program scoring with CuPy/cupyx; falls back to NumPy/SciPy.

API:
    score_programs(adata, gene_sets, layer=None, libnorm="CPM", log1p=True,
                   standardize="z", return_df=True, use_gpu=True, batch_cols=None)

- adata: AnnData with .X or layer CSR/dense
- gene_sets: dict[str, list[str]] mapping program name -> genes (var_names)
- libnorm: "CPM" or None
- standardize: "z" (z-score per-program across cells), "none"
- batch_cols: optional list of obs columns to compute per-batch z-scores
"""
from __future__ import annotations
import warnings, numpy as np, pandas as pd, scipy.sparse as sp
from typing import Dict, List, Optional
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    _HAVE_GPU = True
except Exception:
    cp = None; csp = None; _HAVE_GPU = False

def _as_csr_cpu(mat):
    if sp.issparse(mat):
        return mat.tocsr()
    return sp.csr_matrix(np.asarray(mat))

def _as_csr_gpu(mat):
    if csp is None: raise RuntimeError("cupyx not available")
    if sp.issparse(mat):
        mat = mat.tocsr()
        return csp.csr_matrix((cp.asarray(mat.data), cp.asarray(mat.indices), cp.asarray(mat.indptr)), shape=mat.shape)
    if isinstance(mat, np.ndarray):
        return csp.csr_matrix(cp.asarray(mat))
    raise TypeError("Unsupported matrix type for GPU path")

def _libsize_and_norm_cpu(X):
    lib = np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
    lib[lib==0] = 1.0
    return lib

def _libsize_and_norm_gpu(X):
    lib = cp.asarray(X.sum(axis=1)).ravel() if csp.isspmatrix(X) else X.sum(axis=1)
    lib = lib.get() if isinstance(lib, cp.ndarray) else lib
    lib = lib.astype(float)
    lib[lib==0] = 1.0
    return lib

def _log1p_cpu(M):
    if sp.issparse(M):
        Mc = M.copy()
        Mc.data = np.log1p(Mc.data)
        return Mc
    return np.log1p(M)

def _log1p_gpu(M):
    if csp.isspmatrix(M):
        Mc = M.copy()
        Mc.data = cp.log1p(Mc.data)
        return Mc
    return cp.log1p(M)

def _mean_over_cols_cpu(X, cols: np.ndarray):
    # X: CSR, cols: 1D indices
    if len(cols)==0:
        return np.zeros(X.shape[0], dtype=float)
    sub = X[:, cols]
    if sp.issparse(sub):
        return (sub.sum(axis=1) / max(len(cols),1)).A1
    return sub.mean(axis=1)

def _mean_over_cols_gpu(X, cols):
    if len(cols)==0:
        return cp.zeros(X.shape[0], dtype=cp.float32)
    sub = X[:, cols]
    if csp.isspmatrix(sub):
        s = sub.sum(axis=1)
        if hasattr(s, "A"):
            s = s.A  # cupyx returns sparse matrix-like
        s = cp.asarray(s).ravel()
        return s / float(max(len(cols),1))
    return cp.mean(sub, axis=1)

def _z_score_series_cpu(v: np.ndarray) -> np.ndarray:
    mu = np.nanmean(v); sd = np.nanstd(v)
    return np.zeros_like(v) if (not np.isfinite(sd) or sd==0) else (v - mu) / (sd + 1e-12)

def _z_score_series_gpu(v) :
    mu = cp.nanmean(v); sd = cp.nanstd(v)
    out = cp.zeros_like(v)
    mask = sd!=0
    if mask:
        out = (v - mu) / (sd + 1e-12)
    return out

def _ensure_layer(adata, layer):
    if layer is None:
        return adata.X
    return adata.layers[layer]

def _varname_to_index(adata) -> Dict[str,int]:
    return {str(g): i for i,g in enumerate(map(str, adata.var_names.values))}

def score_programs(
    adata,
    gene_sets: Dict[str, List[str]],
    layer: Optional[str] = None,
    libnorm: Optional[str] = "CPM",
    log1p: bool = True,
    standardize: str = "z",
    return_df: bool = True,
    use_gpu: bool = True,
    batch_cols: Optional[List[str]] = None,
):
    """
    Returns:
        DataFrame (n_cells x n_sets) if return_df,
        else dict[str -> 1D array-like].
    """
    # Select engine
    gpu = bool(use_gpu and _HAVE_GPU)
    X = _ensure_layer(adata, layer)
    X = _as_csr_gpu(X) if gpu else _as_csr_cpu(X)
    var_ix = _varname_to_index(adata)
    sets_ix = {k: np.array([var_ix[g] for g in v if g in var_ix], dtype=int) for k,v in gene_sets.items()}

    # Library size normalization
    if libnorm and libnorm.upper()=="CPM":
        lib = _libsize_and_norm_gpu(X) if gpu else _libsize_and_norm_cpu(X)
        scale = (1e6 / lib).astype(np.float32)
        if gpu:
            scale = cp.asarray(scale)
            if csp.isspmatrix(X):
                X = csp.diags(scale) @ X
            else:
                X = (scale[:,None] * X)
        else:
            if sp.issparse(X):
                X = sp.diags(scale) @ X
            else:
                X = scale[:,None] * X

    # log1p
    if log1p:
        X = _log1p_gpu(X) if gpu else _log1p_cpu(X)

    # Scores
    scores = {}
    for name, cols in sets_ix.items():
        if gpu:
            s = _mean_over_cols_gpu(X, cols)
            if standardize.lower()=="z":
                s = (s - cp.mean(s)) / (cp.std(s)+1e-12) if s.size>1 else s
            scores[name] = s.get()
        else:
            s = _mean_over_cols_cpu(X, cols)
            if standardize.lower()=="z":
                s = _z_score_series_cpu(s)
            scores[name] = s

    if batch_cols:
        # Per-batch z-scoring for each program
        df = pd.DataFrame(scores, index=adata.obs_names)
        for bc in batch_cols:
            if bc not in adata.obs.columns: continue
            g = adata.obs[bc].astype(str)
            for col in df.columns:
                df[col] = df.groupby(g)[col].transform(lambda x: (x - x.mean())/(x.std()+1e-12))
        return df if return_df else {k: df[k].values for k in df.columns}

    return pd.DataFrame(scores, index=adata.obs_names) if return_df else scores
