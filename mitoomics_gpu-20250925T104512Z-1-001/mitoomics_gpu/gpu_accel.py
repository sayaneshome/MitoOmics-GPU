#!/usr/bin/env python3
"""
gpu_accel.py
GPU-accelerated building blocks for MHI:
- PCA/UMAP/Neighbors via cuML (fallback to sklearn)
- Gene set scoring via CuPy (fallback to numpy)
- EV proteomics aggregation & correlations via cuDF/CuPy (fallback to pandas/numpy)
"""

from __future__ import annotations
import anndata as ad
import numpy as np
import pandas as pd
from .gpu_backend import GB, to_xp, to_cpu, df_to_gpu, df_to_cpu

# --------------------- PCA ---------------------

def pca(X, n_components=50, random_state=0):
    """
    X: np.ndarray (cells x genes) or cupy array.
    Returns: (X_pca: np.ndarray, components_: np.ndarray)
    """
    if GB.has_gpu and GB.cuml is not None:
        pca_gpu = GB.cuml.PCA(n_components=n_components, random_state=random_state, whiten=False)
        Xp = pca_gpu.fit_transform(to_xp(X))
        comps = pca_gpu.components_
        return to_cpu(Xp), to_cpu(comps)
    else:
        from sklearn.decomposition import PCA as SKPCA
        pca_cpu = SKPCA(n_components=n_components, random_state=random_state, whiten=False)
        Xp = pca_cpu.fit_transform(np.asarray(X))
        return Xp, pca_cpu.components_

# --------------------- UMAP ---------------------

def umap(X, n_components=2, n_neighbors=15, min_dist=0.3, random_state=0):
    """
    Returns UMAP embedding (cells x 2)
    """
    if GB.has_gpu and GB.cuml is not None:
        um = GB.cuml.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                          min_dist=min_dist, random_state=random_state, verbose=False)
        E = um.fit_transform(to_xp(X))
        return to_cpu(E)
    else:
        import umap
        umc = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state, verbose=False)
        return umc.fit_transform(np.asarray(X))

# --------------------- Neighbors graph ---------------------

def neighbors(X, n_neighbors=15, metric="euclidean"):
    """
    Return indices of neighbors for each point (cells x k)
    """
    if GB.has_gpu and GB.cuml is not None:
        nn = GB.cuml.NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        nn.fit(to_xp(X))
        D, I = nn.kneighbors(to_xp(X))
        return to_cpu(I)
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
        nn.fit(np.asarray(X))
        D, I = nn.kneighbors(np.asarray(X))
        return I

# --------------------- Gene-set scoring ---------------------

def score_gene_sets(adata: ad.AnnData, genesets_df: pd.DataFrame, layer=None, method="zmean"):
    """
    genesets_df columns: ['pathway','gene'] uppercase HGNC.
    Produces per-cell scores for each pathway (returns pandas DataFrame: cells x pathways).
    method='zmean' -> z-score each gene across cells, then mean within set.
    """
    # Build gene x pathway mask
    var_names = pd.Series(adata.var_names.astype(str)).str.upper()
    path_to_genes = genesets_df.groupby("pathway")["gene"].apply(lambda s: set(s.astype(str).str.upper()))
    pathways = sorted(path_to_genes.index.tolist())

    # Matrix: cells x genes
    X = adata.layers[layer] if layer else adata.X
    if hasattr(X, "toarray"):  # sparse
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    Xxp = to_xp(X)

    # z-score per gene
    mu = Xxp.mean(axis=0, keepdims=True)
    sd = Xxp.std(axis=0, ddof=0, keepdims=True) + 1e-8
    Z = (Xxp - mu) / sd

    scores = {}
    for p in pathways:
        genes = path_to_genes[p]
        cols = var_names[var_names.isin(genes)].index.values
        if len(cols) == 0:
            scores[p] = np.zeros((adata.n_obs,), dtype=float)
            continue
        sub = Z[:, to_xp(cols)]
        s = sub.mean(axis=1)
        scores[p] = to_cpu(s).ravel()

    # Return DataFrame aligned to adata.obs_names
    out = pd.DataFrame(scores, index=adata.obs_names)
    return out

# --------------------- EV proteomics utilities ---------------------

def ev_index(ev_df: pd.DataFrame, whitelist: set[str], subject_col="subject_id",
             protein_col="protein", value_col="abundance"):
    """
    Build an 'EV_index' per subject: z-score across proteins (whitelist) then average.
    """
    df = ev_df[[subject_col, protein_col, value_col]].dropna()
    df[protein_col] = df[protein_col].astype(str).str.upper()
    df = df[df[protein_col].isin({p.upper() for p in whitelist})].copy()
    if df.empty:
        return pd.Series(dtype=float)

    # GPU dataframe when possible
    gdf = df_to_gpu(df)
    # pivot: subjects x proteins
    if GB.has_gpu:
        pv = gdf.pivot(index=subject_col, columns=protein_col, values=value_col).fillna(0)
        pv_cpu = pv.to_pandas()
    else:
        pv_cpu = df.pivot_table(index=subject_col, columns=protein_col, values=value_col, aggfunc="mean").fillna(0)

    # z across proteins; mean row-wise
    Z = (pv_cpu - pv_cpu.mean()) / (pv_cpu.std(ddof=0) + 1e-8)
    ev_idx = Z.mean(axis=1).rename("EV_index")
    return ev_idx

def corr_matrix(df: pd.DataFrame):
    """
    Correlation matrix using CuPy if available (fast), else pandas.
    """
    if df.empty or df.shape[1] < 2:
        return pd.DataFrame()
    if GB.has_gpu:
        X = to_xp(df.values.astype(float))
        X = X - X.mean(axis=0, keepdims=True)
        cov = (X.T @ X) / (X.shape[0] - 1)
        sd = GB.xp.sqrt(GB.xp.diag(cov))
        denom = GB.xp.outer(sd, sd) + 1e-12
        corr = cov / denom
        return pd.DataFrame(to_cpu(corr), index=df.columns, columns=df.columns)
    else:
        return df.corr(numeric_only=True)
