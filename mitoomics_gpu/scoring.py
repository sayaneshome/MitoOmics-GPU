from __future__ import annotations
import numpy as np
import pandas as pd

def _to_dense(X):
    if hasattr(X, "toarray"):
        return X.toarray()
    return X

def copy_number_proxy(adata) -> pd.Series:
    genes = adata.var.index.astype(str)
    is_mt = genes.str.upper().str.startswith("MT-")
    if is_mt.sum() == 0:
        is_mt = genes.str.upper().str.contains("MT-")
    X = _to_dense(adata.X)
    total = X.sum(axis=1) + 1e-9
    mt_sum = X[:, is_mt].sum(axis=1) if is_mt.any() else np.zeros(X.shape[0], dtype=float)
    frac = (mt_sum / total)
    batch = adata.obs.get("batch", pd.Series(["batch1"]*adata.n_obs, index=adata.obs_names))
    frac = pd.Series(frac, index=adata.obs_names, name="copy_number_proxy_raw")
    z = frac.groupby(batch).transform(lambda v: (v - v.mean())/(v.std(ddof=1)+1e-9))
    return z.rename("copy_number_proxy")

def program_scores(adata, pathways: dict[str, list[str]]|None=None) -> pd.DataFrame:
    """
    Compute per-cell z-scored program scores for each pathway in `pathways`.
    """
    if pathways is None or len(pathways) == 0:
        return pd.DataFrame(index=adata.obs_names)

    genes = adata.var.index.astype(str)
    X = _to_dense(adata.X)
    g_upper = genes.str.upper().values

    out = {}
    for pname, glist in pathways.items():
        if not glist:
            continue
        pset = set(g.upper() for g in glist)
        mask = np.array([g in pset for g in g_upper], dtype=bool)
        if mask.sum() == 0:
            out[pname] = np.zeros(adata.n_obs, dtype=float)
        else:
            vals = X[:, mask].mean(axis=1)
            s = pd.Series(vals, index=adata.obs_names)
            batch = adata.obs.get("batch", pd.Series(["batch1"]*adata.n_obs, index=adata.obs_names))
            z = s.groupby(batch).transform(lambda v: (v - v.mean())/(v.std(ddof=1)+1e-9))
            out[pname] = z.values
    df = pd.DataFrame(out, index=adata.obs_names)
    # Make nice column names (lowercase to avoid mismatches)
    df.columns = [str(c).lower() for c in df.columns]
    return df

def heterogeneity_scores(adata) -> pd.Series:
    """
    Heterogeneity proxy per subject:
      - Shannon diversity over cell_type distribution.
      - Scaled to [0,1] across subjects.
    """
    ct = adata.obs.get("cell_type")
    if ct is None:
        s = adata.obs.groupby("subject_id")["copy_number_proxy"].std().fillna(0.0)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    def shannon(p):
        p = np.asarray(p, dtype=float)
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    rows = []
    for sid, g in adata.obs.groupby("subject_id"):
        props = g["cell_type"].value_counts(normalize=True).values
        rows.append((sid, shannon(props)))
    df = pd.DataFrame(rows, columns=["subject_id","diversity"])
    v = df["diversity"].values
    v = (v - v.min()) / (v.max() - v.min() + 1e-9)
    return pd.Series(v, index=df["subject_id"].values)
