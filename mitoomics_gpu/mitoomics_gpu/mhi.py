from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from .config import DEFAULT_WEIGHTS

def _scale01(s: pd.Series):
    v = s.values.astype(float)
    lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
    v = (np.clip(v, lo, hi) - lo) / (hi - lo + 1e-9)
    return pd.Series(v, index=s.index)

def combine_components(subject_df: pd.DataFrame, weights=DEFAULT_WEIGHTS, fit_ridge: bool=False, targets: pd.Series|None=None):
    df = subject_df.copy().set_index("subject_id")
    # Pick available components
    cols = [c for c in ["copy_number","fusion_fission","mitophagy","heterogeneity"] if c in df.columns]
    if len(cols) == 0:
        raise ValueError("No components to combine (need at least one of copy_number, fusion_fission, mitophagy, heterogeneity)")

    X = df[cols].apply(_scale01)

    if fit_ridge and targets is not None:
        # RidgeCV to learn weights mapping components -> target
        y = targets.reindex(X.index)
        if y.isna().any():
            y = y.fillna(method="ffill").fillna(method="bfill").fillna(y.mean())
        model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=min(5, len(X)))
        model.fit(X.values, y.values)
        w = pd.Series(model.coef_, index=cols)
    else:
        # Use config weights, renormalized for present columns
        cfg = {
            "copy_number": weights.copy_number,
            "fusion_fission": weights.fusion_fission,
            "mitophagy": weights.mitophagy,
            "heterogeneity": weights.heterogeneity,
        }
        w = pd.Series({c: cfg[c] for c in cols}, index=cols)
        w = w / (w.sum() + 1e-9)

    mhi = (X * w).sum(axis=1).rename("MHI")
    out = df.copy()
    out["MHI"] = mhi
    out = out.reset_index()
    # Also emit learned weights if Ridge was used
    if fit_ridge and targets is not None:
        for c in w.index:
            out[f"weight_{c}"] = w.loc[c]
    return out
