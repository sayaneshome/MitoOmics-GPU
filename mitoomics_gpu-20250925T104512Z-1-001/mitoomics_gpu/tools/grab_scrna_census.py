#!/usr/bin/env python3
# tools/grab_scrna_census.py
from __future__ import annotations
import argparse, numpy as np, pandas as pd
import anndata as ad
import cellxgene_census

CENSUS_VERSION = "2025-01-30"  # pin for reproducibility

def _in_filter(col: str, values: list[str]) -> str:
    vals = ", ".join(repr(v.strip()) for v in sorted({v for v in values if v}))
    return f"{col} in [{vals}]"

def _pick_subject(obs: pd.DataFrame) -> str:
    for k in ("donor_id", "sample_id", "dataset_id"):
        if k in obs.columns:
            return k
    obs["synthetic_subject"] = "subject0"
    return "synthetic_subject"

def _tumor_mask(obs: pd.DataFrame) -> pd.Series:
    bad = {"normal", "healthy", "control", "none", "not reported", "na", "nan", ""}
    cols = [c for c in ("disease", "disease_label", "disease__ontology_label") if c in obs.columns]
    if not cols:
        return pd.Series(True, index=obs.index)
    keep = pd.Series(False, index=obs.index)
    for c in cols:
        s = obs[c].astype(str).str.strip().str.lower()
        keep |= ~s.isin(bad)
    return keep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=120_000)
    ap.add_argument("--tissues", nargs="+", default=["lung","breast","colon","pancreas"])
    ap.add_argument("--tumor-only", action="store_true")
    ap.add_argument("--out", default="data/scrna.h5ad")
    args = ap.parse_args()

    obs_filter = _in_filter("tissue_general", args.tissues)

    # Open pinned census and call get_anndata with minimal, backward-compatible args
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        try:
            # Old API: pass the handle as first arg; minimal arg set
            A = cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                measurement_name="RNA",
                X_name="raw",
                obs_value_filter=obs_filter,
            )
        except TypeError:
            # Newer API (no handle)
            A = cellxgene_census.get_anndata(
                organism="Homo sapiens",
                measurement_name="RNA",
                X_name="raw",
                obs_value_filter=obs_filter,
            )

    # Optional tumor-only filter
    if args.tumor_only:
        mask = _tumor_mask(A.obs)
        A = A[mask].copy()
        if A.n_obs == 0:
            raise SystemExit("No tumor cells matched filters; rerun without --tumor-only or change tissues.")

    # Downsample for speed/memory
    if A.n_obs > args.cells:
        idx = np.random.default_rng(1).choice(A.n_obs, size=args.cells, replace=False)
        A = A[idx].copy()

    # Ensure subject_id exists
    sid_col = _pick_subject(A.obs)
    A.obs["subject_id"] = A.obs[sid_col].astype(str)

    # Light normalization (optional)
    try:
        import scanpy as sc
        sc.pp.filter_genes(A, min_cells=10)
        sc.pp.normalize_total(A, target_sum=1e4)
        sc.pp.log1p(A)
    except Exception:
        pass

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(args.out)
    print(f"[OK] wrote {args.out}  cells={A.n_obs}  genes={A.n_vars}  subjects={A.obs['subject_id'].nunique()}")

if __name__ == "__main__":
    main()
