#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, anndata as ad, scipy.sparse as sp
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    A = ad.read_h5ad(a.inp)

    # Try to set var_names to gene symbols
    sym_col = None
    for c in ("feature_name","gene_symbol","symbol","gene","Gene","Symbol"):
        if c in A.var.columns:
            sym_col = c; break
    if sym_col is None:
        # symbols might be in .var.index already—just use that
        A.var["symbol"] = A.var.index.astype(str)
        sym_col = "symbol"
    else:
        A.var["symbol"] = A.var[sym_col].astype(str)

    # keep original IDs if they exist
    if "feature_id" not in A.var.columns:
        A.var["feature_id"] = A.var.index.astype(str)

    # assign symbols to var_names and make unique
    A.var_names = A.var["symbol"].astype(str).values
    A.var_names_make_unique()

    # QC tables
    X = A.X
    det = (X>0).sum(axis=0).A1 if sp.issparse(X) else (X>0).sum(axis=0)
    det = pd.Series(det, index=A.var_names, name="n_cells_detected")
    det = det.sort_values(ascending=False)
    outbase = Path(a.out).with_suffix("")
    det.to_csv(outbase.with_name(outbase.name + "_gene_detection.csv"))

    # obs summaries
    def top_counts(col):
        if col in A.obs.columns:
            vc = A.obs[col].astype(str).value_counts()
            vc.to_csv(outbase.with_name(outbase.name + f"_obs_{col}_counts.csv"))

    for col in ("tissue_general","disease","cell_type","donor_id","sample_id","dataset_id","subject_id"):
        top_counts(col)

    # write fixed h5ad
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(a.out)
    print(f"[OK] wrote {a.out}  cells={A.n_obs} genes={A.n_vars}")

if __name__ == "__main__":
    main()
