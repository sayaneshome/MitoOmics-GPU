#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, numpy as np, anndata as ad
from pathlib import Path
import cellxgene_census, pandas as pd

CENSUS_VERSION="2025-01-30"

def log(m): print(f"[fast-genelist] {m}", file=sys.stderr)

def _in_list(col, vals):
    return f"{col} in [" + ", ".join(repr(v.strip()) for v in sorted(set(vals)) if v) + "]"

def tumor_mask(obs: pd.DataFrame) -> pd.Series:
    bad={"normal","healthy","control","none","not reported","na","nan",""}
    keep=pd.Series(False,index=obs.index)
    for c in ("disease","disease_label","disease__ontology_label"):
        if c in obs.columns:
            s=obs[c].astype(str).str.strip().str.lower()
            keep |= ~s.isin(bad)
    return keep if keep.any() else pd.Series(True,index=obs.index)

def fetch_chunk(tissues, genes):
    obs_filter=_in_list("tissue_general", tissues)
    var_filter=_in_list("feature_name", genes)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        try:
            A=cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                measurement_name="RNA",
                X_name="raw",
                obs_value_filter=obs_filter,
                var_value_filter=var_filter,
            )
        except TypeError:
            try:
                A=cellxgene_census.get_anndata(
                    census,
                    organism="Homo sapiens",
                    measurement_name="RNA",
                    X_name="raw",
                    obs_value_filter=obs_filter,
                )
            except TypeError:
                A=cellxgene_census.get_anndata(
                    organism="Homo sapiens",
                    measurement_name="RNA",
                    X_name="raw",
                    obs_value_filter=obs_filter,
                    var_value_filter=var_filter,
                )
            keep=[g for g in genes if g in list(A.var_names)]
            if keep:
                A=A[:, keep].copy()
    return A

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--genes", required=True)
    ap.add_argument("--cells", type=int, default=30000)
    ap.add_argument("--tissues", nargs="+", default=["breast"])
    ap.add_argument("--tumor-only", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()

    genes=[g.strip() for g in open(a.genes) if g.strip()]
    genes=sorted(set(genes))
    if not genes:
        raise SystemExit("Gene list is empty.")

    chunks=[genes[i:i+a.chunk_size] for i in range(0,len(genes),a.chunk_size)]
    parts=[]
    for i,g in enumerate(chunks,1):
        log(f"Fetching chunk {i}/{len(chunks)} with {len(g)} genes")
        parts.append(fetch_chunk(a.tissues,g))

    A=ad.concat(parts, axis=1, join="outer", merge="unique", label=None, index_unique=None, fill_value=0)

    if a.tumor_only:
        A=A[tumor_mask(A.obs)].copy()
        if A.n_obs==0:
            raise SystemExit("No tumor cells matched filters.")

    if A.n_obs>a.cells:
        idx=np.random.default_rng(1).choice(A.n_obs, size=a.cells, replace=False)
        A=A[idx].copy()

    sid=None
    for k in ("donor_id","sample_id","dataset_id"):
        if k in A.obs.columns:
            sid=k; break
    A.obs["subject_id"]=A.obs[sid].astype(str) if sid else "subject0"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(a.out)
    log(f"OK wrote {a.out} cells={A.n_obs} genes={A.n_vars} subjects={A.obs['subject_id'].nunique()}")

if __name__=="__main__":
    main()
