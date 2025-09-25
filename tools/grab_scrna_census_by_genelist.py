#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, numpy as np, anndata as ad
from pathlib import Path
import cellxgene_census
import pandas as pd

CENSUS_VERSION = "2025-01-30"

def log(m): print(f"[census-genelist] {m}", file=sys.stderr)

def _in_list(col: str, values: list[str]) -> str:
    vals = ", ".join(repr(v.strip()) for v in sorted({v for v in values if v}))
    return f"{col} in [{vals}]"

def tumor_mask(obs: pd.DataFrame) -> pd.Series:
    bad = {"normal","healthy","control","none","not reported","na","nan",""}
    cols = [c for c in ("disease", "disease_label", "disease__ontology_label") if c in obs.columns]
    if not cols: 
        return pd.Series(True, index=obs.index)
    keep = pd.Series(False, index=obs.index)
    for c in cols:
        s = obs[c].astype(str).str.strip().str.lower()
        keep |= ~s.isin(bad)
    return keep

def fetch_chunk(tissues: list[str], genes: list[str]) -> ad.AnnData:
    obs_filter = _in_list("tissue_general", tissues)
    var_filter = _in_list("feature_name", genes)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        try:
            A = cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                measurement_name="RNA",
                X_name="raw",
                obs_value_filter=obs_filter,
                var_value_filter=var_filter,
            )
        except TypeError:
            try:
                A = cellxgene_census.get_anndata(
                    census,
                    organism="Homo sapiens",
                    measurement_name="RNA",
                    X_name="raw",
                    obs_value_filter=obs_filter,
                )
            except TypeError:
                A = cellxgene_census.get_anndata(
                    organism="Homo sapiens",
                    measurement_name="RNA",
                    X_name="raw",
                    obs_value_filter=obs_filter,
                    var_value_filter=var_filter,
                )
            keep = [g for g in genes if g in list(A.var_names)]
            if keep: A = A[:, keep].copy()
    return A

def main():
    ap = argparse.ArgumentParser(description="Download scRNA for a provided gene symbol list")
    ap.add_argument("--genes", required=True, help="Text file: one gene symbol per line")
    ap.add_argument("--cells", type=int, default=60000)
    ap.add_argument("--tissues", nargs="+", default=["lung","breast","colon","pancreas"])
    ap.add_argument("--tumor-only", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.genes, "r") as f:
        genes = [g.strip() for g in f if g.strip()]
    genes = sorted(set(genes))
    if not genes:
        raise SystemExit("Gene list is empty.")

    chunks = [genes[i:i+args.chunk_size] for i in range(0, len(genes), args.chunk_size)]
    adatas = []
    for i, g in enumerate(chunks, 1):
        log(f"Fetching chunk {i}/{len(chunks)} with {len(g)} genes")
        adatas.append(fetch_chunk(args.tissues, g))

    A = ad.concat(adatas, axis=1, join="outer", merge="unique", label=None, index_unique=None, fill_value=0)

    if args.tumor_only:
        mask = tumor_mask(A.obs)
        A = A[mask].copy()
        if A.n_obs == 0:
            raise SystemExit("No tumor cells matched filters.")

    if A.n_obs > args.cells:
        idx = np.random.default_rng(1).choice(A.n_obs, size=args.cells, replace=False)
        A = A[idx].copy()

    sid = None
    for k in ("donor_id","sample_id","dataset_id"):
        if k in A.obs.columns:
            sid = k; break
    A.obs["subject_id"] = A.obs[sid].astype(str) if sid else "subject0"

    try:
        import scanpy as sc
        sc.pp.filter_genes(A, min_cells=10)
        sc.pp.normalize_total(A, target_sum=1e4)
        sc.pp.log1p(A)
    except Exception:
        pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(args.out)
    log(f"OK wrote {args.out} cells={A.n_obs} genes={A.n_vars} subjects={A.obs['subject_id'].nunique()}")

if __name__ == "__main__":
    main()
