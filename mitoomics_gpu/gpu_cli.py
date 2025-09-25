#!/usr/bin/env python3
"""
cli_gpu.py
Compute MHI with GPU acceleration where available.
- Reads your scRNA .h5ad, EV proteomics CSV, gene sets CSV
- Produces: subject-level MHI table + useful CSVs for plotting
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import anndata as ad
import pandas as pd
import numpy as np

from .gpu_backend import GB
from .gpu_accel import pca, umap, neighbors, score_gene_sets, ev_index, corr_matrix

def compute_mhi(scores_df: pd.DataFrame, ev_idx: pd.Series | None):
    """
    Minimal transparent integration:
    MHI = mean(z-scored [Copy, Fusion, Fission, Mitophagy] + optional EV_index)
    You can replace this with a learned weighting later.
    """
    # pick component columns
    want = [c for c in scores_df.columns if any(k in c.lower() for k in ["copy", "fusion", "fission", "mitophagy", "biogenesis"])]
    M = scores_df[want].copy()
    # z per column
    M = (M - M.mean())/(M.std(ddof=0) + 1e-8)
    if ev_idx is not None and not ev_idx.empty:
        M = M.join(ev_idx, how="left")
        M["EV_index"] = (M["EV_index"] - M["EV_index"].mean())/(M["EV_index"].std(ddof=0) + 1e-8)
        want += ["EV_index"]
    MHI = M.mean(axis=1).rename("MHI")
    out = M.join(MHI)
    return out

def main():
    ap = argparse.ArgumentParser(description="MitoOmics-GPU: GPU-accelerated MHI computation")
    ap.add_argument("--scrna", required=True, help=".h5ad with obs: subject_id, cell_type")
    ap.add_argument("--proteomics", required=False, help="EV proteomics CSV")
    ap.add_argument("--genesets_csv", required=True, help="genesets_curated.csv (pathway,gene)")
    ap.add_argument("--ev_whitelist", required=False, help="ev_whitelist.csv (protein)")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--layer", default=None, help="AnnData layer to use (default .X)")
    ap.add_argument("--neighbors", type=int, default=15)
    ap.add_argument("--do_umap", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    print(f"[backend] GPU available: {GB.has_gpu}")
    adata = ad.read_h5ad(args.scrna)
    assert "subject_id" in adata.obs, "obs['subject_id'] missing"
    if "cell_type" not in adata.obs:
        adata.obs["cell_type"] = "unknown"

    # 1) pathway scores per cell (Copy/Fusion/Fission/Mitophagy/Biogenesis)
    gs = pd.read_csv(args.genesets_csv)
    cell_scores = score_gene_sets(adata, gs, layer=args.layer, method="zmean")  # cells x pathways

    # 2) collapse to subject-level means
    cell_scores["subject_id"] = adata.obs["subject_id"].values
    subj_scores = cell_scores.groupby("subject_id").mean(numeric_only=True)

    # 3) EV index per subject (optional)
    ev_idx = None
    if args.proteomics:
        ev = pd.read_csv(args.proteomics)
        if args.ev_whitelist and Path(args.ev_whitelist).exists():
            wl = set(pd.read_csv(args.ev_whitelist)["protein"].astype(str).str.upper())
        else:
            wl = {"TOMM20","VDAC1","ATP5F1A","ATP5F1B","ATP5MC1","SDHA","COX4I1","NDUFS1","NDUFS2","NDUFA9","MFN2","OPA1","DNM1L","PHB2","PINK1","PRKN","TFAM","SLC25A3"}
        # Resolve column names heuristically
        subj_col = next((c for c in ev.columns if "subject" in c.lower() or "sample" in c.lower() or "donor" in c.lower() or "patient" in c.lower()), None)
        prot_col = next((c for c in ev.columns if "protein" in c.lower() or "gene" in c.lower() or "symbol" in c.lower() or "accession" in c.lower()), None)
        val_col  = next((c for c in ev.columns if any(k in c.lower() for k in ["abundance","intensity","lfq","reporter","value","quantity"])), None)
        if not (subj_col and prot_col and val_col):
            print("[warn] could not resolve EV columns; skipping EV_index.")
        else:
            ev_idx = ev_index(ev, wl, subject_col=subj_col, protein_col=prot_col, value_col=val_col)

    # 4) Compose MHI
    table = compute_mhi(subj_scores, ev_idx)
    table.index.name = "subject_id"
    table.reset_index().to_csv(outdir / "results_summary_GPU.csv", index=False)

    # 5) Optional embeddings for viz (PCA/UMAP on cell x gene space)
    try:
        X = adata.layers[args.layer] if args.layer else adata.X
        if hasattr(X, "toarray"): X = X.toarray()
        X = np.asarray(X, dtype=float)
        X_pca, comps = pca(X, n_components=min(50, X.shape[1]))
        pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]).to_csv(outdir / "embedding_pca_GPU.csv", index=False)
        if args.do_umap:
            E = umap(X_pca, n_components=2, n_neighbors=args.neighbors)
            pd.DataFrame(E, columns=["UMAP1","UMAP2"]).to_csv(outdir / "embedding_umap_GPU.csv", index=False)
    except Exception as e:
        print(f"[warn] embedding step failed: {e}")

    # 6) Diagnostics
    if ev_idx is not None and not ev_idx.empty:
        joined = table.join(ev_idx, how="left")
        joined.to_csv(outdir / "results_with_EV_GPU.csv", index=True)
        # quick corr matrix
        corr = corr_matrix(joined.select_dtypes(include=[float, int]))
        if not corr.empty:
            corr.to_csv(outdir / "corr_components_GPU.csv")

    print(f"[done] wrote results to: {outdir}")
    print(f"[hint] GPU used: {GB.has_gpu}")
if __name__ == "__main__":
    main()
