#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, anndata as ad
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.sparse as sp

def z(x):
    x = np.asarray(x, float)
    mu, sd = np.nanmean(x), np.nanstd(x)
    return np.zeros_like(x) if (not np.isfinite(sd) or sd == 0) else (x - mu) / (sd + 1e-12)

def cpm(mat, lib):
    if sp.issparse(mat):
        return mat.multiply(1e6 / lib[:, None])
    return mat * (1e6 / lib[:, None])

def mean_program_score(A, genes):
    if not genes:
        return np.zeros(A.n_obs, dtype=float)
    M = A[:, genes].X
    lib = (np.asarray(A.X.sum(axis=1)).ravel() if sp.issparse(A.X) else A.X.sum(axis=1)).astype(float)
    lib[lib == 0] = 1.0
    CPM = cpm(M, lib)
    if sp.issparse(CPM): V = np.log1p(CPM).mean(axis=1).A1
    else:                V = np.log1p(CPM).mean(axis=1)
    return z(V)

def extract_fusion_fission_gene_sets(mitocarta_csv):
    df = pd.read_csv(mitocarta_csv)
    # symbol column
    sym = "Symbol" if "Symbol" in df.columns else ("gene" if "gene" in df.columns else None)
    if sym is None:
        raise SystemExit("Could not find Symbol/gene column in mitocarta table.")
    # pathway column
    pw_col = "MitoPathways" if "MitoPathways" in df.columns else ("Pathways" if "Pathways" in df.columns else None)
    if pw_col is None:
        raise SystemExit("Could not find MitoPathways/Pathways in mitocarta table.")
    pw = df[pw_col].astype(str).fillna("")
    sy = df[sym].astype(str)

    # liberal matching to catch both explicit tracks and sub-entries
    is_fusion  = pw.str.contains(r"fusion",  case=False)
    is_fission = pw.str.contains(r"fission", case=False)

    fusion_genes  = sorted(set(sy[is_fusion]))
    fission_genes = sorted(set(sy[is_fission]))
    return fusion_genes, fission_genes

def main():
    ap = argparse.ArgumentParser(description="Add Fusion & Fission columns to results_summary based on scRNA + MitoCarta")
    ap.add_argument("--scrna", required=True, help="h5ad file (e.g., mitoomics_gpu/data/scrna.mito.h5ad)")
    ap.add_argument("--mitocarta-table", required=True, help="mitocarta csv (compat ok)")
    ap.add_argument("--summary", required=True, help="results_breast/results_summary.csv (or _with_disease)")
    ap.add_argument("--out", required=True, help="output CSV with fusion/fission added")
    ap.add_argument("--plots", required=True, help="directory to save plots")
    args = ap.parse_args()

    A = ad.read_h5ad(args.scrna)
    if "subject_id" not in A.obs.columns:
        raise SystemExit("subject_id not found in AnnData.obs")

    fusion, fission = extract_fusion_fission_gene_sets(args.mitocarta_table)

    present = set(map(str, A.var_names))
    fusion_in  = sorted(g for g in fusion  if g in present)
    fission_in = sorted(g for g in fission if g in present)

    if len(fusion_in) == 0 and len(fission_in) == 0:
        raise SystemExit("No fusion or fission genes present in scRNA matrix.")

    fusion_cell  = mean_program_score(A, fusion_in)
    fission_cell = mean_program_score(A, fission_in)

    subj = (A.obs.assign(fusion_cell=fusion_cell, fission_cell=fission_cell)
              .groupby("subject_id")[["fusion_cell","fission_cell"]].mean()
              .rename(columns={"fusion_cell":"fusion","fission_cell":"fission"}))
    subj["fusion_minus_fission"] = subj["fusion"] - subj["fission"]

    df = pd.read_csv(args.summary)
    out = df.merge(subj, left_on="subject_id", right_index=True, how="left")

    Path(args.plots).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # plots (best-effort)
    try:
        # Top 20 balance
        top = subj.sort_values("fusion_minus_fission", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12,6), dpi=160)
        ax.bar(range(len(top)), top["fusion_minus_fission"])
        ax.set_xticks(range(len(top))); ax.set_xticklabels(top.index, rotation=75, ha="right")
        ax.set_ylabel("fusion - fission (z)"); ax.set_title("Top 20 by fusion–fission balance")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout(); fig.savefig(Path(args.plots)/"bar_top20_fusion_minus_fission.png", bbox_inches="tight")

        # scatters vs MHI if available
        if "MHI" in out.columns:
            for col in ["fusion","fission","fusion_minus_fission"]:
                fig, ax = plt.subplots(figsize=(7.2,5.4), dpi=160)
                ax.scatter(out[col], out["MHI"], s=40, alpha=0.85, edgecolor="none")
                ax.set_xlabel(col); ax.set_ylabel("MHI"); ax.set_title(f"{col} vs MHI")
                ax.grid(True, linestyle=":", alpha=0.5)
                fig.tight_layout(); fig.savefig(Path(args.plots)/f"scatter_{col}_vs_MHI.png", bbox_inches="tight")
    except Exception as e:
        print("[WARN] plotting failed:", e)

    print(f"[OK] wrote {args.out} and plots to {args.plots}")
    print(f"present genes → fusion={len(fusion_in)} fission={len(fission_in)}")

if __name__ == "__main__":
    main()
