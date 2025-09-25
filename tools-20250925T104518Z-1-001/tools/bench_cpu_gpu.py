#!/usr/bin/env python3
"""
bench_cpu_gpu.py
Benchmarks GPU vs CPU for program scoring on:
  (a) your h5ad if provided (default: mitoomics_gpu/data/scrna.mito.h5ad)
  (b) synthetic CSR matrices at 30k/60k/100k cells x 2k genes

Outputs:
  - bench/bench_results.json
  - bench/bench_results.csv
  - bench/bench_bar.png
Run:
  python tools/bench_cpu_gpu.py [--h5ad PATH] [--mitocarta CSV]
"""
from __future__ import annotations
import argparse, os, time, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
import scipy.sparse as sp
from pathlib import Path
from typing import Dict, List
try:
    import anndata as ad
except Exception:
    ad = None

from mitoomics_gpu.scverse_api import program_score
from mitoomics_gpu.rapids_program_score import _HAVE_GPU

def get_gene_sets_from_mitocarta(mitocarta_csv: Path, var_names: List[str]) -> Dict[str, List[str]]:
    df = pd.read_csv(mitocarta_csv)
    sym = "Symbol" if "Symbol" in df.columns else ("gene" if "gene" in df.columns else None)
    pw  = "MitoPathways" if "MitoPathways" in df.columns else ("Pathways" if "Pathways" in df.columns else None)
    if sym is None or pw is None:
        # fallback: 3 random gene sets
        vs = list(var_names)[:100]
        return {"setA": vs[:30], "setB": vs[30:60], "setC": vs[60:100]}
    sy = df[sym].astype(str); p = df[pw].astype(str)
    def sel(rx): return sorted(set(sy[p.str.contains(rx, case=False, regex=True)]))
    gs = {
        "fusion": sel("fusion"),
        "fission": sel("fission"),
        "mitophagy": sel("mitophagy"),
    }
    # intersect with var_names
    present = set(map(str, var_names))
    return {k: sorted([g for g in v if g in present]) for k,v in gs.items() if v}

def do_score(adata, gene_sets, use_gpu):
    t0 = time.time()
    df = program_score(adata, gene_sets, use_gpu=use_gpu, into_obs_prefix=None)
    dt = time.time() - t0
    return dt, df

def bench_real(h5ad_path: Path, mitocarta_csv: Path, outdir: Path):
    A = ad.read_h5ad(h5ad_path)
    gs = get_gene_sets_from_mitocarta(mitocarta_csv, list(A.var_names))
    rows = []
    for tag,use_gpu in [("cpu", False), ("gpu", True)]:
        if use_gpu and not _HAVE_GPU:
            rows.append({"dataset":"real", "engine":tag, "cells":A.n_obs, "genes":A.n_vars, "seconds": None})
            continue
        dt,_ = do_score(A, gs, use_gpu)
        rows.append({"dataset":"real", "engine":tag, "cells":A.n_obs, "genes":A.n_vars, "seconds": dt})
    return rows

def synth_adata(n_cells=30000, n_genes=2000, rate=0.05, seed=7):
    rng = np.random.default_rng(seed)
    # approximate sparsity ~95%; Poisson(0.1)
    nnz_per_row = int(n_genes * rate)
    data = []
    indptr = [0]
    indices = []
    for _ in range(n_cells):
        cols = rng.choice(n_genes, size=nnz_per_row, replace=False)
        vals = rng.poisson(0.5, size=nnz_per_row) + 1
        indices.extend(cols)
        data.extend(vals)
        indptr.append(len(indices))
    X = sp.csr_matrix((np.array(data,dtype=np.float32), np.array(indices), np.array(indptr)),
                      shape=(n_cells, n_genes))
    if ad is None:
        raise SystemExit("anndata not installed")
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])
    obs = pd.DataFrame(index=[f"C{i}" for i in range(n_cells)])
    return ad.AnnData(X=X, obs=obs, var=var)

def bench_synth(outdir: Path):
    sizes = [(30000,2000), (60000,2000), (100000,2000)]
    gene_sets = {
        "setA": [f"G{i}" for i in range(0,500,5)],
        "setB": [f"G{i}" for i in range(1,500,5)],
        "setC": [f"G{i}" for i in range(2,500,5)],
        "setD": [f"G{i}" for i in range(3,500,5)],
    }
    rows = []
    for nc, ng in sizes:
        A = synth_adata(n_cells=nc, n_genes=ng, rate=0.03, seed=42)
        for tag,use_gpu in [("cpu", False), ("gpu", True)]:
            if use_gpu and not _HAVE_GPU:
                rows.append({"dataset":f"synth_{nc}x{ng}", "engine":tag, "cells":nc, "genes":ng, "seconds": None})
                continue
            dt,_ = do_score(A, gene_sets, use_gpu)
            rows.append({"dataset":f"synth_{nc}x{ng}", "engine":tag, "cells":nc, "genes":ng, "seconds": dt})
    return rows

def plot_rows(rows, outdir: Path):
    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir/"bench_results.csv", index=False)
    (outdir/"bench_results.json").write_text(json.dumps(rows, indent=2))

    # Plot per dataset
    fig, ax = plt.subplots(figsize=(9,5), dpi=160)
    datasets = df["dataset"].unique()
    xpos = np.arange(len(datasets))
    width = 0.35
    cpu = [float(df[(df.dataset==d)&(df.engine=="cpu")]["seconds"].iloc[0]) for d in datasets]
    gpu = []
    for d in datasets:
        v = df[(df.dataset==d)&(df.engine=="gpu")]["seconds"]
        gpu.append(float(v.iloc[0]) if (len(v)>0 and pd.notna(v.iloc[0])) else np.nan)
    ax.bar(xpos - width/2, cpu, width, label="CPU")
    ax.bar(xpos + width/2, gpu, width, label="GPU")
    ax.set_xticks(xpos); ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Seconds"); ax.set_title("Program scoring — CPU vs GPU")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(outdir/"bench_bar.png", bbox_inches="tight")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="mitoomics_gpu/data/scrna.mito.h5ad")
    ap.add_argument("--mitocarta", default="mitoomics_gpu/data/mitocarta3_table.compat.csv")
    ap.add_argument("--outdir", default="bench")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    rows = []
    # real
    if ad and Path(args.h5ad).exists():
        rows += bench_real(Path(args.h5ad), Path(args.mitocarta), outdir)
    # synth
    rows += bench_synth(outdir)
    plot_rows(rows, outdir)
    print("[OK] wrote", outdir)

if __name__ == "__main__":
    main()
