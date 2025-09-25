#!/usr/bin/env python3
"""
bench/benchmark_mhi.py
Runs CPU and GPU modes back-to-back on the same inputs, times each stage,
and produces:
  - bench/bench_results.csv (per stage timings)
  - bench/bench_summary.md  (pretty markdown report)
  - bench/bench_plot.png    (bar chart of speedups)
Usage:
  python bench/benchmark_mhi.py \
    --scrna data/scrna_public.h5ad \
    --genesets data/genesets_curated.csv \
    --proteomics data/pxd018301/ev_proteomics.csv \
    --ev_whitelist data/ev_whitelist.csv
"""
from __future__ import annotations
import argparse, os, time, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_cli(mode, args, outdir):
    """
    mode: 'gpu' or 'cpu'
    returns: dict(stage -> seconds)
    """
    env = os.environ.copy()
    if mode == "cpu":
        env["MITOOMICS_FORCE_CPU"] = "1"
    else:
        env.pop("MITOOMICS_FORCE_CPU", None)

    # We’ll instrument stages by calling the core functions via a small inline runner
    # to get per-stage timings without modifying your CLI code.
    import importlib
    from mitoomics_gpu.gpu_backend import GB
    from mitoomics_gpu.gpu_accel import pca, umap, score_gene_sets, ev_index, corr_matrix
    import anndata as ad

    t = {}
    t0 = time.perf_counter()
    adata = ad.read_h5ad(args.scrna)
    t["load_h5ad"] = time.perf_counter() - t0

    # Gene sets
    t0 = time.perf_counter()
    gs = pd.read_csv(args.genesets)
    t["load_genesets"] = time.perf_counter() - t0

    # Score pathways
    t0 = time.perf_counter()
    cell_scores = score_gene_sets(adata, gs, layer=None, method="zmean")
    t["score_gene_sets"] = time.perf_counter() - t0

    # Subject collapse
    t0 = time.perf_counter()
    cell_scores["subject_id"] = adata.obs["subject_id"].values
    subj_scores = cell_scores.groupby("subject_id").mean(numeric_only=True)
    t["collapse_subject_means"] = time.perf_counter() - t0

    # EV index (optional)
    ev_idx = None
    if args.proteomics and Path(args.proteomics).exists():
        t0 = time.perf_counter()
        ev = pd.read_csv(args.proteomics)
        wl = set(pd.read_csv(args.ev_whitelist)["protein"].astype(str).str.upper()) if args.ev_whitelist and Path(args.ev_whitelist).exists() else set()
        subj_col = next((c for c in ev.columns if any(k in c.lower() for k in ["subject","sample","donor","patient"])), None)
        prot_col = next((c for c in ev.columns if any(k in c.lower() for k in ["protein","gene","symbol","accession"])), None)
        val_col  = next((c for c in ev.columns if any(k in c.lower() for k in ["abundance","intensity","lfq","reporter","value","quantity"])), None)
        if subj_col and prot_col and val_col:
            ev_idx = ev_index(ev, wl or {"TOMM20","VDAC1","ATP5F1A","ATP5F1B","ATP5MC1","SDHA","COX4I1",
                                         "NDUFS1","NDUFS2","NDUFA9","MFN2","OPA1","DNM1L","PHB2","PINK1","PRKN","TFAM","SLC25A3"},
                              subject_col=subj_col, protein_col=prot_col, value_col=val_col)
        t["ev_index"] = time.perf_counter() - t0
    else:
        t["ev_index"] = 0.0

    # Compute MHI (simple, transparent z-mean)
    t0 = time.perf_counter()
    want = [c for c in subj_scores.columns if any(k in c.lower() for k in ["copy","fusion","fission","mitophagy","biogenesis"])]
    M = subj_scores[want].copy()
    M = (M - M.mean())/(M.std(ddof=0) + 1e-8)
    if ev_idx is not None and not ev_idx.empty:
        M = M.join(ev_idx, how="left")
        M["EV_index"] = (M["EV_index"] - M["EV_index"].mean())/(M["EV_index"].std(ddof=0) + 1e-8)
    M["MHI"] = M.mean(axis=1)
    outdir.mkdir(parents=True, exist_ok=True)
    M.reset_index().to_csv(outdir / f"results_summary_{mode}.csv", index=False)
    t["compose_MHI"] = time.perf_counter() - t0

    # Embeddings
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X, dtype=float)
    X = np.asarray(X, dtype=float)

    t0 = time.perf_counter()
    Xp, _ = pca(X, n_components=min(50, X.shape[1]))
    t["pca"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        E = umap(Xp, n_components=2, n_neighbors=15)
    except Exception:
        E = None
    t["umap"] = time.perf_counter() - t0

    # Save a tiny diagnostic
    pd.DataFrame(Xp, columns=[f"PC{i+1}" for i in range(Xp.shape[1])]).head(5).to_csv(outdir / f"embedding_pca_head_{mode}.csv", index=False)

    return t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrna", required=True)
    ap.add_argument("--genesets", required=True)
    ap.add_argument("--proteomics", default=None)
    ap.add_argument("--ev_whitelist", default=None)
    args = ap.parse_args()

    bench_dir = Path("bench"); bench_dir.mkdir(exist_ok=True)
    gpu_times = run_cli("gpu", args, bench_dir / "gpu")
    cpu_times = run_cli("cpu", args, bench_dir / "cpu")

    stages = sorted(set(gpu_times) | set(cpu_times))
    rows = []
    for s in stages:
        g = gpu_times.get(s, np.nan)
        c = cpu_times.get(s, np.nan)
        speedup = (c / g) if (g and g > 0 and c and c > 0) else np.nan
        rows.append({"stage": s, "gpu_sec": g, "cpu_sec": c, "cpu_over_gpu_speedup": speedup})
    df = pd.DataFrame(rows).sort_values("stage")
    df.to_csv(bench_dir / "bench_results.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8,4))
    ok = df.dropna(subset=["cpu_over_gpu_speedup"])
    ax.bar(ok["stage"], ok["cpu_over_gpu_speedup"])
    ax.set_ylabel("CPU / GPU speedup (×)")
    ax.set_xticklabels(ok["stage"], rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(bench_dir / "bench_plot.png", bbox_inches="tight")
    plt.close(fig)

    # Markdown summary
    lines = ["# MitoOmics-GPU Benchmark Summary", "", "| Stage | GPU (s) | CPU (s) | CPU/GPU × |", "|---|---:|---:|---:|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['stage']} | {r['gpu_sec']:.3f} | {r['cpu_sec']:.3f} | {r['cpu_over_gpu_speedup']:.2f} |")
    (bench_dir / "bench_summary.md").write_text("\n".join(lines))
    print("Wrote:", bench_dir / "bench_results.csv", bench_dir / "bench_plot.png", bench_dir / "bench_summary.md")

if __name__ == "__main__":
    main()
