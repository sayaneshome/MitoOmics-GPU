#!/usr/bin/env python3
import argparse, os, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return np.nan
    x = x[m] - x[m].mean(); y = y[m] - y[m].mean()
    den = (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum()))
    return float((x*y).sum()/den) if den else np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="results_summary.csv (or _with_disease)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topn", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # If annotated file exists, prefer disease_mode for coloring
    disease_col = "disease_mode" if "disease_mode" in df.columns else ( "disease" if "disease" in df.columns else None )

    # 1) Top-N bar (nice formatting)
    top = df.sort_values("MHI", ascending=False).head(args.topn).copy()
    fig, ax = plt.subplots(figsize=(12,6), dpi=160)
    bars = ax.bar(range(len(top)), top["MHI"])
    ax.set_title("Top {} subjects by MHI".format(args.topn))
    ax.set_ylabel("MHI"); ax.set_xticks(range(len(top))); ax.set_xticklabels(top["subject_id"], rotation=75, ha="right")
    # annotate bars
    for i,(b,v) in enumerate(zip(bars, top["MHI"].values)):
        ax.text(i, v+0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
    ax.margins(x=0.01); ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(); fig.savefig(outdir/"pretty_mhi_top{}.png".format(args.topn), bbox_inches="tight")

    # 2) Scatter helpers
    def scatter_xy(xcol, ycol="MHI", name=None):
        name = name or f"{xcol}_vs_{ycol}"
        x = df[xcol]; y = df[ycol]
        r = pearson(x, y)
        fig, ax = plt.subplots(figsize=(7.2,5.4), dpi=160)
        if disease_col:
            # simple categorical coloring (up to ~10 unique)
            cats = df[disease_col].astype(str).fillna("NA")
            levels = cats.value_counts().index.tolist()[:10]
            palette = [plt.cm.tab10(i%10) for i in range(len(levels))]
            color_map = {lvl: palette[i] for i,lvl in enumerate(levels)}
            for lvl in levels:
                m = cats==lvl
                ax.scatter(x[m], y[m], s=36, alpha=0.8, label=lvl, edgecolor="none")
            if len(levels) <= 10:
                ax.legend(frameon=False, fontsize=8, loc="best")
        else:
            ax.scatter(x, y, s=40, alpha=0.85, edgecolor="none")
        # regression line
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 3:
            k, b = np.polyfit(x[m], y[m], 1)
            xx = np.linspace(float(x[m].min()), float(x[m].max()), 100)
            ax.plot(xx, k*xx+b, linewidth=2)
        ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(f"{xcol} vs {ycol}  (r={r:.2f})")
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout(); fig.savefig(outdir/f"pretty_scatter_{name}.png", bbox_inches="tight")

    # make the key scatters
    for col in ["fusion_fission","mitophagy","heterogeneity","copy_number"]:
        if col in df.columns:
            scatter_xy(col, "MHI", col)

    # 3) Correlation heatmap across numeric metrics
    num = df.select_dtypes(include=[np.number]).copy()
    if not num.empty:
        cols = [c for c in num.columns if num[c].nunique(dropna=True) > 1]
        C = num[cols].corr(numeric_only=True).values
        fig, ax = plt.subplots(figsize=(0.5+0.45*len(cols), 0.5+0.45*len(cols)), dpi=160)
        im = ax.imshow(C, vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=75, ha="right", fontsize=8)
        ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=8)
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center", fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation heatmap")
        fig.tight_layout(); fig.savefig(outdir/"pretty_corr_heatmap.png", bbox_inches="tight")

    # 4) Mini HTML gallery
    imgs = [p for p in ["pretty_mhi_top{}.png".format(args.topn),
                        "pretty_scatter_fusion_fission_vs_MHI.png",
                        "pretty_scatter_mitophagy_vs_MHI.png",
                        "pretty_scatter_heterogeneity_vs_MHI.png",
                        "pretty_scatter_copy_number_vs_MHI.png",
                        "pretty_corr_heatmap.png"] if (outdir/p).exists()]
    html = "<h1>MitoOmics — Gorgeous Plots</h1>\n" + "\n".join(f'<div><img src="{i}" style="max-width:100%"></div>' for i in imgs)
    (outdir/"index.html").write_text(html)
    print("[OK] Wrote:", outdir)

if __name__ == "__main__":
    main()
