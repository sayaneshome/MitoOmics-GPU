from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def save_figures(results_df: pd.DataFrame, outdir: Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figs = {}

    # Bar plot of MHI per subject (top 30 for readability)
    df = results_df.sort_values("MHI", ascending=False).head(30)
    plt.figure()
    plt.bar(df["subject_id"].astype(str), df["MHI"].values)
    plt.xticks(rotation=90)
    plt.ylabel("MHI")
    plt.title("Top 30 subjects by MHI")
    fig_path = outdir / "mhi_top30.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    figs["mhi_top30"] = str(fig_path)

    # Components vs MHI scatter (if present)
    for comp in ["copy_number","fusion_fission","mitophagy","heterogeneity"]:
        if comp in results_df.columns:
            plt.figure()
            plt.scatter(results_df[comp].values, results_df["MHI"].values, s=10)
            plt.xlabel(comp)
            plt.ylabel("MHI")
            plt.title(f"{comp} vs MHI")
            p = outdir / f"scatter_{comp}_vs_MHI.png"
            plt.tight_layout()
            plt.savefig(p, dpi=200)
            plt.close()
            figs[f"scatter_{comp}"] = str(p)

    return figs

def write_report(results_df: pd.DataFrame, figs: dict, outdir: Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    # minimal markdown report
    md = outdir / "report.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# MitoOmics-GPU Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- Subjects: **{results_df.shape[0]}**\n")
        f.write("- Metric: **MHI** (0–1 scaled combination of copy_number, fusion_fission, mitophagy, heterogeneity)\n\n")
        f.write("## Top Subjects (by MHI)\n\n")
        top = results_df.sort_values("MHI", ascending=False).head(10)
        f.write(top[["subject_id","MHI"]].to_markdown(index=False))
        f.write("\n\n## Figures\n\n")
        for name, path in figs.items():
            f.write(f"### {name}\n\n")
            f.write(f"![{name}]({Path(path).name})\n\n")
