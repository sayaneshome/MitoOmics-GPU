from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from .io import (
    load_scrna, load_proteomics, load_imaging,
    load_mitocarta_pathways, load_ev_whitelist
)
from .scoring import copy_number_proxy, program_scores, heterogeneity_scores
from .ev_integration import ev_pathway_scores
from .mhi import combine_components
from .report import save_figures, write_report

def main():
    ap = argparse.ArgumentParser(description="MitoOmics-GPU end-to-end pipeline")
    ap.add_argument("--scrna", required=True, help="Path to .h5ad scRNA file")
    ap.add_argument("--proteomics", required=True, help="EV/MDV proteomics CSV (subject_id,protein,abundance)")
    ap.add_argument("--imaging", default=None, help="Optional imaging CSV with subject_id and metrics")
    ap.add_argument("--outdir", required=True, help="Output directory")
    # MitoCarta wiring
    ap.add_argument("--mitocarta-table", required=True,
                    help="Path to mitocarta3_table.csv produced from Human.MitoCarta3.0.xls (sheet 2)")
    ap.add_argument("--ev-whitelist", default=None,
                    help="Optional ev_whitelist.csv with a column 'protein' (or 'Symbol')")
    # Targets / model fit
    ap.add_argument("--fit-ridge", action="store_true", help="Fit Ridge regression if targets provided")
    ap.add_argument("--targets", default=None, help="CSV with columns subject_id,target")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    adata = load_scrna(args.scrna)
    ev_df = load_proteomics(args.proteomics)
    im_df = load_imaging(args.imaging)  # currently unused but loaded for future use

    # ---- Load pathway sets from MitoCarta table
    pathways = load_mitocarta_pathways(args.mitocarta_table)  # dict: pathway -> [symbols]
    if len(pathways) == 0:
        raise ValueError("No pathways found from MitoCarta table. Check columns and CSV content.")

    # ---- Optional EV whitelist
    whitelist = load_ev_whitelist(args.ev_whitelist) if args.ev_whitelist else None

    # ---- Components from scRNA ----
    cnp_cell = copy_number_proxy(adata)
    adata.obs["copy_number_proxy"] = cnp_cell.values
    cnp_subject = adata.obs.groupby("subject_id")["copy_number_proxy"].mean().rename("copy_number")

    prog_scores_cell = program_scores(adata, pathways=pathways)
    adata.obs = adata.obs.join(prog_scores_cell)

    # Define fusion_fission as the mean of any available fusion/fission-like pathways
    # We try canonical names but also fall back to any pathway that contains keywords.
    def pick_cols(df, keys):
        cols = []
        for k in keys:
            if k in df.columns:
                cols.append(k)
        return cols
    fusion_like = pick_cols(prog_scores_cell, ["fusion"]) or [
        c for c in prog_scores_cell.columns if "fusion" in c.lower()
    ]
    fission_like = pick_cols(prog_scores_cell, ["fission"]) or [
        c for c in prog_scores_cell.columns if "fission" in c.lower()
    ]
    ff_cols = list(set(fusion_like + fission_like))
    if ff_cols:
        ff_cell = prog_scores_cell[ff_cols].mean(axis=1).rename("fusion_fission_cell")
        adata.obs["fusion_fission_cell"] = ff_cell
        ff_subject = adata.obs.groupby("subject_id")["fusion_fission_cell"].mean().rename("fusion_fission")
    else:
        ff_subject = pd.Series(dtype=float, name="fusion_fission")

    # Mitophagy subject score (if present)
    if "mitophagy" in prog_scores_cell.columns:
        mitophagy_subject = adata.obs.groupby("subject_id")["mitophagy"].mean().rename("mitophagy")
    else:
        # fallback: mean of any path containing 'mitophagy'
        mito_cols = [c for c in prog_scores_cell.columns if "mitophagy" in c.lower()]
        mitophagy_subject = (prog_scores_cell.assign(subject_id=adata.obs["subject_id"])  # join for groupby safety
                             .groupby("subject_id")[mito_cols].mean().mean(axis=1)
                             .rename("mitophagy")) if mito_cols else pd.Series(dtype=float, name="mitophagy")

    # Heterogeneity (Shannon over cell_type, scaled)
    het_subject = heterogeneity_scores(adata).rename("heterogeneity")

    # ---- Components from EV proteomics ----
    ev_path = ev_pathway_scores(ev_df, pathways=pathways, whitelist=whitelist)

    # ---- Merge subject-level components
    subject_ids = sorted(set(adata.obs["subject_id"].unique()) | set(ev_path["subject_id"].unique()))
    subject_df = pd.DataFrame({"subject_id": subject_ids})

    for s in [cnp_subject, ff_subject, mitophagy_subject, het_subject]:
        if s is not None and len(s) > 0:
            # robust merge: use right index so we don't depend on reset_index naming
            try:
                s_df = s.to_frame(name=getattr(s, 'name', 'mitophagy'))
            except Exception:
                s_df = s  # already a DataFrame
            subject_df = subject_df.merge(s_df, left_on='subject_id', right_index=True, how='left')

    if not ev_path.empty:
        ev_cols = [c for c in ev_path.columns if c != "subject_id"]
        subject_df = subject_df.merge(ev_path, on="subject_id", how="left", suffixes=("","_ev"))

    # Targets (optional)
    ridge_targets = None
    if args.targets:
        tgt = pd.read_csv(args.targets)
        if not {"subject_id","target"}.issubset(tgt.columns):
            raise ValueError("Targets CSV must have columns: subject_id,target")
        ridge_targets = tgt.set_index("subject_id")["target"]

    # Compute MHI
    results = combine_components(
        subject_df,
        fit_ridge=args.fit_ridge and (ridge_targets is not None),
        targets=ridge_targets
    )
    results.to_csv(outdir / "results_summary.csv", index=False)

    figs = save_figures(results, outdir)
    write_report(results, figs, outdir)

    print(f"[OK] Wrote {outdir / 'results_summary.csv'} and report assets.")

if __name__ == "__main__":
    main()
