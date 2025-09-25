#!/usr/bin/env python3
"""
build_gene_assets_from_xls_fixed.py

Reads a Human.MitoCarta3.0 Excel (XLS/XLSX) with columns:
  - Symbol
  - MitoCarta3.0_MitoPathways  (optional but preferred)

Outputs (to --outdir):
  - mitocarta3_genes.txt
  - mitocarta3_table.csv       (cols: gene, MitoPathways?)
  - genesets_curated.csv       (fusion/fission/mitophagy/biogenesis from tags)
  - ev_whitelist.csv           (seeded/merged list)

Usage:
  python tools/build_gene_assets_from_xls_fixed.py \
    --xls data/Human.MitoCarta3.0.xls \
    --outdir data \
    --sheet 0              # OPTIONAL: sheet index/name if first sheet isn't the right one
"""
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
import pandas as pd

GENE_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-]{1,12}$")   # permissive HGNC-like tokens

def log(m: str) -> None:
    print(m, flush=True)

def die(m: str, code: int = 2) -> None:
    log(f"ERROR: {m}")
    sys.exit(code)

def ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_sheet(xls: Path, sheet_arg):
    df = pd.read_excel(xls, sheet_name=sheet_arg)
    if isinstance(df, dict):  # if a dict of sheets was returned
        # take the first one if not specified
        df = next(iter(df.values()))
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xls", required=True, help="Path to Human.MitoCarta3.0.xls/.xlsx")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--sheet", default=None, help="Sheet index or name (optional)")
    args = ap.parse_args()

    xls = Path(args.xls)
    if not xls.exists():
        die(f"File not found: {xls}")
    outdir = Path(args.outdir); ensure_outdir(outdir)

    # Decide sheet selector
    sheet_sel = None
    if args.sheet is not None:
        try:
            sheet_sel = int(args.sheet)
        except ValueError:
            sheet_sel = str(args.sheet)

    # Read sheet
    df = load_sheet(xls, sheet_sel)
    cols = list(df.columns)

    # Validate required columns
    required = ["Symbol"]
    missing = [c for c in required if c not in cols]
    if missing:
        # Helpful dump for debugging
        log("Available columns:")
        for c in cols: log(f"  - {c}")
        die(f"Missing required column(s): {missing}. Make sure you chose the right sheet.")

    # Build trimmed table
    has_mitopath = "MitoCarta3.0_MitoPathways" in cols
    keep = ["Symbol"] + (["MitoCarta3.0_MitoPathways"] if has_mitopath else [])
    t = df[keep].copy()
    t = t.rename(columns={"Symbol":"gene", "MitoCarta3.0_MitoPathways":"MitoPathways"})
    t["gene"] = t["gene"].astype(str).str.strip().str.upper()
    # Filter to gene-like tokens and drop duplicates
    t = t[t["gene"].map(lambda x: bool(GENE_TOKEN_RE.match(x)))]
    t = t.drop_duplicates(subset=["gene"]).reset_index(drop=True)

    # Save mitocarta genes + table
    (outdir/"mitocarta3_genes.txt").write_text("\n".join(sorted(t["gene"].tolist())))
    t.to_csv(outdir/"mitocarta3_table.csv", index=False)
    log(f"wrote mitocarta3_genes.txt ({t.shape[0]} genes)")
    log(f"wrote mitocarta3_table.csv ({t.shape[0]} rows)")

    # Build genesets_curated.csv from MitoPathways tags if present
    import pandas as pd, re
    rows = []
    if has_mitopath:
        for _, r in t.dropna(subset=["MitoPathways"]).iterrows():
            g = r["gene"]
            tags = re.split(r";|,|\||\t|\s{2,}", str(r["MitoPathways"]).lower())
            if any("fusion" in s for s in tags):     rows.append({"pathway":"fusion","gene":g})
            if any("fission" in s for s in tags):    rows.append({"pathway":"fission","gene":g})
            if any(("mitophagy" in s) or ("autophagy" in s and "mito" in s) for s in tags):
                rows.append({"pathway":"mitophagy","gene":g})
            if any(("biogen" in s) or ("organization" in s) or ("import" in s) or ("tfam" in s) for s in tags):
                rows.append({"pathway":"biogenesis","gene":g})
    gs = pd.DataFrame(rows, columns=["pathway","gene"]).dropna()
    if gs.empty:
        gs = pd.DataFrame(columns=["pathway","gene"])
    gs["pathway"] = gs["pathway"].astype(str).str.lower()
    gs["gene"]    = gs["gene"].astype(str).str.upper()
    gs = gs.drop_duplicates().sort_values(["pathway","gene"]).reset_index(drop=True)
    gs.to_csv(outdir/"genesets_curated.csv", index=False)
    log(f"wrote genesets_curated.csv ({len(gs)} rows)")

    # Seed/merge EV whitelist
    wl_seed = {
        "TOMM20","VDAC1","ATP5F1A","ATP5F1B","ATP5MC1","SDHA","COX4I1",
        "NDUFS1","NDUFS2","NDUFA9","MFN2","OPA1","DNM1L","PHB2","PINK1","PRKN","TFAM","SLC25A3"
    }
    wl_path = outdir/"ev_whitelist.csv"
    if wl_path.exists():
        try:
            prev = pd.read_csv(wl_path)["protein"].astype(str).str.upper().tolist()
            wl_seed |= set(prev)
        except Exception:
            pass
    pd.DataFrame(sorted(wl_seed), columns=["protein"]).to_csv(wl_path, index=False)
    log(f"wrote ev_whitelist.csv ({len(wl_seed)} proteins)")
    log("DONE.")

if __name__ == "__main__":
    main()
