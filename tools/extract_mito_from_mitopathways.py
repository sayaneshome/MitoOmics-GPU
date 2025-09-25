#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Extract mito genes from MitoPathways column")
    ap.add_argument("--mitocarta", required=True, help="CSV with columns: gene, MitoPathways")
    ap.add_argument("--symbol-col", default="gene")
    ap.add_argument("--pathways-col", default="MitoPathways")
    ap.add_argument("--out", required=True, help="Output text file (one symbol per line)")
    args = ap.parse_args()

    df = pd.read_csv(args.mitocarta)
    if args.symbol_col not in df.columns or args.pathways_col not in df.columns:
        raise SystemExit(f"Expected columns '{args.symbol_col}' and '{args.pathways_col}' in {args.mitocarta}")

    s = df[args.pathways_col].astype(str).fillna("").str.strip()
    # two candidate rules:
    keep_any  = s.ne("") & ~s.str.lower().isin({"na","nan","none","-","0"})
    keep_mito = keep_any & s.str.contains(r"mito|mitochond", case=False, regex=True)

    # prefer entries explicitly mentioning mito; otherwise keep any non-empty pathway
    mask = keep_mito if keep_mito.sum() >= 100 else keep_any

    genes = (df.loc[mask, args.symbol_col]
               .dropna().astype(str).str.strip()
               .pipe(lambda x: x[x.ne("")]).unique().tolist())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for g in sorted(set(genes)):
            f.write(g + "\n")
    print(f"[OK] wrote {args.out}  n_genes={len(set(genes))}")

if __name__ == "__main__":
    main()
