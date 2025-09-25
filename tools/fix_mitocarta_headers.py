#!/usr/bin/env python3
import argparse, pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.inp)
    cols = set(df.columns)

    # Map 'gene' -> 'Symbol' (and add a few friendly aliases just in case)
    if "gene" in cols and "Symbol" not in cols:
        df["Symbol"] = df["gene"].astype(str)
    if "GeneSymbol" not in cols and "Symbol" in df.columns:
        df["GeneSymbol"] = df["Symbol"]
    if "Gene Symbol" not in cols and "Symbol" in df.columns:
        df["Gene Symbol"] = df["Symbol"]

    # Ensure a generic pathways alias exists
    if "MitoPathways" in df.columns and "Pathways" not in df.columns:
        df["Pathways"] = df["MitoPathways"]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"[OK] wrote {a.out}  columns={list(df.columns)}  rows={len(df)}")

if __name__ == "__main__":
    main()
