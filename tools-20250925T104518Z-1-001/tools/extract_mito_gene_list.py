#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, numpy as np, pandas as pd
from pathlib import Path

def log(m): print(f"[mito-genelist] {m}", file=sys.stderr)

def truthy(x):
    if pd.isna(x): return False
    s = str(x).strip().lower()
    return s in {"1","true","t","yes","y","mito","mitochondrial","member"}

def pick_symbol_col(df: pd.DataFrame, hint: str|None) -> str:
    if hint and hint in df.columns: return hint
    low = {c.lower(): c for c in df.columns}
    for k in ("symbol","gene symbol","genesymbol","gene"):
        if k in low: return low[k]
    raise SystemExit("Could not find a Symbol column. Pass --symbol-col EXACT_NAME.")

def pick_membership_col(df: pd.DataFrame, hint: str|None) -> str|None:
    if hint and hint in df.columns: return hint
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ("is_mito","mitocarta","mitochond","membership","mc3","mito member","mitochondria")):
            return c
    return None

def candidate_numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cl = c.lower()
            if any(k in cl for k in ("maestro","rank","score","mito","localiz","prob","posterior")):
                cols.append(c)
    if not cols:
        # fallback: pick highest variance numeric col
        nc = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if nc:
            v = [(c, float(pd.to_numeric(df[c], errors="coerce").var(skipna=True))) for c in nc]
            v.sort(key=lambda t: (np.isnan(t[1]), -t[1]))
            if v and not np.isnan(v[0][1]): cols = [v[0][0]]
    return cols

def select_by_rank(df: pd.DataFrame, sym_col: str, rank_col: str, top_k: int) -> list[str]:
    # decide direction using MT- heuristic
    asc  = df.sort_values(rank_col, ascending=True )[sym_col].astype(str).head(min(top_k, len(df)))
    desc = df.sort_values(rank_col, ascending=False)[sym_col].astype(str).head(min(top_k, len(df)))
    asc_mt  = (asc.str.upper().str.startswith("MT-")).sum()
    desc_mt = (desc.str.upper().str.startswith("MT-")).sum()
    chosen = asc if asc_mt >= desc_mt else desc
    syms = [g.strip() for g in chosen.dropna().tolist() if g and g != "nan"]
    return sorted(set(syms))

def main():
    ap = argparse.ArgumentParser(description="Extract mito symbol list from MitoCarta CSV")
    ap.add_argument("--mitocarta", required=True)
    ap.add_argument("--symbol-col", default=None)
    ap.add_argument("--membership-col", default=None)
    ap.add_argument("--rank-col", default=None)
    ap.add_argument("--top-k", type=int, default=1500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fail-if-fallback", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.mitocarta)
    sym = pick_symbol_col(df, args.symbol_col)
    log(f"Symbol column: {sym}")

    # 1) membership
    mem = pick_membership_col(df, args.membership_col)
    if mem is not None:
        log(f"Trying membership column: {mem}")
        m = df[mem].map(truthy)
        n = int(m.sum())
        if n >= 50:
            syms = df.loc[m, sym].dropna().astype(str).str.strip().unique().tolist()
            syms = sorted(set(syms))
            log(f"Membership selected {len(syms)} genes")
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                for g in syms: f.write(g+"\n")
            print(f"[OK] wrote {args.out}  n_genes={len(syms)}")
            return
        else:
            log(f"Membership column rejected (only {n} truthy).")

    # 2) rank/score
    rank_col = args.rank_col
    if not rank_col:
        cand = candidate_numeric_cols(df)
        if cand: rank_col = cand[0]
    if rank_col:
        log(f"Using rank/score column: {rank_col}")
        syms = select_by_rank(df, sym, rank_col, args.top_k)
        if len(syms) >= 100:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w") as f:
                for g in syms: f.write(g+"\n")
            print(f"[OK] wrote {args.out}  n_genes={len(syms)}  rank_col={rank_col}")
            return
        else:
            log(f"Rank column produced too few genes ({len(syms)}).")

    # 3) fallback MT- only
    fallback = sorted(set(df[sym].dropna().astype(str).str.strip().tolist()))
    fallback = [g for g in fallback if g.upper().startswith("MT-")]
    if args.fail_if_fallback:
        raise SystemExit("Refusing MT-only fallback. Specify --membership-col or --rank-col EXACT column name.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for g in fallback: f.write(g+"\n")
    print(f"[WARN] wrote {args.out}  n_genes={len(fallback)} (MT-only fallback)")

if __name__ == "__main__":
    main()
