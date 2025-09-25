#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, numpy as np, pandas as pd, anndata as ad
from pathlib import Path
import cellxgene_census

CENSUS_VERSION = "2025-01-30"

def log(m): print(f"[mito-census] {m}", file=sys.stderr)

def _in_list(col: str, values: list[str]) -> str:
    vals = ", ".join(repr(v.strip()) for v in sorted({v for v in values if v}))
    return f"{col} in [{vals}]"

def load_mito_symbols(
    csv_path: str,
    symbol_col_hint: str|None,
    membership_col_hint: str|None,
    maestro_col_hint: str|None,
    top_k: int,
    fail_if_fallback: bool,
) -> list[str]:
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    # symbol column
    sym = None
    if symbol_col_hint and symbol_col_hint in cols:
        sym = symbol_col_hint
    else:
        for k in ("symbol","gene symbol","genesymbol","gene"):
            if k in lower: sym = lower[k]; break
    if sym is None:
        raise SystemExit("No Symbol column found in mitocarta CSV. Pass --mitocarta-symbol-col")

    # membership column (truthy)
    mem = None
    if membership_col_hint and membership_col_hint in cols:
        mem = membership_col_hint
    else:
        for c in cols:
            cl = c.lower()
            if any(k in cl for k in ("is_mito","mitocarta","mitochond","membership","mc3")):
                mem = c; break
    if mem is not None:
        s = df[mem]
        def truthy(x):
            if pd.isna(x): return False
            t = str(x).strip().lower()
            return t in ("1","true","t","yes","y")
        syms = df.loc[s.map(truthy), sym].dropna().astype(str).str.strip().unique().tolist()
        syms = sorted({g for g in syms if g})
        if len(syms) >= 50:
            log(f"Using membership column '{mem}' → {len(syms)} genes")
            return syms

    # score/rank column
    maestro = None
    if maestro_col_hint and maestro_col_hint in cols:
        maestro = maestro_col_hint
    else:
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        cand = []
        for c in numeric_cols:
            cl = c.lower()
            if any(k in cl for k in ("maestro","rank","score","mito","mitocarta","localiz")):
                cand.append(c)
        if not cand and numeric_cols:
            # pick numeric col with largest variance
            variances = [(c, float(pd.to_numeric(df[c], errors="coerce").var(skipna=True))) for c in numeric_cols]
            variances.sort(key=lambda t: (np.isnan(t[1]), -t[1]))
            if variances and not np.isnan(variances[0][1]):
                cand = [variances[0][0]]
        maestro = cand[0] if cand else None

    if maestro is not None:
        asc = df.sort_values(maestro, ascending=True )[sym].head(min(top_k, len(df))).astype(str)
        dsc = df.sort_values(maestro, ascending=False)[sym].head(min(top_k, len(df))).astype(str)
        asc_mt  = (asc.str.upper().str.startswith("MT-")).sum()
        dsc_mt  = (dsc.str.upper().str.startswith("MT-")).sum()
        chosen = asc if asc_mt >= dsc_mt else dsc
        syms = sorted({g.strip() for g in chosen.dropna().tolist() if g and g != "nan"})
        log(f"Using rank/score column '{maestro}' → {len(syms)} genes (top_k={min(top_k, len(df))})")
        return syms

    # fallback: MT- only
    syms = (
        df[sym].dropna().astype(str).str.strip()
        .pipe(lambda s: s[s.str.upper().str.startswith("MT-")])
        .unique().tolist()
    )
    if fail_if_fallback:
        raise SystemExit("Refusing to fallback to MT- only (13 genes). Pass proper --mitocarta-* flags.")
    log(f"Falling back to MT- only → {len(syms)} genes")
    return syms

def tumor_mask(obs: pd.DataFrame) -> pd.Series:
    bad = {"normal","healthy","control","none","not reported","na","nan",""}
    cols = [c for c in ("disease", "disease_label", "disease__ontology_label") if c in obs.columns]
    if not cols: return pd.Series(True, index=obs.index)
    keep = pd.Series(False, index=obs.index)
    for c in cols:
        s = obs[c].astype(str).str.strip().str.lower()
        keep |= ~s.isin(bad)
    return keep

def fetch_chunk(tissues: list[str], genes: list[str]) -> ad.AnnData:
    obs_filter = _in_list("tissue_general", tissues)
    var_filter = _in_list("feature_name", genes)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        try:
            A = cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                measurement_name="RNA",
                X_name="raw",
                obs_value_filter=obs_filter,
                var_value_filter=var_filter,
            )
        except TypeError:
            try:
                A = cellxgene_census.get_anndata(
                    census,
                    organism="Homo sapiens",
                    measurement_name="RNA",
                    X_name="raw",
                    obs_value_filter=obs_filter,
                )
            except TypeError:
                A = cellxgene_census.get_anndata(
                    organism="Homo sapiens",
                    measurement_name="RNA",
                    X_name="raw",
                    obs_value_filter=obs_filter,
                    var_value_filter=var_filter,
                )
            keep = [g for g in genes if g in list(A.var_names)]
            if keep:
                A = A[:, keep].copy()
    return A

def main():
    ap = argparse.ArgumentParser(description="Download tumor scRNA (mitochondrial genes only) from CELLxGENE Census")
    ap.add_argument("--mitocarta", required=True)
    ap.add_argument("--mitocarta-symbol-col", default=None)
    ap.add_argument("--mitocarta-membership-col", default=None)
    ap.add_argument("--mitocarta-maestro-col", default=None)
    ap.add_argument("--top-k", type=int, default=1500)
    ap.add_argument("--fail-if-fallback", action="store_true")
    ap.add_argument("--cells", type=int, default=60000)
    ap.add_argument("--tissues", nargs="+", default=["lung","breast","colon","pancreas"])
    ap.add_argument("--tumor-only", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    genes = load_mito_symbols(
        csv_path=args.mitocarta,
        symbol_col_hint=args.mitocarta_symbol_col,
        membership_col_hint=args.mitocarta_membership_col,
        maestro_col_hint=args.mitocarta_maestro_col,
        top_k=args.top_k,
        fail_if_fallback=args.fail_if_fallback,
    )
    if len(genes) == 0:
        raise SystemExit("No genes selected from MitoCarta.")
    chunks = [genes[i:i+args.chunk_size] for i in range(0, len(genes), args.chunk_size)]

    adatas = []
    for i, g in enumerate(chunks, 1):
        log(f"Fetching chunk {i}/{len(chunks)} with {len(g)} genes")
        adatas.append(fetch_chunk(args.tissues, g))

    A = ad.concat(adatas, axis=1, join="outer", merge="unique", label=None, index_unique=None, fill_value=0)

    if args.tumor_only:
        mask = tumor_mask(A.obs)
        A = A[mask].copy()
        if A.n_obs == 0:
            raise SystemExit("No tumor cells matched filters.")

    if A.n_obs > args.cells:
        idx = np.random.default_rng(1).choice(A.n_obs, size=args.cells, replace=False)
        A = A[idx].copy()

    sid = None
    for k in ("donor_id","sample_id","dataset_id"):
        if k in A.obs.columns:
            sid = k; break
    A.obs["subject_id"] = A.obs[sid].astype(str) if sid else "subject0"

    try:
        import scanpy as sc
        sc.pp.filter_genes(A, min_cells=10)
        sc.pp.normalize_total(A, target_sum=1e4)
        sc.pp.log1p(A)
    except Exception:
        pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(args.out)
    log(f"OK wrote {args.out} cells={A.n_obs} genes={A.n_vars} subjects={A.obs['subject_id'].nunique()}")

if __name__ == "__main__":
    main()
