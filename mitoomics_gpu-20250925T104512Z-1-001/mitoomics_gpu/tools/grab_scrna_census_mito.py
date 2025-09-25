#!/usr/bin/env python3
# tools/grab_scrna_census_mito.py
from __future__ import annotations
import argparse, sys, math, numpy as np, pandas as pd, anndata as ad
from pathlib import Path

# Requires: cellxgene-census, tiledbsoma, anndata (and optionally scanpy)
import cellxgene_census

CENSUS_VERSION = "2025-01-30"  # pin for reproducibility
DEFAULT_TOP_K = 1500           # if we must pick by rank/score

def log(msg: str):
    print(f"[grab_mito] {msg}", file=sys.stderr)

# ---------- Mito gene list from MitoCarta ----------
def load_mito_symbols(
    mitocarta_csv: str,
    symbol_col_hint: str | None = None,
    membership_col_hint: str | None = None,
    maestro_col_hint: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[str]:
    df = pd.read_csv(mitocarta_csv)
    cols_lower = {c.lower(): c for c in df.columns}

    # 1) find a symbol column
    symbol_col = None
    if symbol_col_hint and symbol_col_hint in df.columns:
        symbol_col = symbol_col_hint
    else:
        for key in ("symbol", "gene symbol", "genesymbol", "gene"):
            if key in cols_lower:
                symbol_col = cols_lower[key]; break
    if symbol_col is None:
        raise SystemExit("Could not find a Symbol column in MitoCarta CSV (try --mitocarta-symbol-col).")

    # 2) try a boolean membership column
    if membership_col_hint and membership_col_hint in df.columns:
        mem_col = membership_col_hint
    else:
        mem_col = None
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ("is_mito", "mitocarta", "mitochond", "mc3", "membership")):
                mem_col = c; break

    mito_syms: list[str] = []

    if mem_col is not None:
        log(f"Using membership column: {mem_col}")
        s = df[mem_col]
        def truthy(x):
            if pd.isna(x): return False
            x = str(x).strip().lower()
            return x in ("1","true","t","yes","y")
        mito_syms = df.loc[s.map(truthy), symbol_col].dropna().astype(str).str.strip().unique().tolist()

    # 3) else try a numeric Maestro/Rank column
    if not mito_syms:
        # choose a numeric column that looks like score/rank
        cand_cols = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                cl = c.lower()
                if any(k in cl for k in ("maestro","rank","mito","score")):
                    cand_cols.append(c)
        if cand_cols:
            # pick the "best looking" one
            maestro_col = maestro_col_hint if maestro_col_hint in df.columns else cand_cols[0]
            log(f"Using numeric column for selection: {maestro_col}")
            s = df[maestro_col].astype(float)
            # Decide direction: if correlation with row index suggests "lower is better" (rank), use ascending
            # heuristic: compare top_k by ascending vs descending; take which has more MT- genes as a weak proxy
            syms = df[symbol_col].astype(str)
            asc = df.sort_values(maestro_col, ascending=True).head(min(top_k, len(df)))[symbol_col]
            desc = df.sort_values(maestro_col, ascending=False).head(min(top_k, len(df)))[symbol_col]
            asc_mt = (asc.str.upper().str.startswith("MT-")).sum()
            desc_mt = (desc.str.upper().str.startswith("MT-")).sum()
            choice = asc if asc_mt >= desc_mt else desc
            mito_syms = choice.dropna().astype(str).str.strip().unique().tolist()

    # 4) last-resort fallback: keep only MT- genes from the symbol column
    if not mito_syms:
        log("Falling back to MT- only (mtDNA-encoded genes).")
        mito_syms = (
            df[symbol_col]
            .dropna().astype(str).str.strip()
            .pipe(lambda s: s[s.str.upper().str.startswith("MT-")])
            .unique().tolist()
        )

    mito_syms = sorted({g for g in mito_syms if g and g != "nan"})
    log(f"Selected {len(mito_syms)} mitochondrial symbols.")
    if len(mito_syms) < 10:
        log("WARNING: very small mito gene list; check your membership/maestro settings.")
    return mito_syms

# ---------- Filters & helpers ----------
def in_filter(col: str, values: list[str]) -> str:
    uniq = sorted({v for v in values if v})
    items = ", ".join(repr(v) for v in uniq)
    return f"{col} in [{items}]"

def pick_subject(obs: pd.DataFrame) -> str:
    for k in ("donor_id", "sample_id", "dataset_id"):
        if k in obs.columns:
            return k
    obs["synthetic_subject"] = "subject0"
    return "synthetic_subject"

def tumor_mask(obs: pd.DataFrame) -> pd.Series:
    bad = {"normal","healthy","control","none","not reported","na","nan",""}
    cols = [c for c in ("disease", "disease_label", "disease__ontology_label") if c in obs.columns]
    if not cols:
        return pd.Series(True, index=obs.index)
    keep = pd.Series(False, index=obs.index)
    for c in cols:
        s = obs[c].astype(str).str.strip().str.lower()
        keep |= ~s.isin(bad)
    return keep

# ---------- Fetch a chunk of genes (by var filter) ----------
def fetch_chunk_anndata(
    tissues: list[str],
    genes_chunk: list[str],
    census_version: str = CENSUS_VERSION,
):
    obs_filter = in_filter("tissue_general", tissues)
    var_filter = in_filter("feature_name", genes_chunk)

    # Try old/new get_anndata signatures
    with cellxgene_census.open_soma(census_version=census_version) as census:
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
            # Older versions may not accept var_value_filter -> fetch then subset in-memory
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
            # Subset to requested genes if var filter wasn’t used
            keep = [g for g in genes_chunk if g in list(A.var_names)]
            if keep:
                A = A[:, keep].copy()
    return A

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Download tumor scRNA (mitochondrial genes only) from CELLxGENE Census")
    ap.add_argument("--mitocarta", required=True, help="Path to mitocarta3_table.csv")
    ap.add_argument("--mitocarta-symbol-col", default=None, help="Symbol column name (optional)")
    ap.add_argument("--mitocarta-membership-col", default=None, help="Boolean membership column (optional)")
    ap.add_argument("--mitocarta-maestro-col", default=None, help="Numeric rank/score column (optional)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K when selecting by rank/score")
    ap.add_argument("--cells", type=int, default=120_000, help="Max cells to keep after load")
    ap.add_argument("--tissues", nargs="+", default=["lung","breast","colon","pancreas"], help="tissue_general filters")
    ap.add_argument("--tumor-only", action="store_true", help="Filter out normal/healthy samples")
    ap.add_argument("--chunk-size", type=int, default=400, help="Genes per var-filter chunk to keep query strings modest")
    ap.add_argument("--out", default="data/scrna.h5ad", help="Output AnnData path")
    args = ap.parse_args()

    # 1) Build mito gene list
    mito_syms = load_mito_symbols(
        mitocarta_csv=args.mitocarta,
        symbol_col_hint=args.mitocarta_symbol_col,
        membership_col_hint=args.mitocarta_membership_col,
        maestro_col_hint=args.mitocarta_maestro_col,
        top_k=args.top_k,
    )
    if not mito_syms:
        raise SystemExit("No mitochondrial symbols were selected from MitoCarta.")

    # 2) Fetch in chunks (var filter) and concat by genes (axis=1)
    chunks = [mito_syms[i:i+args.chunk_size] for i in range(0, len(mito_syms), args.chunk_size)]
    adatas: list[ad.AnnData] = []
    for idx, genes_chunk in enumerate(chunks, 1):
        log(f"Fetching chunk {idx}/{len(chunks)} with {len(genes_chunk)} genes...")
        Ai = fetch_chunk_anndata(args.tissues, genes_chunk, census_version=CENSUS_VERSION)
        adatas.append(Ai)

    # Ensure same cells across chunks; concat on var axis
    A = ad.concat(adatas, axis=1, join="outer", merge="unique", label=None, index_unique=None, fill_value=0)

    # 3) Tumor-only filter (optional)
    if args.tumor_only:
        mask = tumor_mask(A.obs)
        A = A[mask].copy()
        if A.n_obs == 0:
            raise SystemExit("No tumor cells matched filters; try without --tumor-only or change tissues.")

    # 4) Downsample cells if needed
    if A.n_obs > args.cells:
        idx = np.random.default_rng(1).choice(A.n_obs, size=args.cells, replace=False)
        A = A[idx].copy()

    # 5) Ensure subject_id exists
    sid_col = None
    for k in ("donor_id", "sample_id", "dataset_id"):
        if k in A.obs.columns:
            sid_col = k; break
    if sid_col is None:
        A.obs["subject_id"] = "subject0"
    else:
        A.obs["subject_id"] = A.obs[sid_col].astype(str)

    # 6) Lightweight normalization (optional, skip if scanpy unavailable)
    try:
        import scanpy as sc
        sc.pp.filter_genes(A, min_cells=10)
        sc.pp.normalize_total(A, target_sum=1e4)
        sc.pp.log1p(A)
    except Exception:
        pass

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    A.write_h5ad(args.out)
    log(f"OK wrote {args.out}  cells={A.n_obs}  genes={A.n_vars}  subjects={A.obs['subject_id'].nunique()}")

if __name__ == "__main__":
    main()
