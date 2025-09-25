from __future__ import annotations
import pandas as pd
import anndata as ad
from collections import defaultdict

def load_scrna(h5ad_path: str):
    adata = ad.read_h5ad(h5ad_path)
    if "subject_id" not in adata.obs:
        adata.obs["subject_id"] = "SUBJ1"
    if "cell_type" not in adata.obs:
        adata.obs["cell_type"] = "unknown"
    if "batch" not in adata.obs:
        adata.obs["batch"] = "batch1"
    adata.var_names_make_unique()
    return adata

def load_proteomics(csv_path: str):
    df = pd.read_csv(csv_path)
    needed = {"subject_id","protein","abundance"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Proteomics CSV must have {needed}, got {set(df.columns)}")
    df["abundance"] = df["abundance"].astype(float)
    df["abundance_norm"] = df.groupby("subject_id")["abundance"].transform(
        lambda x: x / (x.sum() + 1e-9)
    )
    return df

def load_imaging(csv_path: str | None):
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    if "subject_id" not in df.columns:
        raise ValueError("Imaging CSV must include 'subject_id'")
    return df

def load_mitocarta_pathways(mitocarta_csv_path: str) -> dict[str, list[str]]:
    """
    Build pathway -> [symbols] from MitoCarta table.
    Expects columns like:
      - Symbol
      - MitoCarta3.0_MitoPathways (string with semicolon/comma-separated pathways)
    Pathways are normalized to lowercase, spaces -> underscores.
    """
    mc = pd.read_csv(mitocarta_csv_path)
    # Try a few likely column name variants
    symbol_col_candidates = ["Symbol", "symbol", "SYMBOL", "GeneSymbol", "Gene Symbol"]
    path_col_candidates = [
        "MitoCarta3.0_MitoPathways", "MitoPathways", "MitoCarta3.0 MitoPathways",
        "MitoCarta3.0_MitoPathway", "MitoPathway"
    ]

    def first_present(cols, candidates):
        for c in candidates:
            if c in cols:
                return c
        return None

    sym_col = first_present(mc.columns, symbol_col_candidates)
    p_col = first_present(mc.columns, path_col_candidates)

    if sym_col is None:
        raise ValueError(f"Could not find a Symbol column among: {symbol_col_candidates}")
    if p_col is None:
        # If MitoPathways column is absent, return empty and let caller decide fallback
        return {}

    # Clean and split pathway strings
    def split_paths(x):
        if pd.isna(x):
            return []
        s = str(x).replace("|", ";").replace(",", ";")
        parts = [p.strip() for p in s.split(";") if p.strip()]
        return parts

    pathways = defaultdict(list)
    for _, row in mc.iterrows():
        sym = str(row[sym_col]).strip()
        if not sym or sym.lower() in {"nan", "none"}:
            continue
        for p in split_paths(row[p_col]):
            norm = p.lower().replace(" ", "_").replace("-", "_")
            pathways[norm].append(sym)

    # deduplicate
    for k in list(pathways.keys()):
        pathways[k] = sorted(set(pathways[k]))
    return dict(pathways)

def load_ev_whitelist(ev_whitelist_csv: str | None) -> set[str] | None:
    if ev_whitelist_csv is None:
        return None
    df = pd.read_csv(ev_whitelist_csv)
    # Try flexible column names
    col = None
    for c in ["protein","Protein","Symbol","SYMBOL","gene","Gene"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError("ev_whitelist.csv must have a column named one of: protein, Symbol, gene")
    syms = set(str(x).strip().upper() for x in df[col].dropna().tolist())
    return syms if syms else None
