#!/usr/bin/env python3
# tools/autostandardize_proteomics.py
# Auto-detect sheet/columns from PRIDE processed tables and emit:
#   subject_id, protein (gene symbol), abundance
from __future__ import annotations
import argparse, re, sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
import pandas as pd
import numpy as np

# ----------------- heuristics -----------------
SUBJECT_PREFIXES = [
    "LFQ intensity", "Intensity", "Reporter intensity", "Reporter intensity corrected",
    "Abundance", "Normalized Abundance", "iBAQ", "IBAQ", "MS/MS Count", "Quantity", "Area"
]
SUBJECT_ID_CANDIDATES = [
    "SampleID","Sample","Subject","Patient","Raw file","Raw file name","Experiment",
    "Run","FileName","Filename","File name","Channel","TMT Channel","iTRAQ Channel"
]
ABUNDANCE_CANDIDATES = [
    "LFQ intensity","Intensity","Reporter intensity","Reporter intensity corrected",
    "Abundance","Normalized Abundance","iBAQ","IBAQ","Quantity","Area"
]
ANNOTATION_HINTS = {
    "protein ids","majority protein ids","gene names","gene name","protein",
    "fasta headers","description","protein names","sequence","peptides","razor",
    "q-value","fdr","posterior error","score","accession","accessions","coverage"
}
SHEET_DOWNWEIGHTS = ["param","readme","summary","meta","note"]

# ----------------- I/O helpers -----------------
def read_table_auto(input_path: str) -> Tuple[pd.DataFrame, str]:
    """Auto-pick best sheet: must have gene/protein cols and >=2 subject columns if possible."""
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(p)
    ext = p.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(p)
        choices = []
        for sh in xls.sheet_names:
            try:
                df0 = pd.read_excel(p, sheet_name=sh, nrows=200)
            except Exception:
                continue
            gene_col, prot_col, subj_cols, subj_long = detect_columns(df0)
            # score: prefer >=2 wide subject columns; otherwise long-format availability
            wide_ok = len(subj_cols) >= 2
            long_ok = subj_long is not None
            # penalize “params/summary” sheets
            penalty = any(k in sh.strip().lower() for k in SHEET_DOWNWEIGHTS)
            score = (
                2 if wide_ok else (1 if long_ok else 0),
                int(gene_col is not None or prot_col is not None),
                len(subj_cols),
                -int(penalty)
            )
            choices.append((score, sh))
        # pick best score
        choices.sort(reverse=True)
        best = choices[0][1] if choices else xls.sheet_names[0]
        df = pd.read_excel(p, sheet_name=best)
        return df, best
    elif ext in {".csv", ".tsv", ".txt"}:
        sep = "," if ext == ".csv" else "\t"
        return pd.read_csv(p, sep=sep), "<single>"
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def read_table_specific(input_path: str, sheet: str) -> Tuple[pd.DataFrame, str]:
    p = Path(input_path)
    if p.suffix.lower() in {".xlsx",".xls"}:
        # allow numeric index or name
        try:
            idx = int(sheet)
            df = pd.read_excel(p, sheet_name=idx)
            return df, str(sheet)
        except ValueError:
            df = pd.read_excel(p, sheet_name=sheet)
            return df, sheet
    else:
        # sheet ignored for CSV/TSV
        return read_table_auto(input_path)

# ----------------- detection -----------------
def detect_columns(df: pd.DataFrame):
    """Return (gene_col, prot_col, subj_wide_cols, subj_long_tuple or None).
       subj_long_tuple = (subject_id_col, abundance_col) if long-format detected."""
    lc = {str(c).lower(): c for c in df.columns}

    gene_col = None
    for cand in ["gene names","gene name","symbol","gene","genes"]:
        if cand in lc:
            gene_col = lc[cand]; break

    prot_col = None
    for cand in ["majority protein ids","protein ids","protein id","accession","accessions","uniprot","uniprot id","uniprot ids"]:
        if cand in lc:
            prot_col = lc[cand]; break

    # wide-format subject columns by prefixes
    cols = [str(c) for c in df.columns]
    subj_wide = []
    for c in cols:
        for pref in SUBJECT_PREFIXES:
            if str(c).startswith(pref):
                subj_wide.append(c); break

    # if not enough, include numeric columns that aren't clearly annotation
    if len(subj_wide) < 2:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            cl = str(c).lower()
            if any(h in cl for h in ANNOTATION_HINTS):
                continue
            if c not in subj_wide:
                subj_wide.append(c)
        # keep columns with enough non-nulls
        nn = df[subj_wide].notna().sum() if subj_wide else pd.Series(dtype=int)
        subj_wide = [c for c in subj_wide if nn.get(c,0) >= max(3, int(0.02*len(df)))]

    subj_wide = sorted(set(subj_wide))

    # long-format: find a subject id column + one abundance column
    subj_id_col = None
    for cand in SUBJECT_ID_CANDIDATES:
        if cand in lc:
            subj_id_col = lc[cand]; break
    abundance_col = None
    for cand in ABUNDANCE_CANDIDATES:
        # exact column or any column starting with the candidate
        if cand in lc:
            abundance_col = lc[cand]; break
        for col in df.columns:
            if str(col).startswith(cand):
                abundance_col = col; break
        if abundance_col is not None:
            break
    subj_long = (subj_id_col, abundance_col) if (subj_id_col and abundance_col) else None

    return gene_col, prot_col, subj_wide, subj_long

# ----------------- mappings -----------------
def build_uniprot_to_symbol(mitocarta_csv: str) -> Dict[str,str]:
    mc = pd.read_csv(mitocarta_csv)
    sym_col = None
    for c in ["Symbol","GeneSymbol","Gene Symbol","SYMBOL","symbol"]:
        if c in mc.columns: sym_col = c; break
    uni_col = None
    for c in ["UniProt","Uniprot","UniprotID","UniProtID","Uniprot ID","UniProt Accession"]:
        if c in mc.columns: uni_col = c; break
    if sym_col is None or uni_col is None:
        raise ValueError("mitocarta3_table.csv must have UniProt and Symbol columns")
    m: Dict[str,str] = {}
    for _, r in mc[[uni_col, sym_col]].dropna().iterrows():
        u = str(r[uni_col]).strip()
        s = str(r[sym_col]).strip().upper()
        if not u or not s: 
            continue
        for uid in re.split(r"[;,\s]+", u):
            uid = uid.strip()
            if uid:
                m[uid] = s
    return m

def to_symbol(cell: str, u2s: Dict[str,str]) -> Optional[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)): return None
    s = str(cell).strip()
    if not s: return None
    # direct gene symbol heuristic
    if '|' not in s and ';' not in s and ':' not in s and len(s) <= 15 and any(ch.isalpha() for ch in s) and s.upper()==s:
        return s.upper()
    # UniProt-like lists
    toks = re.split(r"[;,\s|]+", s)
    for t in toks:
        t = t.strip()
        if not t: continue
        if t in u2s: return u2s[t]
        m = re.match(r".*\|([OPQ][0-9][A-Z0-9]{3}[0-9])\|.*", t)
        if m and m.group(1) in u2s: return u2s[m.group(1)]
    return None

def normalize_subject(colname: str) -> str:
    s = re.sub(r"^(LFQ intensity|Intensity|Reporter intensity(?: corrected)?|Abundance|Normalized Abundance|iBAQ|IBAQ|MS/MS Count|Quantity|Area)\s*", "", str(colname)).strip()
    s = re.sub(r"\s+", "_", s)
    return s if s else str(colname)

def load_whitelist(path: Optional[str]) -> Optional[Set[str]]:
    if not path: return None
    t = pd.read_csv(path)
    col = None
    for c in ["protein","Protein","Symbol","SYMBOL","gene","Gene"]:
        if c in t.columns: col = c; break
    if not col: raise ValueError("Whitelist must have a column: protein/Symbol/gene")
    return set(str(x).strip().upper() for x in t[col].dropna())

# ----------------- main logic -----------------
def convert_wide(df: pd.DataFrame, gene_col: Optional[str], prot_col: Optional[str],
                 subj_cols: List[str], u2s: Dict[str,str]) -> pd.DataFrame:
    if gene_col:
        prot_series = df[gene_col].astype(str).str.split(r"[;,\s]+").str[0].str.upper()
    elif prot_col:
        prot_series = df[prot_col].apply(lambda x: to_symbol(x, u2s))
    else:
        raise RuntimeError("Could not find gene/protein column to map symbols from.")
    df = df.copy()
    df["_symbol"] = prot_series
    df = df.dropna(subset=["_symbol"])
    long = df[["_symbol"] + subj_cols].melt(
        id_vars=["_symbol"], value_vars=subj_cols,
        var_name="subject_id", value_name="abundance"
    )
    long["abundance"] = pd.to_numeric(long["abundance"], errors="coerce")
    long = long.dropna(subset=["abundance"])
    long = long[long["abundance"] > 0]
    long["subject_id"] = long["subject_id"].apply(normalize_subject)
    out = (long.rename(columns={"_symbol":"protein"})
                .groupby(["subject_id","protein"], as_index=False)["abundance"].sum())
    return out

def convert_long(df: pd.DataFrame, gene_col: Optional[str], prot_col: Optional[str],
                 subj_id_col: str, abundance_col: str, u2s: Dict[str,str]) -> pd.DataFrame:
    if gene_col:
        prot_series = df[gene_col].astype(str).str.split(r"[;,\s]+").str[0].str.upper()
    elif prot_col:
        prot_series = df[prot_col].apply(lambda x: to_symbol(x, u2s))
    else:
        raise RuntimeError("Could not find gene/protein column to map symbols from.")
    out = pd.DataFrame({
        "subject_id": df[subj_id_col].astype(str).map(lambda s: re.sub(r"\s+","_",s.strip())),
        "protein": prot_series,
        "abundance": pd.to_numeric(df[abundance_col], errors="coerce")
    }).dropna(subset=["abundance"])
    out = out[out["abundance"] > 0]
    out = out.groupby(["subject_id","protein"], as_index=False)["abundance"].sum()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Processed table (xlsx/xls/csv/tsv)")
    ap.add_argument("--out", dest="out", required=True, help="Output CSV (subject_id,protein,abundance)")
    ap.add_argument("--mitocarta", required=True, help="mitocarta3_table.csv (for UniProt→Symbol)")
    ap.add_argument("--ev-whitelist", default=None, help="Optional EV whitelist CSV (protein/Symbol/gene col)")
    # Overrides if auto-detect picks wrong things:
    ap.add_argument("--sheet", default=None, help="Sheet name or index to force (for Excel)")
    ap.add_argument("--subject-col", default=None, help="(Long format) subject/sample column")
    ap.add_argument("--abundance-col", default=None, help="(Long format) abundance column")
    ap.add_argument("--gene-col", default=None, help="Explicit gene symbol column (e.g., 'Gene names')")
    ap.add_argument("--protein-col", default=None, help="Explicit protein/UniProt column (if no gene-col)")
    ap.add_argument("--min-samples", type=int, default=2, help="Min wide subject columns to prefer (default 2)")
    args = ap.parse_args()

    # Read table
    if args.sheet:
        df, sheet = read_table_specific(args.inp, args.sheet)
    else:
        df, sheet = read_table_auto(args.inp)
    print(f"[INFO] Loaded table from sheet: {sheet}", file=sys.stderr)

    # Detect columns if not overridden
    gene_col, prot_col, subj_wide, subj_long = detect_columns(df)
    # Apply explicit overrides
    if args.gene_col: gene_col = args.gene_col
    if args.protein_col: prot_col = args.protein_col
    if args.subject_col and args.abundance_col:
        subj_long = (args.subject_col, args.abundance_col)

    print(f"[INFO] Detected gene_col={gene_col} prot_col={prot_col} wide_subjects={len(subj_wide)} long={subj_long is not None}", file=sys.stderr)

    u2s = build_uniprot_to_symbol(args.mitocarta)

    # Prefer wide if enough; else use long if available; else error
    if len(subj_wide) >= args.min_samples:
        out = convert_wide(df, gene_col, prot_col, subj_wide, u2s)
    elif subj_long is not None:
        out = convert_long(df, gene_col, prot_col, subj_long[0], subj_long[1], u2s)
    else:
        # help user debug quickly
        print("[ERROR] Could not find enough subject columns (wide) nor a (subject, abundance) pair (long).", file=sys.stderr)
        print("[HINT] Try one of:", file=sys.stderr)
        print("  --sheet proteins  (or a likely sheet name like 'ProteinGroups', 'report')", file=sys.stderr)
        print("  --subject-col SampleID --abundance-col 'LFQ intensity'  (if long format)", file=sys.stderr)
        print("  --gene-col 'Gene names'   OR   --protein-col 'Protein IDs'", file=sys.stderr)
        raise SystemExit(2)

    # Whitelist
    wl = load_whitelist(args.ev_whitelist)
    if wl:
        out = out[out["protein"].str.upper().isin(wl)]

    # Write
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {out.shape[0]} rows, {out['subject_id'].nunique()} subjects, {out['protein'].nunique()} proteins.", file=sys.stderr)

if __name__ == "__main__":
    main()
