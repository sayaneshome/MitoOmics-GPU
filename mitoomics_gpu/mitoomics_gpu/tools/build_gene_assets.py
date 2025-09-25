#!/usr/bin/env python3
"""
build_gene_assets_auto.py

Auto-detects the correct sheet + header row in a Human.MitoCarta3.0 .xls/.xlsx,
then produces:
  - mitocarta3_genes.txt
  - mitocarta3_table.csv   (cols: gene, optional MitoPathways)
  - genesets_curated.csv   (fusion/fission/mitophagy/biogenesis from tags)
  - ev_whitelist.csv

Usage:
  python3 tools/build_gene_assets_auto.py --xls data/Human.MitoCarta3.0.xls --outdir data
"""

from __future__ import annotations
import re, sys
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd

GENE_TOKEN_RE = re.compile(r"^[A-Z0-9][A-Z0-9\-]{1,12}$")  # permissive HGNC-like
SYMBOL_HEADER_RE = re.compile(r"^(symbol|gene\s*symbol|approved\s*symbol|official\s*symbol|hgnc\s*symbol|gene|gene\s*name|gene_name)$", re.I)
MITOPATH_HEADER_RE = re.compile(r"(mitopathways|mito\s*pathways|mitochondria.*pathway)", re.I)

def log(s: str): print(s, flush=True)
def die(s: str, code: int = 2): log(f"ERROR: {s}"); sys.exit(code)
def ensure_outdir(p: Path): p.mkdir(parents=True, exist_ok=True)

def normalize(col: str) -> str:
    return re.sub(r"[\s\u00A0]+", " ", str(col).strip()).lower()

def find_header_row(df_raw: pd.DataFrame, max_scan_rows: int = 30) -> Optional[int]:
    """
    Look for a row within the first `max_scan_rows` that contains a 'Symbol'-like header.
    Return the row index if found, else None.
    """
    rows = min(max_scan_rows, len(df_raw))
    for r in range(rows):
        row_vals = [normalize(x) for x in df_raw.iloc[r, :].tolist()]
        for val in row_vals:
            if SYMBOL_HEADER_RE.match(val):
                return r
    return None

def score_gene_col_by_content(df: pd.DataFrame, start_row: int = 1) -> Tuple[Optional[int], int]:
    best_idx, best_hits = None, -1
    for j in range(df.shape[1]):
        col = df.iloc[start_row:, j].astype(str).str.strip()
        hits = col.map(lambda x: bool(GENE_TOKEN_RE.match(x))).sum()
        if hits > best_hits:
            best_hits, best_idx = hits, j
    return best_idx, int(best_hits)

def extract_trimmed(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    From a *headered* DataFrame, pick 'Symbol' and optional 'MitoCarta3.0_MitoPathways'.
    Return trimmed df with columns: ['gene'] + optional ['MitoPathways'] or None if Symbol missing.
    """
    cols = list(df.columns)
    norm = [normalize(c).replace(" ", "_") for c in cols]

    # gene col by header name
    gi = None
    for i, c in enumerate(norm):
        if SYMBOL_HEADER_RE.match(c.replace("_", " ")):
            gi = i; break
    if gi is None:
        return None

    # mitopathways (optional)
    mi = None
    for i, c in enumerate(cols):
        if MITOPATH_HEADER_RE.search(str(c)):
            mi = i; break

    keep = [gi] + ([mi] if mi is not None else [])
    t = df.iloc[:, keep].copy()
    rename = {cols[gi]: "gene"}
    if mi is not None:
        rename[cols[mi]] = "MitoPathways"
    t = t.rename(columns=rename)

    t["gene"] = t["gene"].astype(str).str.strip().str.upper()
    t = t[t["gene"].map(lambda x: bool(GENE_TOKEN_RE.match(x)))]
    t = t.drop_duplicates(subset=["gene"]).reset_index(drop=True)
    return t if len(t) else None

def build_assets_from_trimmed(t: pd.DataFrame, outdir: Path) -> None:
    (outdir / "mitocarta3_genes.txt").write_text("\n".join(sorted(t["gene"].tolist())))
    t.to_csv(outdir / "mitocarta3_table.csv", index=False)
    log(f"wrote mitocarta3_genes.txt ({t.shape[0]} genes)")
    log(f"wrote mitocarta3_table.csv ({t.shape[0]} rows)")

    # genesets_curated from tags
    rows = []
    if "MitoPathways" in t.columns:
        for _, r in t.dropna(subset=["MitoPathways"]).iterrows():
            g = r["gene"]
            tags = re.split(r";|,|\||\t|\s{2,}", str(r["MitoPathways"]).lower())
            if any("fusion" in s for s in tags): rows.append({"pathway":"fusion","gene":g})
            if any("fission" in s for s in tags): rows.append({"pathway":"fission","gene":g})
            if any(("mitophagy" in s) or ("autophagy" in s and "mito" in s) for s in tags):
                rows.append({"pathway":"mitophagy","gene":g})
            if any(("biogen" in s) or ("organization" in s) or ("import" in s) or ("tfam" in s) for s in tags):
                rows.append({"pathway":"biogenesis","gene":g})
    gs = pd.DataFrame(rows, columns=["pathway","gene"]).dropna()
    if gs.empty:
        gs = pd.DataFrame(columns=["pathway","gene"])
    gs["pathway"] = gs["pathway"].astype(str).str.lower()
    gs["gene"] = gs["gene"].astype(str).str.upper()
    gs = gs.drop_duplicates().sort_values(["pathway","gene"]).reset_index(drop=True)
    gs.to_csv(outdir / "genesets_curated.csv", index=False)
    log(f"wrote genesets_curated.csv ({len(gs)} rows)")

    # ev_whitelist.csv (seed/merge)
    wl_seed = {
        "TOMM20","VDAC1","ATP5F1A","ATP5F1B","ATP5MC1","SDHA","COX4I1",
        "NDUFS1","NDUFS2","NDUFA9","MFN2","OPA1","DNM1L","PHB2","PINK1","PRKN","TFAM","SLC25A3"
    }
    wl_path = outdir / "ev_whitelist.csv"
    if wl_path.exists():
        try:
            prev = pd.read_csv(wl_path)["protein"].astype(str).str.upper().tolist()
            wl_seed |= set(prev)
        except Exception:
            pass
    pd.DataFrame(sorted(wl_seed), columns=["protein"]).to_csv(wl_path, index=False)
    log(f"wrote ev_whitelist.csv ({len(wl_seed)} proteins)")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--xls", required=True, help="Path to Human.MitoCarta3.0.xls/.xlsx")
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    xls = Path(args.xls)
    if not xls.exists(): die(f"File not found: {xls}")
    outdir = Path(args.outdir); ensure_outdir(outdir)

    xfile = pd.ExcelFile(xls)

    log(">> Scanning all sheets and header rows …")
    best_t: Optional[pd.DataFrame] = None
    best_sheet: Optional[str] = None
    best_hits: int = -1

    for sname in xfile.sheet_names:
        try:
            # 1) Try as-is (headered)
            df_h = pd.read_excel(xls, sheet_name=sname)
            t = extract_trimmed(df_h)
            if t is not None and len(t) > 0:
                hits = len(t)
                log(f"   - sheet '{sname}': found 'Symbol' header with {hits} genes")
                if hits > best_hits:
                    best_t, best_sheet, best_hits = t, sname, hits
                continue

            # 2) Try headerless: find a header row containing 'Symbol', then re-read with header=row
            df_r = pd.read_excel(xls, sheet_name=sname, header=None)
            header_row = find_header_row(df_r, max_scan_rows=30)
            if header_row is not None:
                df_hdr = pd.read_excel(xls, sheet_name=sname, header=header_row)
                t2 = extract_trimmed(df_hdr)
                if t2 is not None and len(t2) > 0:
                    hits = len(t2)
                    log(f"   - sheet '{sname}': detected header at row {header_row} → {hits} genes")
                    if hits > best_hits:
                        best_t, best_sheet, best_hits = t2, sname, hits
                    continue

            # 3) As last resort: pick gene column by content
            gi, hits = score_gene_col_by_content(df_r, start_row=1)
            if gi is not None and hits >= 20:
                gene_series = df_r.iloc[1:, gi].astype(str).str.strip().str.upper()
                t3 = pd.DataFrame({"gene": gene_series})
                t3 = t3[t3["gene"].map(lambda x: bool(GENE_TOKEN_RE.match(x)))]
                t3 = t3.drop_duplicates(subset=["gene"]).reset_index(drop=True)
                if len(t3) > 0:
                    log(f"   - sheet '{sname}': content-detected gene col (hits={len(t3)})")
                    if len(t3) > best_hits:
                        best_t, best_sheet, best_hits = t3, sname, len(t3)

        except Exception as e:
            log(f"   ! sheet '{sname}' skipped due to error: {e}")

    if best_t is None or len(best_t) == 0:
        die("Could not find a usable sheet/column. Open the XLS, export the main table as CSV with headers, then rerun on the CSV.")

    log(f">> Selected sheet: {best_sheet}  (genes: {best_hits})")
    build_assets_from_trimmed(best_t, outdir)
    log("DONE.")

if __name__ == "__main__":
    main()
