from __future__ import annotations
import pandas as pd

def ev_pathway_scores(ev_df: pd.DataFrame,
                      pathways: dict[str, list[str]],
                      whitelist: set[str] | None = None) -> pd.DataFrame:
    """
    Aggregate EV/MDV proteomics to subject-level pathway scores:
      - Optionally filter to whitelist proteins (SYMBOLS uppercased)
      - Map proteins -> pathways via MitoCarta-derived pathway dict
      - Sum normalized abundance per (subject, pathway)
    """
    if ev_df is None or ev_df.empty:
        return pd.DataFrame(columns=["subject_id"])

    df = ev_df.copy()
    df["protein_upper"] = df["protein"].astype(str).str.upper()

    if whitelist:
        df = df[df["protein_upper"].isin(whitelist)]
        if df.empty:
            return pd.DataFrame({"subject_id": ev_df["subject_id"].unique()})

    # Build reverse map: protein -> set(pathways)
    prot2paths = {}
    for pth, syms in pathways.items():
        for s in syms:
            su = str(s).upper()
            prot2paths.setdefault(su, set()).add(pth)

    long = []
    for _, r in df.iterrows():
        prot = r["protein_upper"]
        subs = r["subject_id"]
        val = r["abundance_norm"]
        for pth in prot2paths.get(prot, []):
            long.append((subs, pth, val))

    if not long:
        # No overlaps — return zeros per subject for all pathways
        subs = df["subject_id"].unique()
        return pd.DataFrame({"subject_id": subs})

    cat = pd.DataFrame(long, columns=["subject_id","pathway","value"])
    piv = cat.pivot_table(index="subject_id", columns="pathway", values="value",
                          aggfunc="sum", fill_value=0.0)
    piv = piv.reset_index()
    piv.columns.name = None
    # normalize columns names to lowercase (to match scoring)
    piv.rename(columns={c: c.lower() for c in piv.columns if c != "subject_id"}, inplace=True)
    return piv
