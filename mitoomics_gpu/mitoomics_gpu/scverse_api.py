#!/usr/bin/env python3
from __future__ import annotations
"""
scverse_api.py — program_score shim that prefers GPU (CuPy) and falls back to CPU.
"""
from typing import Dict, List, Optional
import pandas as pd
from .rapids_program_score import score_programs, _HAVE_GPU

def program_score(
    adata,
    gene_sets: Dict[str, List[str]],
    layer: Optional[str] = None,
    use_gpu: bool = True,
    libnorm: Optional[str] = "CPM",
    log1p: bool = True,
    standardize: str = "z",
    batch_cols: Optional[List[str]] = None,
    into_obs_prefix: Optional[str] = None,
):
    df = score_programs(
        adata=adata,
        gene_sets=gene_sets,
        layer=layer,
        libnorm=libnorm,
        log1p=log1p,
        standardize=standardize,
        return_df=True,
        use_gpu=use_gpu and _HAVE_GPU,
        batch_cols=batch_cols,
    )
    if into_obs_prefix:
        for c in df.columns:
            adata.obs[f"{into_obs_prefix}{c}"] = df[c].values
    return df
