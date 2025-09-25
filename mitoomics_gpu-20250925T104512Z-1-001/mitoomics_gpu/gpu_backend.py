#!/usr/bin/env python3
"""
gpu_backend.py
A small shim that gives you GPU types (cupy/cudf/cuML) when available, with
automatic CPU fallbacks (numpy/pandas/sklearn) when not.
"""
from __future__ import annotations



#!/usr/bin/env python3
"""
gpu_backend.py
Unified GPU/CPU backend with a test-friendly override:
  - Set MITOOMICS_FORCE_CPU=1 to force CPU even if CUDA is present.
"""
import os

class GB:
    has_gpu = False
    xp = None
    pd = None
    np = None
    sk = None
    cuml = None

_FORCE_CPU = os.environ.get("MITOOMICS_FORCE_CPU", "0") == "1"

try:
    if not _FORCE_CPU:
        import cupy as _cupy
        import cudf as _cudf
        import cuml
        GB.has_gpu = True
        GB.xp = _cupy
        GB.pd = _cudf
        GB.cuml = cuml
except Exception:
    GB.has_gpu = False

import numpy as _numpy
import pandas as _pandas
from sklearn import decomposition as _sk_decomp

GB.np = _numpy
GB.sk = _sk_decomp
if GB.xp is None: GB.xp = _numpy
if GB.pd is None: GB.pd = _pandas

def to_xp(a):
    if GB.has_gpu:
        return GB.xp.asarray(a)
    return GB.np.asarray(a)

def to_cpu(a):
    if GB.has_gpu and hasattr(a, "get"):
        return a.get()
    return GB.np.asarray(a)

def df_to_gpu(df):
    if GB.has_gpu and not isinstance(df, GB.pd.DataFrame):
        return GB.pd.from_pandas(df)
    return df

def df_to_cpu(df):
    try:
        return df.to_pandas() if GB.has_gpu else df
    except Exception:
        return df


import os

class GB:
    """Global backend with unified types + helpers."""
    has_gpu = False
    xp = None          # cupy or numpy
    pd = None          # cudf or pandas (df API subset)
    np = None          # numpy (always available for small ops)
    sk = None          # sklearn
    cuml = None        # cuML or None

# Try GPU first
try:
    import cupy as _cupy     # arrays on GPU
    import cudf as _cudf     # dataframes on GPU
    import cuml              # GPU ML (PCA/UMAP/KNN, etc.)
    GB.has_gpu = True
    GB.xp = _cupy
    GB.pd = _cudf
    GB.cuml = cuml
except Exception:
    GB.has_gpu = False

# CPU libs
import numpy as _numpy
import pandas as _pandas
from sklearn import decomposition as _sk_decomp
from sklearn.neighbors import NearestNeighbors as _SK_NN

GB.np = _numpy
GB.sk = _sk_decomp

# Ensure xp/pd fallback defined
if GB.xp is None:
    GB.xp = _numpy
if GB.pd is None:
    GB.pd = _pandas

def to_xp(a):
    """Move array-like to xp (cupy on GPU, numpy on CPU)."""
    if GB.has_gpu:
        return GB.xp.asarray(a)
    else:
        return GB.np.asarray(a)

def to_cpu(a):
    """Bring xp array back to CPU numpy."""
    if GB.has_gpu and hasattr(a, "get"):
        return a.get()
    return GB.np.asarray(a)

def df_to_gpu(df):
    if GB.has_gpu and not isinstance(df, GB.pd.DataFrame):
        return GB.pd.from_pandas(df)
    return df

def df_to_cpu(df):
    try:
        return df.to_pandas() if GB.has_gpu else df
    except Exception:
        return df  # already pandas
