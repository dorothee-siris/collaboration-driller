# lib/data_cache.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

@st.cache_resource
def get_core_df() -> pd.DataFrame:
    """All UPCité publications (40 MB parquet)."""
    return pd.read_parquet(DATA_DIR / "upcite_core.parquet")

@st.cache_resource
def get_partners_df() -> pd.DataFrame:
    """Full partners table (20 MB parquet) with light type tweaks."""
    df = pd.read_parquet(DATA_DIR / "upcite_partners.parquet")

    # Light memory-friendly tweaks that are useful everywhere
    for col in ["Partner name", "Partner country", "Partner type"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # This column is huge and not needed for any computation
    df.drop(columns=["Thematic words (top 500)"], errors="ignore", inplace=True)

    return df

@st.cache_resource
def get_country_df() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "upcite_country.parquet")
    # Make sure the main numeric cols are numeric (but don't upcast every time)
    for col in ["copubs", "num_partners", "avg_fwci"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_resource
def get_topics_df() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "all_topics.parquet")

@st.cache_resource
def get_lookup_df() -> pd.DataFrame:
    # if you need it
    return pd.read_parquet(DATA_DIR / "upcite_lookup.parquet")