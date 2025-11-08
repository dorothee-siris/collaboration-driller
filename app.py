# app.py  --  Page 1: Overview
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LIB_DIR = BASE_DIR / "lib"

if str(LIB_DIR) not in sys.path:
    sys.path.append(str(LIB_DIR))

st.set_page_config(
    page_title="UPCité Collaborations – Overview",
    layout="wide",
    page_icon="🌐",
)

st.title("Université Paris Cité – Collaborations overview")

# ---------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------
@st.cache_data
def load_core() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "upcite_core.parquet")


@st.cache_data
def load_partners() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "upcite_partners.parquet")


core_df = load_core()
partners_df = load_partners()

TOP_THRESHOLD = 20
top_partners_df = partners_df[
    partners_df["Count of co-publications"] >= TOP_THRESHOLD
].copy()

# ---------------------------------------------------------------------
# Topline metrics
# ---------------------------------------------------------------------
st.subheader("Topline metrics")

total_pubs = len(core_df)
intl_flag = core_df["is_international"].astype(str).str.upper().eq("TRUE")
intl_pubs = int(intl_flag.sum())
intl_share = intl_pubs / total_pubs if total_pubs else 0.0

n_top_partners = len(top_partners_df)

col1, col2, col3 = st.columns(3)
col1.metric("Total UPCité publications (2020–24)", f"{total_pubs:,}")
col2.metric(
    "International publications",
    f"{intl_pubs:,}",
    f"{intl_share*100:,.1f}%",
)
col3.metric(
    f"Top partners (≥ {TOP_THRESHOLD} co-pubs)",
    f"{n_top_partners:,}",
)

st.markdown("---")

# ---------------------------------------------------------------------
# Top partners table
# ---------------------------------------------------------------------
st.subheader("Top partners – overview table")

filter_choice = st.radio(
    "Filter by geography",
    ["All partners", "France only", "International only"],
    horizontal=True,
)

df_tbl = top_partners_df.copy()
if filter_choice == "France only":
    df_tbl = df_tbl[df_tbl["Partner country"] == "France"]
elif filter_choice == "International only":
    df_tbl = df_tbl[df_tbl["Partner country"] != "France"]

display_cols = [
    "Partner name",
    "Partner country",
    "Partner type",
    "Count of co-publications",
    "Partner's total output (2020-24)",
    "Share of UPCité's production",
    "Share of Partner's total production",
    "average FWCI",
]

df_tbl = df_tbl[display_cols].sort_values(
    "Count of co-publications", ascending=False
)

st.dataframe(
    df_tbl,
    use_container_width=True,
    column_config={
        "Share of UPCité's production": st.column_config.ProgressColumn(
            "Share of UPCité's production",
            format="%.3f",
            min_value=0.0,
            max_value=float(
                df_tbl["Share of UPCité's production"].max() or 0.001
            ),
        ),
        "Share of Partner's total production": st.column_config.ProgressColumn(
            "Share of Partner's total production",
            format="%.3f",
            min_value=0.0,
            max_value=float(
                df_tbl["Share of Partner's total production"].max() or 0.001
            ),
        ),
        "average FWCI": st.column_config.NumberColumn(
            "Average FWCI", format="%.2f"
        ),
    },
)