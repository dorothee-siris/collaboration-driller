# app.py  --  Page 1: Overview
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
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
    page_title="Overview – UPCité Collaborations",
    layout="wide",
    page_icon="🌐",
)

st.title("Overview")
st.caption("Université Paris Cité – collaborations with external partners (2020–24)")

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
# Filters for table + bubble chart
# ---------------------------------------------------------------------
st.subheader("Top partners – overview table")

filter_col1, filter_col2 = st.columns([1, 2])

with filter_col1:
    geo_filter = st.radio(
        "Filter by geography",
        ["All partners", "France only", "International only"],
        horizontal=False,
    )

with filter_col2:
    all_types = (
        top_partners_df["Partner type"]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )
    selected_types = st.multiselect(
        "Filter by partner type",
        options=all_types,
        default=all_types,  # all selected by default
    )

# Apply filters
df_filtered = top_partners_df.copy()

# geography filter
if geo_filter == "France only":
    df_filtered = df_filtered[df_filtered["Partner country"] == "France"]
elif geo_filter == "International only":
    df_filtered = df_filtered[df_filtered["Partner country"] != "France"]

# type filter
if selected_types:
    df_filtered = df_filtered[df_filtered["Partner type"].astype(str).isin(selected_types)]
else:
    # if nothing selected, show empty
    df_filtered = df_filtered.iloc[0:0]

# ---------------------------------------------------------------------
# Table (hide Partner's total output column)
# ---------------------------------------------------------------------
table_cols = [
    "Partner name",
    "Partner country",
    "Partner type",
    "Count of co-publications",
    "Share of UPCité's production",
    "Share of Partner's total production",
    "average FWCI",
]

df_table = df_filtered[table_cols].sort_values(
    "Count of co-publications", ascending=False
)

st.dataframe(
    df_table,
    use_container_width=True,
    column_config={
        "Share of UPCité's production": st.column_config.ProgressColumn(
            "Share of UPCité's production",
            format="%.3f",
            min_value=0.0,
            max_value=float(
                df_table["Share of UPCité's production"].max() or 0.001
            ),
        ),
        "Share of Partner's total production": st.column_config.ProgressColumn(
            "Share of Partner's total production",
            format="%.3f",
            min_value=0.0,
            max_value=float(
                df_table["Share of Partner's total production"].max() or 0.001
            ),
        ),
        "average FWCI": st.column_config.NumberColumn(
            "Average FWCI", format="%.2f"
        ),
    },
)

# ---------------------------------------------------------------------
# Bubble chart: strategic weights (global view)
# ---------------------------------------------------------------------
st.markdown("### Strategic weight of partners (global view)")

if df_filtered.empty:
    st.info("No partners match the current filters.")
else:
    scatter_df = df_filtered.copy()

    # Ensure no NaN in x/y
    scatter_df["x"] = scatter_df["Share of UPCité's production"].fillna(0.0)
    scatter_df["y"] = scatter_df["Share of Partner's total production"].fillna(0.0)

    # Top 100 by co-publications
    scatter_df = scatter_df.sort_values(
        "Count of co-publications", ascending=False
    ).head(100)

    # Color: blue for France, red for international
    def country_color(country: str) -> str:
        return "blue" if country == "France" else "red"

    scatter_df["color"] = scatter_df["Partner country"].apply(country_color)

    max_xy = float(
        max(scatter_df["x"].max(), scatter_df["y"].max()) * 1.05 or 0.01
    )

    fig = px.scatter(
        scatter_df,
        x="x",
        y="y",
        size="Count of co-publications",
        color="Partner country",  # legend will still show FR vs others
        hover_name="Partner name",
        hover_data={
            "Count of co-publications": True,
            "Partner type": True,
            "Partner country": True,
            "x": False,
            "y": False,
        },
        labels={
            "x": "Share of UPCité's production",
            "y": "Share of partner's total production",
        },
    )

    # Override marker colors to blue/red
    # (France may be 1 or more traces depending on how px groups, but
    #  easiest is to map in update_traces by name)
    for trace in fig.data:
        # trace.name is the country label
        for i, trace in enumerate(fig.data):
            if trace.name == "France":
                fig.data[i].marker.update(color="blue")
            else:
                fig.data[i].marker.update(color="red")

    # Diagonal line x = y
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_xy,
        y1=max_xy,
        line=dict(color="gray", dash="dash"),
    )
    fig.update_xaxes(range=[0, max_xy])
    fig.update_yaxes(range=[0, max_xy])

    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        legend_title="Partner country",
    )

    st.plotly_chart(fig, use_container_width=True)
