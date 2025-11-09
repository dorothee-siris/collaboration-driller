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
    "% of international publications",
    f"{intl_share*100:.2f}%",   # e.g. 50.92%
)
col3.metric(
    f"Top partners (≥ {TOP_THRESHOLD} co-pubs)",
    f"{n_top_partners:,}",
)

st.markdown("---")

# ---------------------------------------------------------------------
# Filters for table + bubble chart
# ---------------------------------------------------------------------
st.subheader("Top partners (>20 co-publications over 2020-24) – overview table")

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
    # Empty selection = no type filter (all types)
    selected_types = st.multiselect(
        "Filter by partner type (optional)",
        options=all_types,
        default=[],  # nothing selected -> all included
    )

# Apply filters
df_filtered = top_partners_df.copy()

# geography filter
if geo_filter == "France only":
    df_filtered = df_filtered[df_filtered["Partner country"] == "France"]
elif geo_filter == "International only":
    df_filtered = df_filtered[
        df_filtered["Partner country"].notna()
        & (df_filtered["Partner country"] != "France")
        & (df_filtered["Partner country"] != "None")
    ]

# type filter
if selected_types:
    df_filtered = df_filtered[df_filtered["Partner type"].astype(str).isin(selected_types)]

# ---------------------------------------------------------------------
# Table (hide Partner's total output column)
# ---------------------------------------------------------------------
if df_filtered.empty:
    st.info("There are no partners matching the selected criteria.")
else:
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

    # Create a display copy where shares are expressed in %
    df_table_display = df_table.copy()
    df_table_display["Share of UPCité's production"] *= 100
    df_table_display["Share of Partner's total production"] *= 100

    max_upcite = float(
        (df_table_display["Share of UPCité's production"].max() or 0.01)
    )
    max_partner = float(
        (df_table_display["Share of Partner's total production"].max() or 0.01)
    )

    st.dataframe(
        df_table_display,
        use_container_width=True,
        column_config={
            "Share of UPCité's production": st.column_config.ProgressColumn(
                "Share of UPCité's production",
                format="%.2f%%",  # show e.g. 3.45%
                min_value=0.0,
                max_value=max_upcite,
            ),
            "Share of Partner's total production": st.column_config.ProgressColumn(
                "Share of Partner's total production",
                format="%.2f%%",
                min_value=0.0,
                max_value=max_partner,
            ),
            "average FWCI": st.column_config.NumberColumn(
                "Average FWCI", format="%.2f"
            ),
        },
    )

    # -----------------------------------------------------------------
    # Bubble chart: strategic weights (global view)
    # -----------------------------------------------------------------
    st.markdown("### Strategic weight against top selected partners")

    # How many partners to display?
    max_available = min(100, len(df_filtered))
    n_top = st.slider(
        "Number of partners displayed in the chart",
        min_value=1,
        max_value=max_available if max_available > 0 else 1,
        value=max_available if max_available > 0 else 1,
        step=1,
    )

    # Compute minimum co-publications among partners that will be displayed
    sorted_tmp = df_filtered.sort_values(
        "Count of co-publications", ascending=False
    )
    top_tmp = sorted_tmp.head(n_top)

    if not top_tmp.empty:
        min_copubs = int(top_tmp["Count of co-publications"].min())
        st.markdown(
            f"_All top **{len(top_tmp)}** institutions displayed in the chart below "
            f"have at least **{min_copubs}** co-publications with "
            f"Université Paris Cité, given the selected criteria._"
        )

    # Custom HTML legend (blue / red / grey)
    st.markdown(
        """
        <div style="margin-bottom: 0.5rem;">
          <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background-color:blue;margin-right:4px;"></span>
          <span style="margin-right:12px;">France</span>
          <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background-color:red;margin-right:4px;"></span>
          <span style="margin-right:12px;">International</span>
          <span style="display:inline-block;width:12px;height:12px;border-radius:50%;background-color:#888;margin-right:4px;"></span>
          <span>No country information</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Start from the N partners already selected above
    scatter_df = top_tmp.copy()

    # x = share of partner's total production
    # y = share of UPCité's production
    scatter_df["x"] = scatter_df["Share of Partner's total production"].fillna(0.0)
    scatter_df["y"] = scatter_df["Share of UPCité's production"].fillna(0.0)
    scatter_df["average FWCI"] = scatter_df["average FWCI"].fillna(0.0)

    # Extra columns for hover (keep raw shares as decimals)
    scatter_df["share_upcite"] = scatter_df["Share of UPCité's production"].fillna(0.0)
    scatter_df["share_partner"] = scatter_df["Share of Partner's total production"].fillna(0.0)

    # Category for coloring: France / International / No country
    def geo_category(country: str) -> str:
        if country == "France":
            return "France"
        if country is None or pd.isna(country) or country == "None" or country == "":
            return "No country"
        return "International"

    scatter_df["Geo category"] = scatter_df["Partner country"].apply(geo_category)

    # Axis ranges slightly above max, independent for X and Y
    max_x = float(scatter_df["x"].max() or 0.0)
    max_y = float(scatter_df["y"].max() or 0.0)

    x_max = max_x + 0.05 if max_x > 0 else 0.05
    y_max = max_y + 0.05 if max_y > 0 else 0.05

    # optional cap at 1.0 if you prefer:
    # x_max = min(x_max, 1.0)
    # y_max = min(y_max, 1.0)

    # limit for the diagonal line x = y so it stays inside the frame
    diag_max = min(x_max, y_max)

    # Bubble size = partner's total output
    size_col = "Partner's total output (2020-24)"
    scatter_df[size_col] = scatter_df[size_col].fillna(0.0)

    fig = px.scatter(
        scatter_df,
        x="x",
        y="y",
        size=size_col,
        size_max=40,
        color="Geo category",
        color_discrete_map={
            "France": "blue",
            "International": "red",
            "No country": "#888888",
        },
        hover_name="Partner name",
        # custom_data order:
        # 0 country
        # 1 type
        # 2 co-pubs
        # 3 FWCI
        # 4 share_upcite (UPCité's share)
        # 5 share_partner (partner's share)
        # 6 partner total output
        custom_data=[
            "Partner country",
            "Partner type",
            "Count of co-publications",
            "average FWCI",
            "share_upcite",
            "share_partner",
            size_col,
        ],
        labels={
            "x": "Share of partner's total production",
            "y": "Share of UPCité's production",
        },
    )

    # Very thin black outline for bubbles
    fig.update_traces(
        marker=dict(
            line=dict(color="black", width=0.5)
        )
    )

    # Custom hovertemplate (shares in %, plus partner total)
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>"
            "country: %{customdata[0]}<br>"
            "type: %{customdata[1]}<br>"
            "co-publications with UPCité (2020–24): %{customdata[2]:,}<br>"
            "share of UPCité's total output: %{customdata[4]:.1%}<br>"
            "average FWCI: %{customdata[3]:.2f}<br>"
            "partner's totals (2020–24): %{customdata[6]:,}<br>"
            "share of partner's total output: %{customdata[5]:.1%}<extra></extra>"
        )
    )

    # Remove Plotly legend (we use the HTML legend)
    fig.update_layout(showlegend=False)

    # Diagonal line x = y (clipped to the smaller axis max)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=diag_max,
        y1=diag_max,
        line=dict(color="gray", dash="dash"),
    )

    # Axes: independent ranges, percent ticks, larger labels
    fig.update_xaxes(
        range=[0, x_max],
        tickformat=".0%",          # 0%, 10%, 20%, ...
        dtick=0.1,                 # 0.1 in data terms = 10%
        title_font=dict(size=15),
    )
    fig.update_yaxes(
        range=[0, y_max],
        tickformat=".0%",
        dtick=0.1,
        title_font=dict(size=15),
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)