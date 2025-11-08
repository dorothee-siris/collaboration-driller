# pages/2_Geographic_focus.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------
# Paths & data loaders
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # go up from /pages
DATA_DIR = BASE_DIR / "data"
LIB_DIR = BASE_DIR / "lib"

if str(LIB_DIR) not in sys.path:
    sys.path.append(str(LIB_DIR))


@st.cache_data
def load_partners() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "upcite_partners.parquet")


st.title("Geographic focus")

partners_df = load_partners()

# ---------------------------------------------------------------------
# Choropleth: co-pubs by country (France excluded)
# ---------------------------------------------------------------------
country_agg = (
    partners_df.groupby("Partner country", as_index=False)[
        "Count of co-publications"
    ]
    .sum()
    .rename(columns={"Count of co-publications": "Co-publications"})
)

# Exclude France to avoid visual dominance
country_agg = country_agg[country_agg["Partner country"] != "France"]

if country_agg.empty:
    st.info("No international partners found in the dataset.")
else:
    st.subheader("Global co-publications map (France excluded)")

    fig_map = px.choropleth(
        country_agg,
        locations="Partner country",
        locationmode="country names",
        color="Co-publications",
        color_continuous_scale="Reds",
        hover_name="Partner country",
        title="UPCité co-publications by partner country (2020–24)",
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))

    st.plotly_chart(fig_map, use_container_width=True)

    # -----------------------------------------------------------------
    # Drill-down: partners in one country
    # -----------------------------------------------------------------
    st.subheader("Partners by country")

    selected_country = st.selectbox(
        "Select a country to list its partners:",
        sorted(country_agg["Partner country"].unique()),
    )

    sub = partners_df[partners_df["Partner country"] == selected_country].copy()
    sub = sub.sort_values("Count of co-publications", ascending=False)

    cols = [
        "Partner name",
        "Partner country",
        "Partner type",
        "Count of co-publications",
        "average FWCI",
        "Share of UPCité's production",
        "Share of Partner's total production",
    ]

    st.dataframe(
        sub[cols],
        use_container_width=True,
        column_config={
            "Share of UPCité's production": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "Share of Partner's total production": st.column_config.NumberColumn(
                format="%.3f"
            ),
            "average FWCI": st.column_config.NumberColumn(format="%.2f"),
        },
    )
