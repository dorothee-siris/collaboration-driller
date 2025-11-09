# pages/2_Geographic_focus.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from lib.taxonomy import (
    build_taxonomy_lookups,
    get_domain_color,
    get_domain_for_field,
    canonical_field_order,
)

# -------------------------------------------------------------------------
# Paths & data loading
# -------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_geo_data():
    df_country = pd.read_parquet(DATA_DIR / "upcite_country.parquet")
    df_partners = pd.read_parquet(DATA_DIR / "upcite_partners.parquet")
    return df_country, df_partners


df_country, df_partners = load_geo_data()
look = build_taxonomy_lookups()

DOMAINS: List[str] = [
    d for d in ["Health Sciences", "Life Sciences", "Physical Sciences", "Social Sciences"]
    if d in look["domain_order"]
]
DOMAIN_COLOR_MAP = {d: get_domain_color(d) for d in DOMAINS}
CANONICAL_FIELDS = canonical_field_order()


def _parse_pipe(s, cast=float):
    """Parse a 'a | b | c' string into a list with given cast."""
    if pd.isna(s):
        return []
    return [cast(x.strip()) for x in str(s).split("|") if str(x).strip() != ""]


# -------------------------------------------------------------------------
# Page title
# -------------------------------------------------------------------------

st.title("Geographic focus of collaborations")

st.markdown(
    """
This page explores where Université Paris Cité collaborates in the world and
how each country's portfolio looks in terms of disciplines.
"""
)

# -------------------------------------------------------------------------
# WORLD MAP SECTION
# -------------------------------------------------------------------------

st.markdown("## World map of co-publications by country")

metric_choice = st.selectbox(
    "Metric used for the colour scale:",
    [
        "Number of co-publications",
        "Number of partners",
        "Average FWCI",
    ],
)

metric_map = {
    "Number of co-publications": ("copubs", "Number of co-publications"),
    "Number of partners": ("num_partners", "Number of partners"),
    "Average FWCI": ("avg_fwci", "Average FWCI"),
}
metric_col, metric_label = metric_map[metric_choice]

min_copubs_all = int(df_country["copubs"].min())
max_copubs_all = int(df_country["copubs"].max())
default_thresh = 20 if min_copubs_all <= 20 <= max_copubs_all else min_copubs_all

min_copubs_filter = st.slider(
    "Minimum number of co-publications to include a country on the map",
    min_value=min_copubs_all,
    max_value=max_copubs_all,
    value=default_thresh,
    step=5,
)

df_map = df_country[df_country["copubs"] >= min_copubs_filter].copy()

if df_map.empty:
    st.info("There are no countries with at least the selected minimum number of co-publications.")
else:
    # Build domain breakdown text for hover (share vs country totals)
    def _domain_breakdown_txt(row):
        shares = _parse_pipe(row.get("domain_share_vs_country"), float)
        if not shares:
            return "n/a"
        pairs = list(zip(DOMAINS, shares[: len(DOMAINS)]))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        parts = [f"{int(round(s * 100))}% {name}" for name, s in pairs if s > 0]
        return ", ".join(parts) if parts else "n/a"

    df_map["domain_breakdown"] = df_map.apply(_domain_breakdown_txt, axis=1)

    # Custom data for hover
    customdata = np.stack(
        [
            df_map["copubs"].to_numpy(),
            df_map["share_vs_upcite_total"].to_numpy(),
            df_map["num_partners"].to_numpy(),
            df_map["avg_fwci"].to_numpy(),
            df_map["domain_breakdown"].to_numpy(),
        ],
        axis=-1,
    )

    fig_map = px.choropleth(
        df_map,
        locations="country",
        locationmode="country names",
        color=metric_col,
        color_continuous_scale="Reds",
        hover_name="country",
        labels={metric_col: metric_label},
    )

    fig_map.update_traces(
        customdata=customdata,
        hovertemplate=(
            "<b>%{location}</b><br>"
            "Co-publications: %{customdata[0]:,.0f}<br>"
            "Share vs UPCité total: %{customdata[1]:.1%}<br>"
            "Number of partners: %{customdata[2]:,.0f}<br>"
            "Average FWCI: %{customdata[3]:.2f}<br>"
            "Domain breakdown (share of this country's co-publications): %{customdata[4]}"
            "<extra></extra>"
        ),
    )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        coloraxis_colorbar=dict(title=metric_label),
    )

    st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------------------------------------------------
# COUNTRY DRILL-DOWN
# -------------------------------------------------------------------------

st.markdown("## Country drill-down")

all_countries = sorted(df_country["country"].dropna().unique().tolist())
default_index = all_countries.index("France") if "France" in all_countries else 0

selected_country = st.selectbox(
    "Select a country to explore in detail:",
    all_countries,
    index=default_index,
)

row_c = df_country.loc[df_country["country"] == selected_country]
if row_c.empty:
    st.warning("No aggregated data available for this country.")
    st.stop()

row_c = row_c.iloc[0]

# --- Topline metrics ---

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Co-publications with UPCité (2020–24)", f"{int(row_c['copubs']):,}")

with col2:
    st.metric(
        "Share of UPCité's production",
        f"{row_c['share_vs_upcite_total']*100:.2f}%",
    )

with col3:
    st.metric("Number of distinct partners", f"{int(row_c['num_partners']):,}")

with col4:
    st.metric("Average FWCI", f"{row_c['avg_fwci']:.2f}")

# --- Yearly breakdown ---

st.markdown("### Temporal profile")

years = list(range(2020, 2025))
year_counts = _parse_pipe(row_c["copubs_per_year"], int)
year_counts = year_counts[: len(years)]
df_year = pd.DataFrame({"year": years[: len(year_counts)], "copubs": year_counts})

fig_year = px.bar(
    df_year,
    x="year",
    y="copubs",
    labels={"copubs": "Number of co-publications", "year": "Year"},
)
fig_year.update_layout(margin=dict(l=0, r=0, t=20, b=0))
st.plotly_chart(fig_year, use_container_width=True)

# -------------------------------------------------------------------------
# Thematic profile for the selected country
# -------------------------------------------------------------------------

st.markdown("### Thematic profile of collaborations")

# Domain legend
legend_html = "<div style='margin-bottom:0.5rem;'>"
for d in DOMAINS:
    color = DOMAIN_COLOR_MAP[d]
    legend_html += (
        f"<span style='display:inline-block;width:12px;height:12px;"
        f"border-radius:50%;background-color:{color};margin-right:4px;'></span>"
        f"<span style='margin-right:12px;'>{d}</span>"
    )
legend_html += "</div>"

st.markdown(legend_html, unsafe_allow_html=True)

baseline_label = st.radio(
    "Baseline for percentage shares in the charts and table below:",
    [
        "Share of this country's co-publications with UPCité",
        "Share of UPCité's total production",
        "Share of UPCité's international collaborations",
    ],
    index=0,
)

if "country" in baseline_label:
    baseline = "country"
elif "international" in baseline_label:
    baseline = "intl"
else:
    baseline = "upcite"


def _domain_df(row, baseline_code: str) -> pd.DataFrame:
    counts = _parse_pipe(row["copubs_per_domain"], int)
    if baseline_code == "country":
        shares = _parse_pipe(row["domain_share_vs_country"], float)
    elif baseline_code == "intl":
        shares = _parse_pipe(row["domain_share_vs_intl"], float)
    else:
        shares = _parse_pipe(row["domain_share_vs_upcite"], float)
    fwcis = _parse_pipe(row["fwci_per_domain"], float)

    n = min(len(DOMAINS), len(counts), len(shares), len(fwcis))
    df = pd.DataFrame(
        {
            "domain": DOMAINS[:n],
            "count": counts[:n],
            "share": shares[:n],
            "fwci": fwcis[:n],
        }
    )
    df = df[df["count"] > 0].copy()
    df["share_pct"] = df["share"] * 100
    df.sort_values("share_pct", ascending=True, inplace=True)
    return df


def _field_df(row, baseline_code: str) -> pd.DataFrame:
    counts = _parse_pipe(row["copubs_per_field"], int)

    if baseline_code == "country":
        shares_col = "field_share_vs_country"
    elif baseline_code == "intl":
        shares_col = "field_share_vs_intl"
    else:
        shares_col = "field_share_vs_upcite"

    shares = _parse_pipe(row[shares_col], float)
    fwcis = _parse_pipe(row["fwci_per_field"], float)

    n = min(len(CANONICAL_FIELDS), len(counts), len(shares), len(fwcis))
    fields = CANONICAL_FIELDS[:n]

    df = pd.DataFrame(
        {
            "field": fields,
            "count": counts[:n],
            "share": shares[:n],
            "fwci": fwcis[:n],
        }
    )
    df = df[df["count"] > 0].copy()
    df["share_pct"] = df["share"] * 100
    df["domain"] = [get_domain_for_field(f) for f in df["field"]]
    df.sort_values("share_pct", ascending=True, inplace=True)
    return df


def _subfield_df(row, baseline_code: str) -> pd.DataFrame:
    records = []
    name2id = look["name2id"]
    subfields_by_field = look["subfields_by_field"]

    for field in CANONICAL_FIELDS:
        fid = name2id.get(field)
        if fid is None:
            continue

        count_col = f'Copubs per subfield within "{field}" (id: {fid})'
        if count_col not in row.index:
            continue

        if baseline_code == "country":
            share_col = f'Subfield share vs country within "{field}" (id: {fid})'
        elif baseline_code == "intl":
            share_col = f'Subfield share vs intl within "{field}" (id: {fid})'
        else:
            share_col = f'Subfield share vs UPCité within "{field}" (id: {fid})'

        fwci_col = f'FWCI per subfield within "{field}" (id: {fid})'

        counts = _parse_pipe(row[count_col], int)
        shares = _parse_pipe(row.get(share_col, ""), float)
        fwcis = _parse_pipe(row.get(fwci_col, ""), float)

        subfields = subfields_by_field.get(field, [])
        n = min(len(counts), len(shares), len(fwcis), len(subfields))
        domain = get_domain_for_field(field)

        for i in range(n):
            c = counts[i]
            if c <= 0:
                continue
            records.append(
                {
                    "domain": domain,
                    "field": field,
                    "subfield": subfields[i],
                    "count": c,
                    "share": shares[i],
                    "fwci": fwcis[i],
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["share_pct"] = df["share"] * 100
    df.sort_values(["share_pct", "count"], ascending=[False, False], inplace=True)
    return df


df_dom = _domain_df(row_c, baseline)
df_field = _field_df(row_c, baseline)
df_sub = _subfield_df(row_c, baseline)

# --- Domain & field charts ------------------------------------------------

col_dom, col_field = st.columns(2)

with col_dom:
    st.markdown("#### Breakdown by domain")
    if df_dom.empty:
        st.info("No domain-level data for this country.")
    else:
        customdata_dom = df_dom[["count", "fwci"]].to_numpy()
        fig_dom = px.bar(
            df_dom,
            x="share_pct",
            y="domain",
            orientation="h",
            color="domain",
            color_discrete_map=DOMAIN_COLOR_MAP,
            labels={"share_pct": "Share of co-publications (%)", "domain": ""},
            text="count",
        )
        fig_dom.update_traces(
            texttemplate="%{text:,d}",
            textposition="outside",
            customdata=customdata_dom,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Share: %{x:.1f}%<br>"
                "Co-publications: %{customdata[0]:,.0f}<br>"
                "Average FWCI: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )
        fig_dom.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_dom, use_container_width=True)

with col_field:
    st.markdown("#### Breakdown by field")
    if df_field.empty:
        st.info("No field-level data for this country.")
    else:
        customdata_f = df_field[["count", "fwci"]].to_numpy()
        fig_field = px.bar(
            df_field,
            x="share_pct",
            y="field",
            orientation="h",
            color="domain",
            color_discrete_map=DOMAIN_COLOR_MAP,
            labels={"share_pct": "Share of co-publications (%)", "field": ""},
            text="count",
        )
        fig_field.update_traces(
            texttemplate="%{text:,d}",
            textposition="outside",
            customdata=customdata_f,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Share: %{x:.1f}%<br>"
                "Co-publications: %{customdata[0]:,.0f}<br>"
                "Average FWCI: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )
        fig_field.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            legend_title_text="Domain",
        )
        st.plotly_chart(fig_field, use_container_width=True)

# --- Subfield table -------------------------------------------------------

st.markdown("#### Breakdown by subfield")

if df_sub.empty:
    st.info("No subfield-level data for this country.")
else:
    display_cols = ["domain", "field", "subfield", "share_pct", "count", "fwci"]
    df_sub_display = df_sub[display_cols].rename(
        columns={
            "domain": "Domain",
            "field": "Field",
            "subfield": "Subfield",
            "share_pct": "Share (%)",
            "count": "Co-publications",
            "fwci": "FWCI",
        }
    )

    st.dataframe(
        df_sub_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Share (%)": st.column_config.NumberColumn(
                "Share (%)",
                min_value=0.0,
                max_value=float(df_sub_display["Share (%)"].max() or 0.0),
                format="%.0f%%",
            ),
            "Co-publications": st.column_config.NumberColumn(
                "Co-publications", format="%d"
            ),
            "FWCI": st.column_config.NumberColumn("FWCI", format="%.2f"),
        },
    )

# -------------------------------------------------------------------------
# Partner table for selected country
# -------------------------------------------------------------------------

st.markdown("### Partner institutions in this country")

df_country_partners = df_partners[df_partners["Partner country"] == selected_country].copy()

if df_country_partners.empty:
    st.info("No partner institutions found for this country in the dataset.")
else:
    table_cols = [
        "Partner name",
        "Partner type",
        "Count of co-publications",
        "Share of UPCité's production",
        "Share of Partner's total production",
        "average FWCI",
    ]
    df_pt = df_country_partners[table_cols].sort_values(
        "Count of co-publications", ascending=False
    )

    st.dataframe(
        df_pt,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Share of UPCité's production": st.column_config.NumberColumn(
                "Share of UPCité's production",
                min_value=0.0,
                max_value=1.0,
                format="%.2f%%",
            ),
            "Share of Partner's total production": st.column_config.NumberColumn(
                "Share of partner's total production",
                min_value=0.0,
                max_value=1.0,
                format="%.2f%%",
            ),
            "average FWCI": st.column_config.NumberColumn(
                "Average FWCI", format="%.2f"
            ),
        },
    )
