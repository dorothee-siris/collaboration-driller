# pages/2_Geographic_focus.py

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.taxonomy import (
    build_taxonomy_lookups,
    canonical_field_order,
    get_field_color,
    get_domain_color,
    get_domain_for_field,
)

# -------------------------------------------------------------------------
# Page / paths / constants
# -------------------------------------------------------------------------

st.set_page_config(layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# --- Taxonomy lookups (domain order, colours, fields, subfields) ---

_TAX = build_taxonomy_lookups()

# Use canonical domain order, but drop "Other" for the 4 macro-domains
DOMAINS = [d for d in _TAX["domain_order"] if d != "Other"]

# colour map for those domains, coming from taxonomy's palette
DOMAIN_COLORS = {d: get_domain_color(d) for d in DOMAINS}

# Emojis to visually mark domains in tables (including Other for subfields)
DOMAIN_EMOJI = {
    "Health Sciences": "🟥",
    "Life Sciences": "🟩",
    "Physical Sciences": "🟦",
    "Social Sciences": "🟨",
    "Other": "⬜",
}

# canonical field list, grouped by domain, taken from all_topics.parquet
CANONICAL_FIELDS = canonical_field_order()

# -------------------------------------------------------------------------
# Data loading helpers
# -------------------------------------------------------------------------


@st.cache_data
def load_geo_data():
    df_country = pd.read_parquet(DATA_DIR / "upcite_country.parquet")
    df_partners = pd.read_parquet(DATA_DIR / "upcite_partners.parquet")
    return df_country, df_partners


@st.cache_data
def load_subfields_by_field():
    """
    Build a mapping: field name -> ordered list of subfield names,
    using all_topics.parquet. We sort first by field, then by subfield_id
    if present, otherwise by subfield_name.
    """
    df_topics = pd.read_parquet(DATA_DIR / "all_topics.parquet")

    sort_cols = []
    if "field_name" in df_topics.columns:
        sort_cols.append("field_name")
    if "subfield_id" in df_topics.columns:
        sort_cols.append("subfield_id")
    elif "subfield_name" in df_topics.columns:
        sort_cols.append("subfield_name")

    df_sf = (
        df_topics.sort_values(sort_cols)[["field_name", "subfield_name"]]
        .drop_duplicates()
        .dropna()
    )

    mapping = {}
    for f, grp in df_sf.groupby("field_name"):
        mapping[f] = grp["subfield_name"].tolist()
    return mapping


SUBFIELDS_BY_FIELD = load_subfields_by_field()
df_country, df_partners = load_geo_data()


def parse_pipe(s, cast=float):
    """
    Parse a 'a | b | c' string into [cast(a), cast(b), cast(c)].
    Handles NaN/None/"None" robustly.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    txt = str(s).strip()
    if not txt or txt.lower() == "none":
        return []
    parts = [p.strip() for p in txt.split("|") if p.strip()]
    out = []
    for p in parts:
        if p.lower() == "none":
            continue
        try:
            out.append(cast(p))
        except Exception:
            # silently skip malformed values
            continue
    return out


def parse_year_domain_counts(raw: str) -> pd.DataFrame:
    """
    Parse strings like:
    '2020 (126 ; 33 ; 175 ; 374) | 2021 (104 ; 46 ; 201 ; 373) | ...'
    into columns: year, domain, copubs.
    Domain order is DOMAINS.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return pd.DataFrame(columns=["year", "domain", "copubs"])

    text = str(raw)
    parts = [p.strip() for p in text.split("|") if p.strip()]
    records = []

    for part in parts:
        # Extract year and inside of parentheses
        m = re.match(r"(\d{4})\s*\(([^)]*)\)", part)
        if not m:
            continue
        year = int(m.group(1))
        nums = [n.strip() for n in m.group(2).split(";") if n.strip()]
        nums = [int(n) for n in nums]
        n = min(len(nums), len(DOMAINS))
        for i in range(n):
            records.append(
                {"year": year, "domain": DOMAINS[i], "copubs": nums[i]}
            )

    return pd.DataFrame(records)


# ---------- helpers to parse "a | b | c" strings into numeric lists ----------


def _parse_pipe_numbers(s, as_float=False):
    if pd.isna(s):
        return []
    vals = []
    for tok in str(s).split("|"):
        tok = tok.strip()
        if not tok:
            vals.append(0.0 if as_float else 0)
        else:
            if as_float:
                vals.append(float(tok.replace(",", ".")))
            else:
                vals.append(int(tok))
    return vals


def build_country_field_df(country_row: pd.Series) -> pd.DataFrame:
    """
    Build a DataFrame with one row per field, in canonical taxonomy order,
    with count & share columns for this country.

    - share_vs_country: vs this country's total co-publications (0–1)
    - share_vs_intl:    vs all UPCité international co-publications (0–1)
    """
    field_names = CANONICAL_FIELDS
    n_fields = len(field_names)

    counts = _parse_pipe_numbers(country_row.get("copubs_per_field", ""), as_float=False)
    shares_country = _parse_pipe_numbers(
        country_row.get("field_share_vs_country", ""), as_float=True
    )
    shares_intl = _parse_pipe_numbers(
        country_row.get("field_share_vs_intl", ""), as_float=True
    )
    fwci_vals = _parse_pipe_numbers(country_row.get("fwci_per_field", ""), as_float=True)

    def _pad(lst, pad_value=0.0):
        lst = list(lst)
        if len(lst) < n_fields:
            lst += [pad_value] * (n_fields - len(lst))
        return lst[:n_fields]

    counts = _pad(counts, 0)
    shares_country = _pad(shares_country, 0.0)
    shares_intl = _pad(shares_intl, 0.0)
    fwci_vals = _pad(fwci_vals, 0.0)

    df = pd.DataFrame(
        {
            "Field": field_names,
            "count": counts,
            "share_vs_country": shares_country,
            "share_vs_intl": shares_intl,
            "fwci": fwci_vals,
        }
    )

    # Add domain + colour for each field
    df["domain"] = df["Field"].apply(get_domain_for_field)
    df["color"] = df["Field"].apply(get_field_color)

    return df


# -------------------------------------------------------------------------
# Title and intro
# -------------------------------------------------------------------------

st.title("Geographic focus of collaborations")

st.markdown(
    """
This page explores **where** Université Paris Cité collaborates in the world and
what the **thematic profile** of each partner country looks like.
"""
)

# -------------------------------------------------------------------------
# WORLD MAP SECTION
# -------------------------------------------------------------------------

st.markdown("## World map of co-publications by country")

metric_choice = st.selectbox(
    "Metric used for the colour scale:",
    ["Number of co-publications", "Number of partners", "Average FWCI"],
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
    st.info(
        "There are no countries with at least the selected minimum number of "
        "co-publications."
    )
else:

    def domain_breakdown_txt(row):
        """Text like '44% Life Sciences, 31% Physical Sciences, ...'."""
        shares = parse_pipe(row.get("domain_share_vs_country"), float)
        if not shares:
            return "n/a"
        pairs = list(zip(DOMAINS, shares[: len(DOMAINS)]))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        parts = [f"{int(round(s * 100))}% {name}" for name, s in pairs if s > 0]
        return ", ".join(parts) if parts else "n/a"

    df_map["domain_breakdown"] = df_map.apply(domain_breakdown_txt, axis=1)

    customdata = np.stack(
        [
            df_map["copubs"].to_numpy(),
            df_map["share_vs_upcite_intl"].to_numpy(),
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
            "Share of UPCité's international collaborations: %{customdata[1]:.1%}<br>"
            "Number of partners: %{customdata[2]:,.0f}<br>"
            "Average FWCI: %{customdata[3]:.2f}<br>"
            "Domain breakdown (share of this country's co-publications): %{customdata[4]}"
            "<extra></extra>"
        ),
    )

    fig_map.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        coloraxis_colorbar=dict(title=metric_label),
        height=650,
    )

    st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------------------------------------------------
# COUNTRY DRILL-DOWN
# -------------------------------------------------------------------------

st.markdown("## Country drill-down")

country_options = sorted(df_country["country"].unique())
selected_country = st.selectbox(
    "Select a country",
    country_options,
    index=None,
    placeholder="Choose a country to explore",
)

if selected_country is None:
    st.info("Select a country above to see detailed indicators for one country.")
    st.stop()

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
        "Share of UPCité's international collaborations",
        f"{row_c['share_vs_upcite_intl'] * 100:.2f}%",
    )

with col3:
    st.metric("Number of distinct partners", f"{int(row_c['num_partners']):,}")

with col4:
    st.metric("Average FWCI", f"{row_c['avg_fwci']:.2f}")

# --- Domain legend just below metrics ---

legend_html = "<div style='margin: 0.8rem 0 0.4rem 0;'>"
for d in DOMAINS:
    color = DOMAIN_COLORS[d]
    legend_html += (
        f"<span style='display:inline-block;width:12px;height:12px;"
        f"border-radius:50%;background-color:{color};margin-right:4px;'></span>"
        f"<span style='margin-right:14px;'>{d}</span>"
    )
legend_html += "</div>"

st.markdown(legend_html, unsafe_allow_html=True)

# --- Yearly breakdown by domain (stacked bar) ---

st.markdown("### Temporal profile by domain")

df_year_dom = parse_year_domain_counts(row_c.get("copubs_per_year_domain"))
df_year_dom = df_year_dom[df_year_dom["copubs"] > 0]

if df_year_dom.empty:
    st.info("No yearly/domain breakdown is available for this country.")
else:
    fig_year_dom = px.bar(
        df_year_dom,
        x="year",
        y="copubs",
        color="domain",
        color_discrete_map=DOMAIN_COLORS,
        barmode="stack",
        labels={
            "year": "Year",
            "copubs": "Number of co-publications",
            "domain": "",
        },
    )
    fig_year_dom.update_layout(
        margin=dict(l=0, r=0, t=25, b=0),
        showlegend=False,
    )
    st.plotly_chart(fig_year_dom, use_container_width=True)

# -------------------------------------------------------------------------
# Thematic profile (fields, subfields)
# -------------------------------------------------------------------------

st.markdown("### Thematic profile of collaborations")

# --- Field-level breakdown (two baselines: vs country & vs intl) ----------

df_fields = build_country_field_df(row_c)
df_fields["share_country_pct"] = df_fields["share_vs_country"] * 100.0
df_fields["share_intl_pct"] = df_fields["share_vs_intl"] * 100.0

col_fc, col_fi = st.columns(2)

with col_fc:
    st.markdown("#### By field – share of this country's co-publications")
    if df_fields.empty:
        st.info("No field-level data for this country.")
    else:
        customdata_fc = df_fields[["count", "fwci"]].to_numpy()
        fig_fc = go.Figure()
        fig_fc.add_bar(
            x=df_fields["share_country_pct"],
            y=df_fields["Field"],
            orientation="h",
            marker=dict(color=df_fields["color"]),
            customdata=customdata_fc,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Share (vs this country's collaborations): %{x:.1f}%<br>"
                "Co-publications: %{customdata[0]:,.0f}<br>"
                "Average FWCI: %{customdata[1]:.2f}<extra></extra>"
            ),
            text=df_fields["count"],
            textposition="outside",
        )
        fig_fc.update_layout(
            margin=dict(l=0, r=0, t=25, b=0),
            xaxis_title="Share (%)",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig_fc, use_container_width=True)

with col_fi:
    st.markdown("#### By field – share of UPCité’s international collaborations")
    if df_fields.empty:
        st.info("No field-level data for this country.")
    else:
        customdata_fi = df_fields[["count", "fwci"]].to_numpy()
        fig_fi = go.Figure()
        fig_fi.add_bar(
            x=df_fields["share_intl_pct"],
            y=df_fields["Field"],
            orientation="h",
            marker=dict(color=df_fields["color"]),
            customdata=customdata_fi,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Share (vs all UPCité international co-publications in this field): "
                "%{x:.1f}%<br>"
                "Co-publications with this country: %{customdata[0]:,.0f}<br>"
                "Average FWCI: %{customdata[1]:.2f}<extra></extra>"
            ),
            text=df_fields["count"],
            textposition="outside",
        )
        fig_fi.update_layout(
            margin=dict(l=0, r=0, t=25, b=0),
            xaxis_title="Share (%)",
            yaxis_title="",
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

# --- Subfield table (two baselines, progress bars) ------------------------


def make_subfield_df(row) -> pd.DataFrame:
    records = []

    for field in CANONICAL_FIELDS:
        base = f'within "{field}"'
        count_col = next(
            (c for c in row.index if c.startswith("Copubs per subfield") and base in c),
            None,
        )
        share_intl_col = next(
            (c for c in row.index if c.startswith("Subfield share vs intl") and base in c),
            None,
        )
        share_country_col = next(
            (c for c in row.index if c.startswith("Subfield share vs country") and base in c),
            None,
        )
        fwci_col = next(
            (c for c in row.index if c.startswith("FWCI per subfield") and base in c),
            None,
        )

        if not count_col or not share_intl_col or not share_country_col or not fwci_col:
            continue

        counts = parse_pipe(row[count_col], int)
        shares_intl = parse_pipe(row[share_intl_col], float)
        shares_country = parse_pipe(row[share_country_col], float)
        fwcis = parse_pipe(row[fwci_col], float)

        subfields = SUBFIELDS_BY_FIELD.get(field, [])
        n = min(len(counts), len(shares_intl), len(shares_country), len(fwcis), len(subfields))
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
                    "share_intl": shares_intl[i],
                    "share_country": shares_country[i],
                    "fwci": fwcis[i],
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["share_intl_pct"] = df["share_intl"] * 100.0
    df["share_country_pct"] = df["share_country"] * 100.0
    # Order by largest share vs country first, then count
    df.sort_values(
        ["share_country_pct", "count"], ascending=[False, False], inplace=True
    )
    return df


st.markdown("#### Breakdown by subfield")

df_sub = make_subfield_df(row_c)

if df_sub.empty:
    st.info("No subfield-level data for this country.")
else:
    # Add a domain marker with emojis
    df_sub_display = df_sub[
        [
            "domain",
            "field",
            "subfield",
            "share_country_pct",
            "share_intl_pct",
            "count",
            "fwci",
        ]
    ].copy()

    df_sub_display["Domain"] = df_sub_display["domain"].apply(
        lambda d: f"{DOMAIN_EMOJI.get(d, '⬜')} {d}"
    )

    df_sub_display = df_sub_display.rename(
        columns={
            "field": "Field",
            "subfield": "Subfield",
            "share_country_pct": "Share vs country (%)",
            "share_intl_pct": "Share vs all UPCité intl (%)",
            "count": "Co-publications",
            "fwci": "FWCI",
        }
    )

    # Reorder columns with Domain first
    df_sub_display = df_sub_display[
        [
            "Domain",
            "Field",
            "Subfield",
            "Share vs country (%)",
            "Share vs all UPCité intl (%)",
            "Co-publications",
            "FWCI",
        ]
    ]

    st.dataframe(
        df_sub_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Share vs country (%)": st.column_config.ProgressColumn(
                "Share vs country (%)",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%",  # 1 decimal
                help="Share of this country's co-publications with UPCité in this subfield.",
            ),
            "Share vs all UPCité intl (%)": st.column_config.ProgressColumn(
                "Share vs all UPCité intl (%)",
                min_value=0.0,
                max_value=100.0,
                format="%.1f%%",  # 1 decimal
                help=(
                    "Share of all UPCité international co-publications in this subfield "
                    "that involve this country."
                ),
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
    # Filter by partner type
    partner_types = sorted(df_country_partners["Partner type"].dropna().unique())
    selected_types = st.multiselect(
        "Filter by partner type",
        options=partner_types,
        default=partner_types,
    )

    if selected_types:
        df_country_partners = df_country_partners[
            df_country_partners["Partner type"].isin(selected_types)
        ]

    if df_country_partners.empty:
        st.info("No partner institutions match the selected type(s).")
    else:
        # Prepare table + convert shares to % for nicer display
        base_cols = [
            "Partner name",
            "Partner type",
            "Count of co-publications",
            "Share of UPCité's production",
            "Share of Partner's total production",
            "average FWCI",
        ]
        df_pt = df_country_partners[base_cols].copy()

        df_pt["Share of UPCité's production (%)"] = (
            df_pt["Share of UPCité's production"] * 100.0
        )
        df_pt["Share of Partner's total production (%)"] = (
            df_pt["Share of Partner's total production"] * 100.0
        )

        df_pt = df_pt.sort_values(
            "Count of co-publications", ascending=False
        )

        df_pt_display = df_pt[
            [
                "Partner name",
                "Partner type",
                "Count of co-publications",
                "Share of UPCité's production (%)",
                "Share of Partner's total production (%)",
                "average FWCI",
            ]
        ]

        st.dataframe(
            df_pt_display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Share of UPCité's production (%)": st.column_config.ProgressColumn(
                    "Share of UPCité's production",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.2f%%",
                ),
                "Share of Partner's total production (%)": st.column_config.ProgressColumn(
                    "Share of Partner's total production",
                    min_value=0.0,
                    max_value=100.0,
                    format="%.2f%%",
                ),
                "average FWCI": st.column_config.NumberColumn(
                    "Average FWCI", format="%.2f"
                ),
            },
        )
