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

# Order used in the pre-aggregated "per domain" numeric series
# (domain 1..4 in upcite_country): Life / Social / Physical / Health
DOMAIN_SERIES_ORDER = [
    "Life Sciences",
    "Social Sciences",
    "Physical Sciences",
    "Health Sciences",
]

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

# Total UPCité publications over 2020–24 (all outputs, not only intl. copubs)
UPCITE_TOTAL_PUBLICATIONS = 89069

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

    The numeric series are ordered by domain ID 1..4, which correspond to:
    Life Sciences / Social Sciences / Physical Sciences / Health Sciences
    (see DOMAIN_SERIES_ORDER).
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return pd.DataFrame(columns=["year", "domain", "copubs"])

    text = str(raw)
    parts = [p.strip() for p in text.split("|") if p.strip()]
    records = []

    for part in parts:
        m = re.match(r"(\d{4})\s*\(([^)]*)\)", part)
        if not m:
            continue
        year = int(m.group(1))
        nums = [n.strip() for n in m.group(2).split(";") if n.strip()]
        nums = [int(n) for n in nums]

        n_domains = min(len(nums), len(DOMAIN_SERIES_ORDER))
        for i in range(n_domains):
            records.append(
                {
                    "year": year,
                    "domain": DOMAIN_SERIES_ORDER[i],
                    "copubs": nums[i],
                }
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

    The series in upcite_country are stored by field ID in ascending order
    (11, 12, ..., 36). We map those IDs to field names, then reindex to the
    canonical field order used in the taxonomy.
    """
    # Parse numeric series from the row
    counts = _parse_pipe_numbers(country_row.get("copubs_per_field", ""), as_float=False)
    shares_country = _parse_pipe_numbers(
        country_row.get("field_share_vs_country", ""), as_float=True
    )
    shares_intl = _parse_pipe_numbers(
        country_row.get("field_share_vs_intl", ""), as_float=True
    )
    fwci_vals = _parse_pipe_numbers(country_row.get("fwci_per_field", ""), as_float=True)

    n = len(counts)

    # Pad other arrays to the same length, just in case
    def _pad_to_n(lst, pad_value=0.0):
        lst = list(lst)
        if len(lst) < n:
            lst += [pad_value] * (n - len(lst))
        return lst[:n]

    shares_country = _pad_to_n(shares_country, 0.0)
    shares_intl = _pad_to_n(shares_intl, 0.0)
    fwci_vals = _pad_to_n(fwci_vals, 0.0)

    # Values are in ascending field ID order from 11 upward
    field_ids = list(range(11, 11 + n))
    fid2name = _TAX["field_id_to_name"]
    field_names_from_ids = [fid2name.get(str(fid), f"Field {fid}") for fid in field_ids]

    # Raw df for existing fields
    df_raw = pd.DataFrame(
        {
            "Field": field_names_from_ids,
            "count": counts,
            "share_vs_country": shares_country,
            "share_vs_intl": shares_intl,
            "fwci": fwci_vals,
        }
    )

    # Reindex onto the canonical field list: one row per canonical field,
    # filled with zeros where this country has no co-publications.
    df = (
        df_raw.set_index("Field")
        .reindex(CANONICAL_FIELDS, fill_value=0)
        .reset_index()
        .rename(columns={"index": "Field"})
    )

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

color_scale_map = {
    "copubs": "Greens",        # co-publications → green
    "num_partners": "Blues",   # partners → blue
    "avg_fwci": "Reds",        # FWCI → red
}
color_scale = color_scale_map.get(metric_col, "Reds")

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

        # Map the 4 numeric values to the correct domains (Life / Social / Physical / Health)
        pairs = list(zip(DOMAIN_SERIES_ORDER, shares[: len(DOMAIN_SERIES_ORDER)]))
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
        color_continuous_scale=color_scale,
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
            "Domain breakdown: %{customdata[4]}"
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
# COUNTRY FOCUS
# -------------------------------------------------------------------------

st.markdown("## Country Focus")

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

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Co-publications with UPCité (2020–24)", f"{int(row_c['copubs']):,}")

# NEW metric: share vs *all* UPCité publications
share_vs_total = row_c["copubs"] / UPCITE_TOTAL_PUBLICATIONS if UPCITE_TOTAL_PUBLICATIONS else 0.0
with col2:
    st.metric(
        "Share of UPCité's total output",
        f"{share_vs_total * 100:.2f}%",
    )

with col3:
    st.metric(
        "Share of UPCité's international collaborations",
        f"{row_c['share_vs_upcite_intl'] * 100:.2f}%",
    )

with col4:
    st.metric("Number of distinct partners", f"{int(row_c['num_partners']):,}")

with col5:
    st.metric("Average FWCI", f"{row_c['avg_fwci']:.2f}")


st.markdown("### Yearly distribution by domain")

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
        height=300,
    )
    st.plotly_chart(fig_year_dom, use_container_width=True)

# -------------------------------------------------------------------------
# Thematic profile (fields, subfields)
# -------------------------------------------------------------------------

st.markdown("### Thematic profile by field and subfield")

st.markdown(
    """
**How to read the percentages below**

For each **field** and **subfield**, two different shares are shown:

- **Share vs this country's co-publications**  
  Among *all* co-publications between Université Paris Cité and the selected country,  
  what proportion belongs to this field or subfield?  
  → This tells you how important a topic is **within the bilateral collaboration** with this country.

- **Share vs all UPCité international co-publications in this category**  
  Among *all* international co-publications of UPCité in the same field or subfield  
  (with any country), what proportion involves this country?  
  → This tells you how important the **country is for UPCité in that topic**, compared with all other international partners.

High values on the **first** metric highlight topics that are a **specialty of the relationship**.  
High values on the **second** metric highlight topics where the country is a **strategic partner for UPCité**.
"""
)

# --- Field-level breakdown (two baselines: vs country & vs intl) ----------

df_fields = build_country_field_df(row_c)
df_fields["share_country_pct"] = df_fields["share_vs_country"] * 100.0
df_fields["share_intl_pct"] = df_fields["share_vs_intl"] * 100.0

# Precompute max shares for optional common x-scale
if df_fields.empty:
    max_share_country = max_share_intl = 1.0
else:
    max_share_country = float(df_fields["share_country_pct"].max() or 0.0)
    if max_share_country <= 0:
        max_share_country = 1.0

    max_share_intl = float(df_fields["share_intl_pct"].max() or 0.0)
    if max_share_intl <= 0:
        max_share_intl = 1.0

# Toggle: use same horizontal scale for both field charts
lock_field_axes = st.toggle(
    "Use the same horizontal scale for both field charts",
    value=False,
)

shared_max = max(max_share_country, max_share_intl) if lock_field_axes else None

col_fc, col_fi = st.columns(2)

with col_fc:
    st.markdown(
        f"#### Distribution by field  \n"
        f"against all co-publications with {selected_country}"
    )
    if df_fields.empty:
        st.info("No field-level data for this country.")
    else:
        customdata_fc = df_fields[["count", "fwci"]].to_numpy()

        # Base bar chart (no text, we’ll add counts as separate annotations)
        fig_fc = px.bar(
            df_fields,
            x="share_country_pct",
            y="Field",  # <-- capital F
            orientation="h",
            color="domain",
            color_discrete_map=DOMAIN_COLORS,
            labels={"share_country_pct": "Share (%)", "Field": ""},
        )

        fig_fc.update_traces(
            customdata=customdata_fc,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Share (vs this country's collaborations): %{x:.1f}%<br>"
                "Co-publications: %{customdata[0]:,.0f}<br>"
                "Average FWCI: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )

        # --- Left gutter for counts + gridlines + fonts + % ticks ---

        # Use shared max if toggle is on, otherwise the country-specific max
        max_plot_c = shared_max if shared_max is not None else max_share_country
        gutter_c = max_plot_c * 0.20  # 20% of max as left gutter

        fig_fc.update_xaxes(
            range=[-gutter_c, max_plot_c * 1.05],
            showgrid=True,
            gridcolor="#e0e0e0",
            ticksuffix="%",
            tickfont=dict(size=12),
        )
        fig_fc.update_yaxes(tickfont=dict(size=13))

        # Counts in the gutter (between y-axis and x=0)
        for field_name, cnt in zip(df_fields["Field"], df_fields["count"]):
            fig_fc.add_annotation(
                x=-gutter_c * 0.98,
                y=field_name,
                text=f"{int(cnt)}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=12, color="#444"),
            )

        # Slightly taller chart
        fig_fc.update_layout(
            margin=dict(l=0, r=0, t=25, b=10),
            showlegend=False,
            height=700,  # ~50% taller than default
        )

        st.plotly_chart(fig_fc, use_container_width=True)

with col_fi:
    st.markdown(
        "#### Distribution by field  \n"
        "against all UPCité’s international co-publications"
    )
    if df_fields.empty:
        st.info("No field-level data for this country.")
    else:
        customdata_fi = df_fields[["count", "fwci"]].to_numpy()

        fig_fi = px.bar(
            df_fields,
            x="share_intl_pct",
            y="Field",
            orientation="h",
            color="domain",
            color_discrete_map=DOMAIN_COLORS,
            labels={"share_intl_pct": "Share (%)", "Field": ""},
        )

        fig_fi.update_traces(
            customdata=customdata_fi,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Share (vs all UPCité international co-publications in this field): "
                "%{x:.1f}%<br>"
                "Co-publications with this country: %{customdata[0]:,.0f}<br>"
                "Average FWCI: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )

        # Use shared max if toggle is on, otherwise the intl-specific max
        max_plot_i = shared_max if shared_max is not None else max_share_intl
        gutter_i = max_plot_i * 0.20

        fig_fi.update_xaxes(
            range=[-gutter_i, max_plot_i * 1.05],
            showgrid=True,
            gridcolor="#e0e0e0",
            ticksuffix="%",
            tickfont=dict(size=12),
        )
        fig_fi.update_yaxes(tickfont=dict(size=13))

        for field_name, cnt in zip(df_fields["Field"], df_fields["count"]):
            fig_fi.add_annotation(
                x=-gutter_i * 0.98,
                y=field_name,
                text=f"{int(cnt)}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=12, color="#444"),
            )

        fig_fi.update_layout(
            margin=dict(l=0, r=0, t=25, b=10),
            showlegend=False,
            height=700,
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

def make_partner_top5_subfields(row: pd.Series) -> str:
    """
    For a partner row, aggregate all subfields across all fields using the
    'Relative share per subfield ... vs Partner total' columns, and return a
    string with the top 5 subfields by share, formatted:
      'Subfield 1 (x.x%) | Subfield 2 (y.y%) | ...'
    """
    records = []

    for field in CANONICAL_FIELDS:
        base = f'within "{field}"'
        # Look for columns like:
        # 'Relative share per subfield within "Agricultural and Biological Sciences" (id: 11) vs Partner total'
        share_partner_col = next(
            (
                c
                for c in row.index
                if c.startswith("Relative share per subfield")
                and base in c
                and "vs Partner total" in c
            ),
            None,
        )
        if not share_partner_col:
            continue

        shares_partner = parse_pipe(row[share_partner_col], float)
        subfields = SUBFIELDS_BY_FIELD.get(field, [])
        n = min(len(shares_partner), len(subfields))

        for i in range(n):
            s = shares_partner[i]
            if s is None or s <= 0:
                continue
            records.append(
                {
                    "subfield": subfields[i],
                    "share_partner": s,
                }
            )

    if not records:
        return ""

    df = pd.DataFrame(records)
    df.sort_values("share_partner", ascending=False, inplace=True)
    top = df.head(5)

    return " | ".join(
        f"{r.subfield} ({r.share_partner * 100:.1f}%)"
        for _, r in top.iterrows()
    )


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
        "Filter by partner type (optional)",
        options=partner_types,
        default=[],  # no filter selected by default
    )

    if selected_types:
        df_country_partners = df_country_partners[
            df_country_partners["Partner type"].isin(selected_types)
        ]

    if df_country_partners.empty:
        st.info("No partner institutions match the selected type(s).")
    else:
        # Compute Top 5 subfields for each partner (using partner-level shares)
        df_country_partners["Top 5 subfields"] = df_country_partners.apply(
            make_partner_top5_subfields, axis=1
        )
        
        # Prepare table + convert shares to % for nicer display
        base_cols = [
            "Partner name",
            "Partner type",
            "Count of co-publications",
            "Share of UPCité's production",
            "Share of Partner's total production",
            "average FWCI",
            "Top 5 subfields",   # <-- add this line
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
                "Top 5 subfields",
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
