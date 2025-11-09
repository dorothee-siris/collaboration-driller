# pages/2_Geographic_focus.py

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------------------------------------------------
# Page / paths / constants
# -------------------------------------------------------------------------

# (Only the first call across the app has an effect, but it's safe here.)
st.set_page_config(layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# 4 macro-domains and their colours (used consistently across charts)
DOMAINS = ["Health Sciences", "Life Sciences", "Physical Sciences", "Social Sciences"]
DOMAIN_COLORS = {
    "Health Sciences": "#e41a1c",    # red
    "Life Sciences": "#4daf4a",      # green
    "Physical Sciences": "#377eb8",  # blue
    "Social Sciences": "#984ea3",    # purple
}

# Canonical field list, ordered by (domain, field)
CANONICAL_FIELDS = [
    # Life Sciences
    "Agricultural and Biological Sciences",
    "Biochemistry, Genetics and Molecular Biology",
    "Immunology and Microbiology",
    # Health Sciences
    "Medicine",
    "Neuroscience",
    "Nursing",
    "Pharmacology, Toxicology and Pharmaceutics",
    "Veterinary",
    "Dentistry",
    "Health Professions",
    # Physical Sciences
    "Chemical Engineering",
    "Chemistry",
    "Computer Science",
    "Earth and Planetary Sciences",
    "Energy",
    "Engineering",
    "Environmental Science",
    "Materials Science",
    "Mathematics",
    "Physics and Astronomy",
    # Social Sciences
    "Arts and Humanities",
    "Business, Management and Accounting",
    "Decision Sciences",
    "Economics, Econometrics and Finance",
    "Psychology",
    "Social Sciences",
]

FIELD_TO_DOMAIN = {
    "Agricultural and Biological Sciences": "Life Sciences",
    "Biochemistry, Genetics and Molecular Biology": "Life Sciences",
    "Immunology and Microbiology": "Life Sciences",
    "Medicine": "Health Sciences",
    "Neuroscience": "Health Sciences",
    "Nursing": "Health Sciences",
    "Pharmacology, Toxicology and Pharmaceutics": "Health Sciences",
    "Veterinary": "Health Sciences",
    "Dentistry": "Health Sciences",
    "Health Professions": "Health Sciences",
    "Chemical Engineering": "Physical Sciences",
    "Chemistry": "Physical Sciences",
    "Computer Science": "Physical Sciences",
    "Earth and Planetary Sciences": "Physical Sciences",
    "Energy": "Physical Sciences",
    "Engineering": "Physical Sciences",
    "Environmental Science": "Physical Sciences",
    "Materials Science": "Physical Sciences",
    "Mathematics": "Physical Sciences",
    "Physics and Astronomy": "Physical Sciences",
    "Arts and Humanities": "Social Sciences",
    "Business, Management and Accounting": "Social Sciences",
    "Decision Sciences": "Social Sciences",
    "Economics, Econometrics and Finance": "Social Sciences",
    "Psychology": "Social Sciences",
    "Social Sciences": "Social Sciences",
}


def field_domain(field: str) -> str:
    return FIELD_TO_DOMAIN.get(field, "Other")


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
        height=550,
    )

    st.plotly_chart(fig_map, use_container_width=True)

# -------------------------------------------------------------------------
# COUNTRY DRILL-DOWN
# -------------------------------------------------------------------------

st.markdown("## Country drill-down")

all_countries = sorted(df_country["country"].dropna().unique().tolist())
default_index = all_countries.index("France") if "France" in all_countries else 0

selected_country = st.selectbox(
    "Select a country to explore in detail:", all_countries, index=default_index
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
        legend_title_text="",
    )
    st.plotly_chart(fig_year_dom, use_container_width=True)

# -------------------------------------------------------------------------
# Thematic profile (domains, fields, subfields)
# -------------------------------------------------------------------------

st.markdown("### Thematic profile of collaborations")

# --- Domain-level breakdown (share vs country totals) ---------------------


def make_domain_df(row) -> pd.DataFrame:
    counts = parse_pipe(row["copubs_per_domain"], int)
    shares_country = parse_pipe(row["domain_share_vs_country"], float)
    fwcis = parse_pipe(row["fwci_per_domain"], float)

    n = min(len(DOMAINS), len(counts), len(shares_country), len(fwcis))
    df = pd.DataFrame(
        {
            "domain": DOMAINS[:n],
            "count": counts[:n],
            "share_country": shares_country[:n],
            "fwci": fwcis[:n],
        }
    )
    df = df[df["count"] > 0].copy()
    df["share_country_pct"] = df["share_country"] * 100
    df.sort_values("share_country_pct", ascending=True, inplace=True)
    return df


df_dom = make_domain_df(row_c)

col_dom, col_fields = st.columns([1, 2])

with col_dom:
    st.markdown("#### Breakdown by domain (share of this country's co-publications)")
    if df_dom.empty:
        st.info("No domain-level data for this country.")
    else:
        customdata_dom = df_dom[["count", "fwci"]].to_numpy()
        fig_dom = px.bar(
            df_dom,
            x="share_country_pct",
            y="domain",
            orientation="h",
            color="domain",
            color_discrete_map=DOMAIN_COLORS,
            labels={"share_country_pct": "Share (%)", "domain": ""},
            text="count",
        )
        fig_dom.update_traces(
            texttemplate="%{text:,d}",
            textposition="outside",
            customdata=customdata_dom,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Share (vs this country's collaborations): %{x:.1f}%<br>"
                "Co-publications: %{customdata[0]:,.0f}<br>"
                "Average FWCI: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )
        fig_dom.update_layout(
            margin=dict(l=0, r=0, t=25, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig_dom, use_container_width=True)

# --- Field-level breakdown (two baselines: vs country & vs intl) ----------


def make_field_df(row) -> pd.DataFrame:
    counts = parse_pipe(row["copubs_per_field"], int)
    shares_intl = parse_pipe(row["field_share_vs_intl"], float)
    shares_country = parse_pipe(row["field_share_vs_country"], float)
    fwcis = parse_pipe(row["fwci_per_field"], float)

    n = min(
        len(CANONICAL_FIELDS), len(counts), len(shares_intl), len(shares_country), len(fwcis)
    )
    df = pd.DataFrame(
        {
            "field": CANONICAL_FIELDS[:n],
            "count": counts[:n],
            "share_intl": shares_intl[:n],
            "share_country": shares_country[:n],
            "fwci": fwcis[:n],
        }
    )
    df = df[df["count"] > 0].copy()
    df["domain"] = [field_domain(f) for f in df["field"]]
    df["share_intl_pct"] = df["share_intl"] * 100
    df["share_country_pct"] = df["share_country"] * 100

    # Keep canonical order (already given by CANONICAL_FIELDS)
    df["order"] = [CANONICAL_FIELDS.index(f) for f in df["field"]]
    df.sort_values("order", inplace=True)
    df.drop(columns=["order"], inplace=True)
    return df


df_field = make_field_df(row_c)

col_fc, col_fi = st.columns(2)

with col_fc:
    st.markdown("#### By field – share of this country's co-publications")
    if df_field.empty:
        st.info("No field-level data for this country.")
    else:
        customdata_fc = df_field[["count", "fwci"]].to_numpy()
        fig_fc = px.bar(
            df_field,
            x="share_country_pct",
            y="field",
            orientation="h",
            color="domain",
            color_discrete_map=DOMAIN_COLORS,
            labels={"share_country_pct": "Share (%)", "field": ""},
            text="count",
        )
        fig_fc.update_traces(
            texttemplate="%{text:,d}",
            textposition="outside",
            customdata=customdata_fc,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Share (vs this country's collaborations): %{x:.1f}%<br>"
                "Co-publications: %{customdata[0]:,.0f}<br>"
                "Average FWCI: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )
        fig_fc.update_layout(
            margin=dict(l=0, r=0, t=25, b=0),
            legend_title_text="Domain",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

with col_fi:
    st.markdown("#### By field – share of UPCité’s international collaborations")
    if df_field.empty:
        st.info("No field-level data for this country.")
    else:
        customdata_fi = df_field[["count", "fwci"]].to_numpy()
        fig_fi = px.bar(
            df_field,
            x="share_intl_pct",
            y="field",
            orientation="h",
            color="domain",
            color_discrete_map=DOMAIN_COLORS,
            labels={"share_intl_pct": "Share (%)", "field": ""},
            text="count",
        )
        fig_fi.update_traces(
            texttemplate="%{text:,d}",
            textposition="outside",
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
        fig_fi.update_layout(
            margin=dict(l=0, r=0, t=25, b=0),
            legend_title_text="Domain",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

# --- Subfield table (two baselines, progress bars) ------------------------


def make_subfield_df(row) -> pd.DataFrame:
    records = []

    for field in CANONICAL_FIELDS:
        # Field ID appears in the column labels as '(id: XX)'; we don't need id,
        # we just reuse the exact column names that exist in upcite_country.
        # Example:
        #   Copubs per subfield within "Agricultural and Biological Sciences" (id: 11)
        #   Subfield share vs intl within "Agricultural and Biological Sciences" (id: 11)
        #   Subfield share vs country within "Agricultural and Biological Sciences" (id: 11)
        #   FWCI per subfield within "Agricultural and Biological Sciences" (id: 11)
        #
        # We'll look for any column starting with these patterns.
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
        domain = field_domain(field)

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
    df["share_intl_pct"] = df["share_intl"] * 100
    df["share_country_pct"] = df["share_country"] * 100
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
    ].rename(
        columns={
            "domain": "Domain",
            "field": "Field",
            "subfield": "Subfield",
            "share_country_pct": "Share vs country (%)",
            "share_intl_pct": "Share vs all UPCité intl (%)",
            "count": "Co-publications",
            "fwci": "FWCI",
        }
    )

    st.dataframe(
        df_sub_display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Share vs country (%)": st.column_config.ProgressColumn(
                "Share vs country (%)",
                min_value=0.0,
                max_value=100.0,
                format="%.0f%%",
                help="Share of this country's co-publications with UPCité in this subfield.",
            ),
            "Share vs all UPCité intl (%)": st.column_config.ProgressColumn(
                "Share vs all UPCité intl (%)",
                min_value=0.0,
                max_value=100.0,
                format="%.0f%%",
                help="Share of all UPCité international co-publications in this subfield "
                "that involve this country.",
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
            "Share of UPCité's production": st.column_config.ProgressColumn(
                "Share of UPCité's production",
                min_value=0.0,
                max_value=1.0,
                format="%.2f%%",
            ),
            "Share of Partner's total production": st.column_config.ProgressColumn(
                "Share of Partner's total production",
                min_value=0.0,
                max_value=1.0,
                format="%.2f%%",
            ),
            "average FWCI": st.column_config.NumberColumn(
                "Average FWCI", format="%.2f"
            ),
        },
    )
