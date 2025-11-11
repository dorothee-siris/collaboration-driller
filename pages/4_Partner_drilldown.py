# pages/4_Partner_drilldown.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.taxonomy import (
    build_taxonomy_lookups,
    canonical_field_order,
    get_domain_color,
    get_domain_for_field,
    get_field_color,
    get_subfield_color,
)
from lib.data_cache import get_partners_df, get_topics_df

from lib.debug_tools import render_debug_sidebar
render_debug_sidebar()


# ---------------------------------------------------------------------
# Paths & basic config
# ---------------------------------------------------------------------

# In a multipage app, it's best to call set_page_config only once in app.py.
# If you already do it there, comment out the next line.
# st.set_page_config(layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

# ---------------------------------------------------------------------
# Global taxonomy helpers
# ---------------------------------------------------------------------
_TAX = build_taxonomy_lookups()

# Canonical field order & subfields
CANONICAL_FIELDS = canonical_field_order()
SUBFIELDS_BY_FIELD: Dict[str, List[str]] = _TAX["subfields_by_field"]

# Field IDs → names (for mapping pipe-series which are in field-id order)
FIELD_ID_TO_NAME: Dict[int, str] = {
    int(fid): name for fid, name in _TAX["field_id_to_name"].items()
}
FIELD_IDS_BY_ID_ORDER: List[int] = sorted(FIELD_ID_TO_NAME.keys())
FIELD_NAMES_BY_ID: List[str] = [FIELD_ID_TO_NAME[i] for i in FIELD_IDS_BY_ID_ORDER]


@st.cache_resource
def get_domain_meta() -> Tuple[List[int], List[str]]:
    """Domain IDs and names in ascending domain_id order."""
    topics = get_topics_df()
    meta = (
        topics[["domain_id", "domain_name"]]
        .drop_duplicates()
        .sort_values("domain_id")
    )
    dom_ids = meta["domain_id"].astype(int).tolist()
    dom_names = meta["domain_name"].tolist()
    return dom_ids, dom_names


DOMAIN_IDS, DOMAIN_NAMES_BY_ID = get_domain_meta()
DOMAIN_COLORS = {name: get_domain_color(name) for name in DOMAIN_NAMES_BY_ID}
# Domain list in the same order as the partner's pipe-series (domain_id ascending)
DOMAINS = [d for d in DOMAIN_NAMES_BY_ID if d != "Other"]

# Emoji marker for domains (for tables)
DOMAIN_EMOJI = {
    "Health Sciences": "🟥",
    "Life Sciences": "🟩",
    "Physical Sciences": "🟦",
    "Social Sciences": "🟨",
    "Other": "⬜",
}

# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------

partners_df = get_partners_df()


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def parse_pipe_ints(s: Any) -> List[int]:
    if pd.isna(s) or s == "":
        return []
    vals: List[int] = []
    for p in str(s).split("|"):
        p = p.strip()
        if not p:
            vals.append(0)
        else:
            try:
                vals.append(int(p))
            except Exception:
                vals.append(0)
    return vals


def parse_pipe_floats(s: Any) -> List[float]:
    if pd.isna(s) or s == "":
        return []
    vals: List[float] = []
    for p in str(s).split("|"):
        p = p.strip()
        if not p:
            vals.append(0.0)
        else:
            try:
                vals.append(float(p.replace(",", ".")))
            except Exception:
                vals.append(0.0)
    return vals


def pad_to_length(lst: List[Any], n: int, pad_value: Any) -> List[Any]:
    lst = list(lst)
    if len(lst) < n:
        lst += [pad_value] * (n - len(lst))
    return lst[:n]


def parse_partner_year_domain_counts(raw: Any) -> pd.DataFrame:
    """
    Parse strings like:
      '2020 (1480 ; 936 ; 3603 ; 2122) | 2021 (...) | ...'
    into a long DataFrame with columns: year, domain, copubs.
    Numbers are ordered by domain_id ascending.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return pd.DataFrame(columns=["year", "domain", "copubs"])

    text = str(raw)
    parts = [p.strip() for p in text.split("|") if p.strip()]
    records: List[Dict[str, Any]] = []

    for part in parts:
        m = re.match(r"(\d{4})\s*\(([^)]*)\)", part)
        if not m:
            continue
        year = int(m.group(1))
        nums = [n.strip() for n in m.group(2).split(";") if n.strip()]
        nums_int = [int(n) for n in nums]

        n = min(len(nums_int), len(DOMAIN_NAMES_BY_ID))
        for i in range(n):
            records.append(
                {
                    "year": year,
                    "domain": DOMAIN_NAMES_BY_ID[i],
                    "copubs": nums_int[i],
                }
            )

    return pd.DataFrame(records)


def build_partner_field_df(row: pd.Series) -> pd.DataFrame:
    """
    One row per field in canonical order, with:
      - count: co-publications in this field with the partner
      - fwci: field-level FWCI of those co-publications
      - share_with_partner: share of all co-publications with this partner
      - rel_vs_upcite: Relative share per field vs UPCité total
      - rel_vs_partner: Relative share per field vs Partner total
    """
    counts = parse_pipe_ints(row.get("Copubs per field", ""))
    fwci_vals = parse_pipe_floats(row.get("FWCI per field", ""))
    rel_vs_upcite = parse_pipe_floats(row.get("Relative share per field vs UPCité total", ""))
    rel_vs_partner = parse_pipe_floats(row.get("Relative share per field vs Partner total", ""))

    n = len(counts)
    fwci_vals = pad_to_length(fwci_vals, n, 0.0)
    rel_vs_upcite = pad_to_length(rel_vs_upcite, n, 0.0)
    rel_vs_partner = pad_to_length(rel_vs_partner, n, 0.0)

    # Partner series follow ascending field_id order (11..36)
    field_ids = FIELD_IDS_BY_ID_ORDER[:n]
    field_names = [FIELD_ID_TO_NAME[i] for i in field_ids]

    total_copubs = float(row.get("Count of co-publications", 0) or 0)
    if total_copubs <= 0:
        # Avoid division by zero; if no co-pubs, shares are 0
        share_with_partner = [0.0] * n
    else:
        share_with_partner = [c / total_copubs for c in counts]

    df_raw = pd.DataFrame(
        {
            "Field": field_names,
            "count": counts,
            "fwci": fwci_vals,
            "share_with_partner": share_with_partner,
            "rel_vs_upcite": rel_vs_upcite,
            "rel_vs_partner": rel_vs_partner,
        }
    )

    # Reindex onto canonical field order so charts align across partners
    df = (
        df_raw.set_index("Field")
        .reindex(CANONICAL_FIELDS, fill_value=0)
        .reset_index()
        .rename(columns={"index": "Field"})
    )

    df["share_with_partner_pct"] = df["share_with_partner"] * 100.0
    df["rel_vs_upcite_pct"] = df["rel_vs_upcite"] * 100.0
    df["rel_vs_partner_pct"] = df["rel_vs_partner"] * 100.0
    df["domain"] = df["Field"].apply(get_domain_for_field)
    df["color"] = df["Field"].apply(get_field_color)
    return df


def make_partner_subfield_df(row: pd.Series) -> pd.DataFrame:
    """
    Build a long table of subfields for this partner with:
      - subfield co-pubs
      - share within this partner's co-publications
      - relative share vs UPCité total (subfield)
      - relative share vs partner total (subfield)
      - partner's total pubs in that subfield
      - FWCI
    """
    total_copubs = float(row.get("Count of co-publications", 0) or 0)
    records: List[Dict[str, Any]] = []

    for field in CANONICAL_FIELDS:
        base = f'within "{field}"'

        count_col = next(
            (c for c in row.index if c.startswith("Copubs per subfield") and base in c),
            None,
        )
        fwci_col = next(
            (c for c in row.index if c.startswith("FWCI per subfield") and base in c),
            None,
        )
        share_upcite_col = next(
            (
                c
                for c in row.index
                if c.startswith("Relative share per subfield")
                and base in c
                and "vs Partner total" not in c
            ),
            None,
        )
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
        partner_total_col = next(
            (
                c
                for c in row.index
                if c.startswith("Pubs per subfield")
                and base in c
                and "(partner total)" in c
            ),
            None,
        )

        if not count_col or not fwci_col or not share_upcite_col or not share_partner_col:
            continue

        counts = parse_pipe_ints(row[count_col])
        fwcis = parse_pipe_floats(row[fwci_col])
        shares_upcite = parse_pipe_floats(row[share_upcite_col])
        shares_partner = parse_pipe_floats(row[share_partner_col])
        partner_totals = (
            parse_pipe_ints(row[partner_total_col]) if partner_total_col in row.index else []
        )

        subfields = SUBFIELDS_BY_FIELD.get(field, [])
        n = min(
            len(counts),
            len(fwcis),
            len(shares_upcite),
            len(shares_partner),
            len(subfields),
        )
        partner_totals = pad_to_length(partner_totals, n, 0)

        domain = get_domain_for_field(field)

        for i in range(n):
            c = counts[i]
            if c <= 0:
                continue
            s_mix = (c / total_copubs) if total_copubs > 0 else 0.0
            records.append(
                {
                    "Domain": domain,
                    "Field": field,
                    "Subfield": subfields[i],
                    "copubs": c,
                    "share_mix": s_mix,
                    "share_vs_partner_total": shares_partner[i],
                    "share_vs_upcite_total": shares_upcite[i],
                    "partner_total_subfield": partner_totals[i],
                    "fwci": fwcis[i],
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Sort: biggest share within co-publications, then count
    df.sort_values(["share_mix", "copubs"], ascending=[False, False], inplace=True)
    return df


# ---------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------
st.title("Partner drilldown")

# 1) Search & selection ------------------------------------------------
search = st.text_input("Search partner by name (min. 2 characters)", "")
partner_names = partners_df["Partner name"].dropna().unique()

if search and len(search.strip()) >= 2:
    partner_names = [n for n in partner_names if search.lower() in n.lower()]

if len(partner_names) == 0:
    st.info("No matching partners found.")
    st.stop()

selected_partner = st.selectbox(
    "Select a partner",
    sorted(partner_names),
    index=None,
    placeholder="Choose a partner to explore",
)

if selected_partner is None:
    st.info("Select a partner above to display detailed indicators.")
    st.stop()

partner_row = partners_df.loc[partners_df["Partner name"] == selected_partner]
if partner_row.empty:
    st.warning("No detailed data available for this partner.")
    st.stop()

partner_row = partner_row.iloc[0]

# ---------------------------------------------------------------------
# 2) Topline metrics
# ---------------------------------------------------------------------
pname = partner_row["Partner name"]
pcountry = partner_row.get("Partner country", "")
ptype = partner_row.get("Partner type", "")
pid = str(partner_row.get("Partner ID", ""))

st.markdown(
    f"**{pname}**  \n"
    f"{pcountry} – {ptype if ptype else 'Unknown type'}  \n"
    f"`OpenAlex ID: {pid}`"
)

copubs = int(partner_row.get("Count of co-publications", 0) or 0)
partner_total_output = int(partner_row.get("Partner's total output (2020-24)", 0) or 0)
share_upcite_total = float(partner_row.get("Share of UPCité's production", 0.0) or 0.0)
share_partner_total = float(partner_row.get("Share of Partner's total production", 0.0) or 0.0)
avg_fwci = float(partner_row.get("average FWCI", 0.0) or 0.0)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Co-publications with UPCité (2020–24)", f"{copubs:,}")

with col2:
    st.metric(
        "Share of UPCité's total output",
        f"{share_upcite_total * 100:.2f}%",
    )

with col3:
    st.metric(
        "Share of partner's total output",
        f"{share_partner_total * 100:.2f}%",
    )

with col4:
    st.metric("Average FWCI of co-publications", f"{avg_fwci:.2f}")

# ---------------------------------------------------------------------
# 3) Yearly distribution by domain
# ---------------------------------------------------------------------
st.markdown("### Yearly distribution by domain")

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

df_year_dom = parse_partner_year_domain_counts(partner_row.get("Copubs per year and domain"))
df_year_dom = df_year_dom[df_year_dom["copubs"] > 0]

if df_year_dom.empty:
    st.info("No yearly/domain breakdown is available for this partner.")
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
        margin=dict(l=0, r=10, t=10, b=10),
        showlegend=False,
        height=380,
    )
    st.plotly_chart(fig_year_dom, width="stretch")

# ---------------------------------------------------------------------
# 4) Distribution by field: shares & FWCI
# ---------------------------------------------------------------------
st.markdown("### Thematic profile of collaborations")

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

df_fields = build_partner_field_df(partner_row)

col_share, col_fwci = st.columns(2)

with col_share:
    st.markdown(
        "#### Distribution by field  \n"
        "among co-publications with this partner"
    )
    if df_fields.empty or df_fields["count"].sum() == 0:
        st.info("No field-level co-publications recorded for this partner.")
    else:
        fig_share = px.bar(
            df_fields,
            x="share_with_partner_pct",
            y="Field",
            orientation="h",
            color="domain",
            color_discrete_map=DOMAIN_COLORS,
            labels={"share_with_partner_pct": "Share (%)", "Field": ""},
            custom_data=["count", "fwci"],  # <-- key change
        )

        fig_share.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Share of this partner's co-publications: %{x:.1f}%<br>"
                "Co-publications: %{customdata[0]:,.0f}<br>"
                "Average FWCI: %{customdata[1]:.2f}"
                "<extra></extra>"
            ),
        )

        max_share = float(df_fields["share_with_partner_pct"].max() or 0.0)
        if max_share <= 0:
            max_share = 1.0
        gutter = max_share * 0.20

        fig_share.update_xaxes(
            range=[-gutter, max_share * 1.05],
            showgrid=True,
            gridcolor="#e0e0e0",
            ticksuffix="%",
            tickfont=dict(size=12),
        )
        fig_share.update_yaxes(tickfont=dict(size=13))

        # Counts in gutter
        for field_name, cnt in zip(df_fields["Field"], df_fields["count"]):
            fig_share.add_annotation(
                x=-gutter * 0.98,
                y=field_name,
                text=f"{int(cnt)}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=12, color="#444"),
            )

        fig_share.update_layout(
            margin=dict(l=0, r=10, t=25, b=10),
            showlegend=False,
            height=600,
        )

        st.plotly_chart(fig_share, width="stretch")

with col_fwci:
    st.markdown(
        "#### Distribution by field  \n"
        "Average FWCI of co-publications"
    )
    if df_fields.empty or df_fields["count"].sum() == 0:
        st.info("No field-level co-publications recorded for this partner.")
    else:
        fig_fwci = px.bar(
            df_fields,
            x="fwci",
            y="Field",
            orientation="h",
            color="domain",
            color_discrete_map=DOMAIN_COLORS,
            labels={"fwci": "Average FWCI", "Field": ""},
            custom_data=["count", "share_with_partner_pct"],  # <-- key change
        )

        fig_fwci.update_traces(
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Average FWCI: %{x:.2f}<br>"
                "Co-publications: %{customdata[0]:,.0f}<br>"
                "Share of this partner's co-publications: %{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
        )

        max_fwci = float(df_fields["fwci"].max() or 0.0)
        if max_fwci <= 0:
            max_fwci = 1.0
        gutter2 = max_fwci * 0.20

        fig_fwci.update_xaxes(
            range=[-gutter2, max_fwci * 1.15],
            showgrid=True,
            gridcolor="#e0e0e0",
            tickfont=dict(size=12),
        )
        fig_fwci.update_yaxes(tickfont=dict(size=13))

        for field_name, cnt in zip(df_fields["Field"], df_fields["count"]):
            fig_fwci.add_annotation(
                x=-gutter2 * 0.98,
                y=field_name,
                text=f"{int(cnt)}",
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                font=dict(size=12, color="#444"),
            )

        fig_fwci.update_layout(
            margin=dict(l=0, r=10, t=25, b=10),
            showlegend=False,
            height=600,
        )

        st.plotly_chart(fig_fwci, width="stretch")

# ---------------------------------------------------------------------
# 5) Strategic weight per field (bubble chart)
# ---------------------------------------------------------------------
st.markdown("### Strategic weight of co-publications by field")

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

lock_axes = st.toggle(
    "Use the same scale for X and Y", 
    value=False, 
    help="When on, both axes use the same max so the diagonal is visually meaningful."
)

df_bub = df_fields[df_fields["count"] > 0].copy()
if df_bub.empty:
    st.info("No fields with co-publications to display.")
else:
    # Limit to fields where either share vs UPCité or share vs partner total is > 0
    df_bub = df_bub[
        (df_bub["rel_vs_upcite_pct"] > 0) | (df_bub["rel_vs_partner_pct"] > 0)
    ]
    if df_bub.empty:
        st.info("No non-zero strategic weights available for this partner.")
    else:

        
        # Independent axis ranges for X and Y (values are already in %)
        max_x = float(df_bub["rel_vs_upcite_pct"].max() or 0.0)
        max_y = float(df_bub["rel_vs_partner_pct"].max() or 0.0)

        x_max = max_x * 1.05 if max_x > 0 else 1.0
        y_max = max_y * 1.05 if max_y > 0 else 1.0

        # If toggle is ON, force both axes to the same max
        if lock_axes:
            v = max(x_max, y_max)
            x_max = y_max = v

        # Diagonal must stay within the visible frame
        diag_max = min(x_max, y_max)

        fig_bub = px.scatter(
            df_bub,
            x="rel_vs_upcite_pct",
            y="rel_vs_partner_pct",
            size="count",  # bubble size = count
            size_max=40,
            color="domain",
            color_discrete_map=DOMAIN_COLORS,
            hover_name="Field",
            labels={
                "rel_vs_upcite_pct": "Share vs UPCité total in this field (%)",
                "rel_vs_partner_pct": "Share vs partner total in this field (%)",
                "fwci": "Average FWCI",
                "domain": "",
            },
            custom_data=["count", "fwci"],  # <-- key change
        )

        fig_bub.update_traces(
            hovertemplate=(
                "<b>%{hovertext}</b><br>"
                "Share vs UPCité total: %{x:.2f}%<br>"
                "Share vs partner total: %{y:.2f}%<br>"
                "Average FWCI: %{customdata[1]:.2f}<br>"
                "Co-publications: %{customdata[0]:,.0f}"
                "<extra></extra>"
            ),
        )

        # y = x diagonal, clipped to the smaller axis
        fig_bub.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=diag_max,
            y1=diag_max,
            line=dict(color="#444", dash="dash"),
        )

        fig_bub.update_xaxes(range=[0, x_max], showgrid=True, gridcolor="#eee")
        fig_bub.update_yaxes(range=[0, y_max], showgrid=True, gridcolor="#eee")

        fig_bub.update_layout(
            margin=dict(l=0, r=10, t=25, b=10),
            height=520,
            showlegend=False,
            legend_title_text="",
        )

        st.plotly_chart(fig_bub, width="stretch")

# ---------------------------------------------------------------------
# 6) Subfield table
# ---------------------------------------------------------------------
st.markdown("### Subfield detail")

df_sub = make_partner_subfield_df(partner_row)

if df_sub.empty:
    st.info("No subfield-level data for this partner.")
else:
    # Domain marker (emoji + name)
    df_sub_display = df_sub.copy()
    df_sub_display["Domain marker"] = df_sub_display["Domain"].apply(
        lambda d: f"{DOMAIN_EMOJI.get(d, '⬜')} {d}"
    )

    df_sub_display = df_sub_display[
        [
            "Domain marker",
            "Field",
            "Subfield",
            "share_mix",
            "share_vs_partner_total",
            "share_vs_upcite_total",
            "copubs",
            "partner_total_subfield",
            "fwci",
        ]
    ].rename(
        columns={
            "Domain marker": "Domain",
            "share_mix": "Share within co-publications",
            "share_vs_partner_total": "Share vs partner total",
            "share_vs_upcite_total": "Share vs UPCité total",
            "copubs": "Co-publications",
            "partner_total_subfield": "Partner's total pubs in subfield",
            "fwci": "FWCI",
        }
    )

    st.dataframe(
        df_sub_display,
        hide_index=True,
        width="stretch",
        column_config={
            "Share within co-publications": st.column_config.ProgressColumn(
                "Share within co-publications",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
                help="Fraction of this partner's co-publications with UPCité that fall in this subfield.",
            ),
            "Share vs partner total": st.column_config.ProgressColumn(
                "Share vs partner total",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
                help="Share of the partner's total output in this subfield that involves UPCité.",
            ),
            "Share vs UPCité total": st.column_config.ProgressColumn(
                "Share vs UPCité total",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
                help="Share of UPCité's total output in this subfield that involves this partner.",
            ),
            "Co-publications": st.column_config.NumberColumn(
                "Co-publications", format="%d"
            ),
            "Partner's total pubs in subfield": st.column_config.NumberColumn(
                "Partner's total pubs in subfield", format="%d"
            ),
            "FWCI": st.column_config.NumberColumn("FWCI", format="%.2f"),
        },
    )
