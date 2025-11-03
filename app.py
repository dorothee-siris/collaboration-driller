# app.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Paths & imports
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LIB_DIR = BASE_DIR / "lib"

if str(LIB_DIR) not in sys.path:
    sys.path.append(str(LIB_DIR))

from taxonomy import (  # type: ignore
    build_taxonomy_lookups,
    get_field_color,
    get_subfield_color,
    get_domain_color,
)

# Optional: click events on map (if installed)
try:
    from streamlit_plotly_events import plotly_events  # type: ignore

    HAVE_PLOTLY_EVENTS = True
except Exception:  # pragma: no cover
    HAVE_PLOTLY_EVENTS = False


# -----------------------------------------------------------------------------
# Caching loaders
# -----------------------------------------------------------------------------
@st.cache_data
def load_core() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "upcite_core.parquet")


@st.cache_data
def load_partners() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "upcite_partners.parquet")


@st.cache_data
def load_lookup_row() -> pd.Series:
    df = pd.read_parquet(DATA_DIR / "upcite_lookup.parquet")
    return df.iloc[0]


@st.cache_data
def load_topics() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "all_topics.parquet")


# -----------------------------------------------------------------------------
# Helpers to parse pipe-separated strings
# -----------------------------------------------------------------------------
def parse_pipe_ints(s: Any) -> List[int]:
    if pd.isna(s) or s == "":
        return []
    parts = str(s).split("|")
    vals: List[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            vals.append(0)
        else:
            try:
                vals.append(int(p))
            except ValueError:
                vals.append(0)
    return vals


def parse_pipe_floats(s: Any) -> List[float]:
    if pd.isna(s) or s == "":
        return []
    parts = str(s).split("|")
    vals: List[float] = []
    for p in parts:
        p = p.strip()
        if not p:
            vals.append(0.0)
        else:
            try:
                vals.append(float(p))
            except ValueError:
                vals.append(0.0)
    return vals


def pipe_ratio(counts_str: Any, totals_str: Any, fmt: str = "{:.4f}") -> str:
    a = parse_pipe_ints(counts_str)
    b = parse_pipe_ints(totals_str)
    if not a or not b:
        return ""
    n = min(len(a), len(b))
    shares = []
    for i in range(n):
        denom = b[i]
        if denom > 0:
            shares.append(fmt.format(a[i] / denom))
        else:
            shares.append(fmt.format(0.0))
    return " | ".join(shares)


# -----------------------------------------------------------------------------
# Build field / subfield universes (ID-ordered, consistent with breakdowns)
# -----------------------------------------------------------------------------
@st.cache_data
def build_taxonomy_from_topics():
    topics = load_topics()

    field_meta = (
        topics[["field_id", "field_name", "domain_name"]]
        .drop_duplicates()
        .sort_values("field_id")
    )
    field_ids = field_meta["field_id"].astype(int).tolist()
    field_names = field_meta["field_name"].tolist()
    field_index = {name: i for i, name in enumerate(field_names)}
    field_id_by_name = dict(zip(field_names, field_ids))
    field_domain_by_name = dict(zip(field_names, field_meta["domain_name"]))

    sub_meta = (
        topics[["field_id", "field_name", "subfield_id", "subfield_name", "domain_name"]]
        .drop_duplicates()
        .sort_values(["field_id", "subfield_id"])
    )
    field_to_subnames: Dict[int, List[str]] = {}
    subfield_domain_by_name: Dict[str, str] = {}
    for (fid, _fname), group in sub_meta.groupby(["field_id", "field_name"]):
        subnames = group["subfield_name"].tolist()
        field_to_subnames[int(fid)] = subnames
        for sn, dn in zip(group["subfield_name"], group["domain_name"]):
            subfield_domain_by_name[str(sn)] = str(dn)

    return {
        "field_ids": field_ids,
        "field_names": field_names,
        "field_index": field_index,
        "field_id_by_name": field_id_by_name,
        "field_domain_by_name": field_domain_by_name,
        "field_to_subnames": field_to_subnames,
        "subfield_domain_by_name": subfield_domain_by_name,
    }


# -----------------------------------------------------------------------------
# Build hierarchical filter options from UPCité lookup (for strategic graph)
# -----------------------------------------------------------------------------
@st.cache_data
def build_scope_options() -> Tuple[List[Any], List[str]]:
    """
    Returns:
      - keys: list of scope keys
      - labels: list of labels to display in dropdown
    key formats:
      - ("all",)
      - ("field", field_name)
      - ("subfield", field_name, subfield_name, field_id, sub_idx)
    """
    lookup_row = load_lookup_row()
    tax_meta = build_taxonomy_from_topics()
    look = build_taxonomy_lookups()

    field_ids = tax_meta["field_ids"]
    field_names = tax_meta["field_names"]
    field_id_by_name = tax_meta["field_id_by_name"]
    field_to_subnames = tax_meta["field_to_subnames"]

    # UPCité field counts (ID order)
    field_counts = parse_pipe_ints(lookup_row["Pubs breakdown per field"])
    field_counts_by_id = dict(zip(field_ids, field_counts))

    keys: List[Any] = []
    labels: List[str] = []

    # Global
    keys.append(("all",))
    labels.append("All UPCité output")

    # Use taxonomy's hierarchical view for nice domain grouping
    lookups = build_taxonomy_lookups()
    fields_by_domain = lookups["fields_by_domain"]
    domain_order = lookups["domain_order"]

    for dom in domain_order:
        fields = fields_by_domain.get(dom, [])
        for fname in fields:
            if fname not in field_id_by_name:
                continue
            fid = field_id_by_name[fname]
            # skip fields not in our ordered list
            if fid not in field_counts_by_id:
                continue

            total_f = field_counts_by_id[fid]
            if total_f < 20:
                continue

            # Field-level option
            keys.append(("field", fname))
            labels.append(f"{dom} / {fname}")

            # Subfields within this field
            col_sub = f'Pubs per subfield within "{fname}" (id: {fid})'
            if col_sub not in lookup_row.index:
                continue

            sub_counts = parse_pipe_ints(lookup_row[col_sub])
            subnames = field_to_subnames.get(fid, [])
            for idx, (sn, cnt) in enumerate(zip(subnames, sub_counts)):
                if cnt >= 20:
                    keys.append(("subfield", fname, sn, fid, idx))
                    labels.append(f"    ↳ {sn}")

    return keys, labels


# -----------------------------------------------------------------------------
# Scatter data builder for strategic weights
# -----------------------------------------------------------------------------
def build_scatter_df(partners: pd.DataFrame, scope_key: Any) -> pd.DataFrame:
    tax_meta = build_taxonomy_from_topics()
    field_index = tax_meta["field_index"]

    df = partners.copy()

    if scope_key[0] == "all":
        df["x"] = df["Share of UPCité's production"]
        df["y"] = df["Share of Partner's total production"]
        df["count_scope"] = df["Count of co-publications"]

    elif scope_key[0] == "field":
        field_name = scope_key[1]
        idx = field_index.get(field_name, None)
        if idx is None:
            return df.head(0)

        def _get_idx_float(s, i=idx):
            vals = parse_pipe_floats(s)
            return vals[i] if i < len(vals) else 0.0

        def _get_idx_int(s, i=idx):
            vals = parse_pipe_ints(s)
            return vals[i] if i < len(vals) else 0

        df["x"] = df["Relative share per field vs UPCité total"].apply(_get_idx_float)
        df["y"] = df["Relative share per field vs Partner total"].apply(_get_idx_float)
        df["count_scope"] = df["Copubs per field"].apply(_get_idx_int)

    elif scope_key[0] == "subfield":
        _, field_name, sub_name, fid, sub_idx = scope_key

        copub_col = f'Copubs per subfield within "{field_name}" (id: {fid})'
        share_upc_col = f'Relative share per subfield within "{field_name}" (id: {fid})'
        share_part_col = f'Relative share per subfield within "{field_name}" (id: {fid}) vs Partner total'

        def _get_idx_int(s, i=sub_idx):
            vals = parse_pipe_ints(s)
            return vals[i] if i < len(vals) else 0

        def _get_idx_float(s, i=sub_idx):
            vals = parse_pipe_floats(s)
            return vals[i] if i < len(vals) else 0.0

        if copub_col not in df.columns:
            return df.head(0)

        df["count_scope"] = df[copub_col].apply(_get_idx_int)
        df["x"] = df[share_upc_col].apply(_get_idx_float)
        df["y"] = df[share_part_col].apply(_get_idx_float)

    # Filter and keep top 100 by count_scope
    df = df[df["count_scope"] > 0].copy()
    df = df.sort_values("count_scope", ascending=False).head(100)

    return df


# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="UPCité Collaborations Explorer",
    layout="wide",
    page_icon="🌐",
)

st.title("Université Paris Cité – Collaborations Explorer")

st.caption(
    "Explore UPCité’s international collaborations, strategic partners and thematic profiles "
    "using OpenAlex-based data."
)

# Load core datasets
core_df = load_core()
partners_df = load_partners()
lookup_row = load_lookup_row()
tax_meta = build_taxonomy_from_topics()

# Restrict to “top partners” (>= 20 co-publications)
TOP_THRESHOLD = 20
top_partners_df = partners_df[partners_df["Count of co-publications"] >= TOP_THRESHOLD].copy()

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
view = st.sidebar.radio(
    "Sections",
    [
        "Overview",
        "World map",
        "Strategic weights",
        "Partner drilldown",
    ],
)


# -----------------------------------------------------------------------------
# OVERVIEW: topline metrics + partners table
# -----------------------------------------------------------------------------
if view == "Overview":
    st.subheader("Topline metrics")

    total_pubs = len(core_df)
    intl_flag = core_df["is_international"].astype(str).str.upper().eq("TRUE")
    intl_pubs = int(intl_flag.sum())
    intl_share = intl_pubs / total_pubs if total_pubs else 0.0

    total_copubs = int((core_df["Partners"].fillna("").str.strip() != "").sum())
    n_top_partners = len(top_partners_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total UPCité publications (2020–24)", f"{total_pubs:,}")
    col2.metric("International publications", f"{intl_pubs:,}", f"{intl_share*100:,.1f}%")
    col3.metric("Publications with partners", f"{total_copubs:,}")
    col4.metric(f"Top partners (≥ {TOP_THRESHOLD} co-pubs)", f"{n_top_partners:,}")

    st.markdown("---")

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

    # Minimal columns for the table
    display_cols = [
        "Partner name",
        "Partner country",
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
                max_value=float(df_tbl["Share of UPCité's production"].max() or 0.001),
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


# -----------------------------------------------------------------------------
# WORLD MAP: choropleth + per-country partners
# -----------------------------------------------------------------------------
elif view == "World map":
    st.subheader("Global co-publications map (France excluded)")

    # Aggregate co-pubs by country (excluding France to avoid visual imbalance)
    country_agg = (
        partners_df.groupby("Partner country", as_index=False)[
            "Count of co-publications"
        ]
        .sum()
        .rename(columns={"Count of co-publications": "Co-publications"})
    )
    country_agg = country_agg[country_agg["Partner country"] != "France"]

    if country_agg.empty:
        st.info("No international partners found in the dataset.")
    else:
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

        selected_country = None

        if HAVE_PLOTLY_EVENTS:
            events = plotly_events(
                fig_map,
                click_event=True,
                hover_event=False,
                key="map",
            )
            st.caption("Click on a country to drill down to its partners.")
            if events:
                selected_country = events[0].get("location")
        else:
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption(
                "To drill down, select a country below. "
                "Install `streamlit-plotly-events` if you want click-based selection."
            )

        if not selected_country:
            selected_country = st.selectbox(
                "Select a country to list its partners:",
                [""] + sorted(country_agg["Partner country"].unique()),
            )
            if selected_country == "":
                selected_country = None

        if selected_country:
            st.markdown(f"### Top partners in **{selected_country}**")
            sub = partners_df[partners_df["Partner country"] == selected_country].copy()
            sub = sub.sort_values("Count of co-publications", ascending=False)

            cols = [
                "Partner name",
                "Partner country",
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


# -----------------------------------------------------------------------------
# STRATEGIC WEIGHTS: scatter plot with hierarchical filter
# -----------------------------------------------------------------------------
elif view == "Strategic weights":
    st.subheader("Strategic weights of partners")

    st.markdown(
        "Each point is a partner. "
        "**X-axis**: share of UPCité’s output in the selected scope that involves this partner.  "
        "**Y-axis**: share of the partner’s own output in the scope that involves UPCité.  "
        "Bubble size = average FWCI of co-publications."
    )

    keys, labels = build_scope_options()
    labels_by_key = {k: v for k, v in zip(keys, labels)}
    key_by_label = {v: k for k, v in zip(keys, labels)}

    selected_label = st.selectbox("Scope", labels, index=0)
    selected_key = key_by_label[selected_label]

    scatter_df = build_scatter_df(top_partners_df, selected_key)

    if scatter_df.empty:
        st.info("No partners found in this scope (or UPCité has < 20 pubs here).")
    else:
        max_xy = float(
            max(scatter_df["x"].max(), scatter_df["y"].max()) * 1.05 or 0.01
        )

        fig = px.scatter(
            scatter_df,
            x="x",
            y="y",
            size="average FWCI",
            color="Partner country",
            hover_name="Partner name",
            hover_data={
                "Count of co-publications": True,
                "x": False,
                "y": False,
            },
            labels={
                "x": "Relative share vs UPCité",
                "y": "Relative share vs partner total",
            },
        )

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
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))

        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# PARTNER DRILLDOWN
# -----------------------------------------------------------------------------
elif view == "Partner drilldown":
    st.subheader("Drill down into a specific partner")

    search = st.text_input("Search partner by name (min. 2 characters)", "")
    matches = partners_df["Partner name"].dropna().unique()

    if search and len(search.strip()) >= 2:
        matches = [m for m in matches if search.lower() in m.lower()]

    if len(matches) == 0:
        st.info("No matching partners found.")
    else:
        selected_partner = st.selectbox("Select a partner", sorted(matches))

        partner_row = partners_df[
            partners_df["Partner name"] == selected_partner
        ].iloc[0]

        st.markdown(
            f"**{partner_row['Partner name']}** – {partner_row['Partner country']}  \n"
            f"(ROR: `{partner_row.get('Partner ROR', '')}`, "
            f"OpenAlex: `{partner_row.get('Partner OpenAlex ID', '')}`)"
        )

        copubs = partner_row["Count of co-publications"]
        tot_output = partner_row["Partner's total output (2020-24)"]
        share_upcite = partner_row["Share of UPCité's production"]
        share_partner = partner_row["Share of Partner's total production"]
        avg_fwci = partner_row["average FWCI"]

        st.markdown(
            f"- Co-publications with UPCité: **{copubs:,}**  \n"
            f"- Partner's total output (2020–24): **{tot_output:,}**  \n"
            f"- Share of UPCité's production: **{share_upcite:.3f}**  \n"
            f"- Share of partner's total production: **{share_partner:.3f}**  \n"
            f"- Average FWCI of co-publications: **{avg_fwci:.2f}**"
        )

        # ---------- Distribution across fields ----------
        st.markdown("### Distribution of co-publications across fields")

        field_counts = parse_pipe_ints(partner_row["Copubs per field"])
        field_ids = tax_meta["field_ids"]
        field_names = tax_meta["field_names"]
        field_domain_by_name = tax_meta["field_domain_by_name"]

        field_rows = []
        for fid, fname, cnt in zip(field_ids, field_names, field_counts):
            if cnt > 0:
                dom = field_domain_by_name.get(fname, "Other")
                field_rows.append(
                    {
                        "Field": fname,
                        "Count": cnt,
                        "Domain": dom,
                        "Color": get_field_color(fname),
                    }
                )

        if field_rows:
            fdf = pd.DataFrame(field_rows).sort_values("Count", ascending=True)
            fig_f = go.Figure()
            fig_f.add_trace(
                go.Bar(
                    x=fdf["Count"],
                    y=fdf["Field"],
                    orientation="h",
                    marker_color=fdf["Color"],
                    text=fdf["Count"],
                    textposition="outside",
                )
            )
            fig_f.update_layout(
                height=400,
                margin=dict(l=0, r=10, t=10, b=10),
                xaxis_title="Co-publications",
            )
            st.plotly_chart(fig_f, use_container_width=True)
        else:
            st.info("This partner has no field-level co-publications recorded.")

        # ---------- Top subfields ----------
        st.markdown("### Top 30 subfields (by co-publications)")

        field_to_subnames = tax_meta["field_to_subnames"]
        subfield_domain_by_name = tax_meta["subfield_domain_by_name"]

        sub_rows = []
        for fid, fname in zip(tax_meta["field_ids"], tax_meta["field_names"]):
            col_sub = f'Copubs per subfield within "{fname}" (id: {fid})'
            if col_sub not in partner_row.index:
                continue
            counts = parse_pipe_ints(partner_row[col_sub])
            subnames = field_to_subnames.get(fid, [])
            for sn, cnt in zip(subnames, counts):
                if cnt <= 0:
                    continue
                dom = subfield_domain_by_name.get(sn, "Other")
                sub_rows.append(
                    {
                        "Subfield": sn,
                        "Field": fname,
                        "Domain": dom,
                        "Count": cnt,
                        "Color": get_subfield_color(sn),
                    }
                )

        if sub_rows:
            sdf = (
                pd.DataFrame(sub_rows)
                .sort_values("Count", ascending=True)
                .tail(30)
            )
            fig_s = go.Figure()
            fig_s.add_trace(
                go.Bar(
                    x=sdf["Count"],
                    y=sdf["Subfield"],
                    orientation="h",
                    marker_color=sdf["Color"],
                    text=sdf["Count"],
                    textposition="outside",
                )
            )
            fig_s.update_layout(
                height=600,
                margin=dict(l=0, r=10, t=10, b=10),
                xaxis_title="Co-publications",
            )
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.info("No subfield-level co-publications available for this partner.")

        # ---------- List of co-publications ----------
        st.markdown("### List of co-publications (sample)")

        # Use ROR as robust lookup in core_df
        ror_code = str(partner_row.get("Partner ROR", "")).strip()
        if ror_code:
            pattern = f"[{ror_code}]"
        else:
            pattern = partner_row["Partner name"]

        core_match = core_df[
            core_df["Partners"].astype(str).str.contains(
                pattern, na=False, regex=False
            )
        ].copy()

        core_match = core_match.sort_values("publication_year", ascending=False)

        cols = [
            "id",
            "title",
            "publication_year",
            "fwci",
            "cited_by_count",
            "domain_name",
            "field_name",
            "subfield_name",
        ]

        st.caption("Showing up to 50 most recent co-publications below:")
        st.dataframe(core_match[cols].head(50), use_container_width=True)

        csv = core_match[cols].to_csv(index=False).encode("utf-8-sig")
        safe_name = (
            partner_row["Partner name"]
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )
        st.download_button(
            "⬇️ Download all co-publications as CSV",
            data=csv,
            file_name=f"upcite_copubs_{safe_name}.csv",
            mime="text/csv",
        )