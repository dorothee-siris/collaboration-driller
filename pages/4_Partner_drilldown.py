# pages/4_Partner_drilldown.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Any, Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.taxonomy import (
    get_field_color,
    get_subfield_color,
)

# ---------------------------------------------------------------------
# Paths & loaders
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LIB_DIR = BASE_DIR / "lib"

if str(LIB_DIR) not in sys.path:
    sys.path.append(str(LIB_DIR))


@st.cache_data
def load_core() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "upcite_core.parquet")


@st.cache_data
def load_partners() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "upcite_partners.parquet")


@st.cache_data
def load_topics() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "all_topics.parquet")


# ---------------------------------------------------------------------
# Helpers
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
            except ValueError:
                vals.append(0)
    return vals


@st.cache_data
def build_taxonomy_from_topics() -> Dict:
    topics = load_topics()

    field_meta = (
        topics[["field_id", "field_name", "domain_name"]]
        .drop_duplicates()
        .sort_values("field_id")
    )
    field_ids = field_meta["field_id"].astype(int).tolist()
    field_names = field_meta["field_name"].tolist()
    field_domain_by_name = dict(zip(field_names, field_meta["domain_name"]))

    sub_meta = (
        topics[
            ["field_id", "field_name", "subfield_id", "subfield_name", "domain_name"]
        ]
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
        "field_domain_by_name": field_domain_by_name,
        "field_to_subnames": field_to_subnames,
        "subfield_domain_by_name": subfield_domain_by_name,
    }


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------
st.title("Partner drilldown")

core_df = load_core()
partners_df = load_partners()
tax_meta = build_taxonomy_from_topics()

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

    pname = partner_row["Partner name"]
    pcountry = partner_row["Partner country"]
    ptype = partner_row.get("Partner type", "")
    pid = str(partner_row.get("Partner ID", ""))

    st.markdown(
        f"**{pname}** – {pcountry}  ({ptype})  \n"
        f"Partner ID: `{pid}`"
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

    # ----------------- Distribution across fields -------------------
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

    # ----------------- Top subfields -------------------
    st.markdown("### Top 30 subfields (by co-publications)")

    field_to_subnames = tax_meta["field_to_subnames"]
    subfield_domain_by_name = tax_meta["subfield_domain_by_name"]

    sub_rows = []
    for fid, fname in zip(field_ids, field_names):
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

    # ----------------- List of co-publications -------------------
    st.markdown("### List of co-publications (sample)")

    # Use Partner ID inside lineage_affiliations
    core_match = core_df[
        core_df["lineage_affiliations"]
        .astype(str)
        .str.contains(pid, na=False)
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
        "topic_name",
    ]

    st.caption("Showing up to 50 most recent co-publications below:")
    st.dataframe(core_match[cols].head(50), use_container_width=True)

    csv = core_match[cols].to_csv(index=False).encode("utf-8-sig")
    safe_name = (
        pname.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )
    st.download_button(
        "⬇️ Download all co-publications as CSV",
        data=csv,
        file_name=f"upcite_copubs_{safe_name}.csv",
        mime="text/csv",
    )
