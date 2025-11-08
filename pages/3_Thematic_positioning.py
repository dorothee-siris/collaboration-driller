# pages/3_Thematic_positioning.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from lib.taxonomy import build_taxonomy_lookups  # from lib/taxonomy.py

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
LIB_DIR = BASE_DIR / "lib"

if str(LIB_DIR) not in sys.path:
    sys.path.append(str(LIB_DIR))


# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Pipe parsing helpers
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
                vals.append(float(p))
            except ValueError:
                vals.append(0.0)
    return vals


# ---------------------------------------------------------------------
# Build taxonomy metadata aligned with breakdown ordering
# ---------------------------------------------------------------------
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
    field_index = {name: i for i, name in enumerate(field_names)}
    field_id_by_name = dict(zip(field_names, field_ids))
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
        "field_index": field_index,
        "field_id_by_name": field_id_by_name,
        "field_domain_by_name": field_domain_by_name,
        "field_to_subnames": field_to_subnames,
        "subfield_domain_by_name": subfield_domain_by_name,
    }


# ---------------------------------------------------------------------
# Build hierarchical scope options from UPCité lookup
# ---------------------------------------------------------------------
@st.cache_data
def build_scope_options() -> Tuple[List[Any], List[str]]:
    """
    Returns:
      - keys: list of scope keys
      - labels: list of labels to display
    key formats:
      - ("all",)
      - ("field", field_name)
      - ("subfield", field_name, subfield_name, field_id, sub_idx)
    """
    lookup_row = load_lookup_row()
    tax_meta = build_taxonomy_from_topics()

    field_ids = tax_meta["field_ids"]
    field_names = tax_meta["field_names"]
    field_id_by_name = tax_meta["field_id_by_name"]
    field_to_subnames = tax_meta["field_to_subnames"]

    # UPCité field counts in correct order
    field_counts = parse_pipe_ints(lookup_row["Pubs breakdown per field"])
    field_counts_by_id = dict(zip(field_ids, field_counts))

    keys: List[Any] = []
    labels: List[str] = []

    # Global
    keys.append(("all",))
    labels.append("All UPCité output")

    lookups = build_taxonomy_lookups()
    fields_by_domain = lookups["fields_by_domain"]
    domain_order = lookups["domain_order"]

    for dom in domain_order:
        fields = fields_by_domain.get(dom, [])
        for fname in fields:
            if fname not in field_id_by_name:
                continue
            fid = field_id_by_name[fname]
            if fid not in field_counts_by_id:
                continue

            total_f = field_counts_by_id[fid]
            if total_f < 20:
                continue

            # Field-level option
            keys.append(("field", fname))
            labels.append(f"{dom} / {fname}")

            # Subfields within this field
            col_sub = f'Pubs per subfield within "{fname}" (id: {fid}) (partner total)'
            if col_sub not in lookup_row.index:
                continue

            sub_counts = parse_pipe_ints(
                lookup_row[col_sub].replace("(partner total)", "")
                if "(partner total)" in col_sub
                else lookup_row[col_sub]
            )
            # But better: use copub counts for threshold; upcite_lookup has similar
            col_upc_sub = f'Pubs per subfield within "{fname}" (id: {fid})'
            if col_upc_sub in lookup_row.index:
                upc_sub_counts = parse_pipe_ints(lookup_row[col_upc_sub])
            else:
                upc_sub_counts = []

            subnames = field_to_subnames.get(fid, [])
            for idx, sn in enumerate(subnames):
                cnt = upc_sub_counts[idx] if idx < len(upc_sub_counts) else 0
                if cnt >= 20:
                    keys.append(("subfield", fname, sn, fid, idx))
                    labels.append(f"    ↳ {sn}")

    return keys, labels


# ---------------------------------------------------------------------
# Build scatter data
# ---------------------------------------------------------------------
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
        share_part_col = (
            f'Relative share per subfield within "{field_name}" (id: {fid}) '
            f'vs Partner total'
        )

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

    df = df[df["count_scope"] > 0].copy()
    df = df.sort_values("count_scope", ascending=False).head(100)

    return df


# ---------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------
st.title("Thematic positioning & strategic weights")

st.markdown(
    "Each point is a partner.  \n"
    "**X-axis**: share of UPCité’s output in the selected scope that involves this partner.  \n"
    "**Y-axis**: share of the partner’s own output in the same scope that involves UPCité.  \n"
    "Bubble size = average FWCI of co-publications."
)

partners_df = load_partners()
TOP_THRESHOLD = 20
top_partners_df = partners_df[
    partners_df["Count of co-publications"] >= TOP_THRESHOLD
].copy()

keys, labels = build_scope_options()
labels_by_key = {k: v for k, v in zip(keys, labels)}
key_by_label = {v: k for k, v in zip(keys, labels)}

selected_label = st.selectbox("Scope", labels, index=0)
selected_key = key_by_label[selected_label]

scatter_df = build_scatter_df(top_partners_df, selected_key)

if scatter_df.empty:
    st.info(
        "No partners found in this scope (or UPCité has < 20 publications in this category)."
    )
else:
    max_xy = float(max(scatter_df["x"].max(), scatter_df["y"].max()) * 1.05 or 0.01)

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
