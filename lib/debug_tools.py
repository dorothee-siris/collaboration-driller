# lib/debug_tools.py

import os
import psutil
import streamlit as st


def render_debug_sidebar():
    """Show memory usage and a cache-clear button in the sidebar."""
    with st.sidebar.expander("⚙️ Debug / performance", expanded=False):
        proc = psutil.Process(os.getpid())
        mb = proc.memory_info().rss / 1024**2
        st.caption(f"Current RAM (Python process): **{mb:,.1f} MB**")

        if st.button("Clear all caches & rerun"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()