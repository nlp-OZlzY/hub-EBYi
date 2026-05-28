"""Streamlit entry point with multi-page navigation."""

import streamlit as st

pg = st.navigation([
    st.Page("web_page_upload.py", title="文件管理"),
    st.Page("web_page_chat.py", title="图文对话"),
])
pg.run()
