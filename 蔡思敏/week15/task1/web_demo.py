"""
多模态RAG聊天机器人 - Web入口

功能：
- 页面导航：文件管理 | 图文对话

使用方法：
    streamlit run web_demo.py
"""

import streamlit as st

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="多模态RAG聊天机器人",
    page_icon="🤖",
    layout="wide"
)

# ==================== 多页面导航 ====================
pg = st.navigation([
    st.Page("web_page_upload.py", title="文件管理", icon="📁"),
    st.Page("web_page_chat.py", title="图文对话", icon="💬"),
])

# ==================== 运行 ====================
pg.run()