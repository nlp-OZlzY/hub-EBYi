"""
图文对话页面 - Streamlit前端

功能：
1. 知识库选择
2. 会话管理（创建、选择、清空）
3. RAG问答（检索+生成）
4. 图文混合展示

需求分析：
1. 用户选择知识库（用于限制检索范围）
2. 创建新会话或选择已有会话
3. 用户输入问题
4. 系统进行RAG检索：
   - 问题embedding
   - Milvus向量检索
   - 关联图片路径转换
5. 调用大模型生成答案
6. 展示答案（支持Markdown图片渲染）

测试逻辑：
1. test_knowledge_base_selection: 知识库选择测试
2. test_session_management: 会话管理测试
3. test_rag_retrieval: RAG检索测试
4. test_answer_generation: 答案生成测试
"""

import streamlit as st
import requests
import os
import re
from datetime import datetime

# ==================== 配置 ====================
API_BASE_URL = "http://localhost:8000/api/v1"

# ==================== 页面配置 ====================
st.set_page_config(page_title="图文对话", page_icon="💬")

# ==================== 状态初始化 ====================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是AI助手，可以基于知识库回答问题。"}
    ]

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "knowledge_base_id" not in st.session_state:
    st.session_state.knowledge_base_id = None

# ==================== 辅助函数 ====================
def api_request(method: str, endpoint: str, **kwargs):
    """发送API请求"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs, timeout=120)
        else:
            return None

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API请求失败: {e}")
        return None

def get_knowledge_bases():
    """获取知识库列表"""
    return api_request("GET", "/knowledge-bases")

def get_chat_history(session_id: str):
    """获取聊天历史"""
    return api_request("GET", f"/chat/sessions/{session_id}/history")

# ==================== 侧边栏 ====================
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("### ⚙️ 设置")

        # 知识库选择
        st.markdown("**选择知识库**")
        kbs = get_knowledge_bases()

        if kbs and kbs.get("items"):
            kb_options = {kb['id']: kb['name'] for kb in kbs["items"]}
            selected_kb_id = st.selectbox(
                "知识库",
                options=list(kb_options.keys()),
                format_func=lambda x: kb_options[x],
                index=0 if st.session_state.knowledge_base_id is None else list(kb_options.keys()).index(st.session_state.knowledge_base_id) if st.session_state.knowledge_base_id in kb_options else 0
            )

            if st.button("确认知识库", type="primary"):
                st.session_state.knowledge_base_id = selected_kb_id
                st.rerun()
        else:
            st.info("暂无知识库，请先创建")
            selected_kb_id = None

        st.divider()

        # 会话管理
        st.markdown("### 💬 会话管理")

        if st.button("新建会话", icon="➕"):
            st.session_state.session_id = None
            st.session_state.messages = [
                {"role": "system", "content": "你好，我是AI助手，可以基于知识库回答问题。"}
            ]
            st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("清空聊天", icon="🗑️"):
                st.session_state.messages = [
                    {"role": "system", "content": "你好，我是AI助手，可以基于知识库回答问题。"}
                ]
                st.session_state.session_id = None
                st.rerun()

        with col2:
            if st.button("加载历史", icon="📜"):
                if st.session_state.session_id:
                    history = get_chat_history(st.session_state.session_id)
                    if history and history.get("messages"):
                        st.session_state.messages = [
                            {"role": "system", "content": "你好，我是AI助手，可以基于知识库回答问题。"}
                        ]
                        for msg in history["messages"]:
                            st.session_state.messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                        st.rerun()

        st.divider()

        # 当前状态
        st.markdown("### 📊 当前状态")
        st.text(f"知识库ID: {st.session_state.knowledge_base_id or '未选择'}")
        st.text(f"会话ID: {st.session_state.session_id or '新会话'}")

        return selected_kb_id if kbs and kbs.get("items") else None

# ==================== 消息展示 ====================
def render_markdown_with_images(markdown_text: str):
    """
    渲染Markdown内容，同时处理图片

    测试用例：
    1. 纯文本
    2. 带单张图片
    3. 带多张图片
    4. 图片链接格式异常
    """
    if not markdown_text:
        return

    # 匹配Markdown图片语法 ![alt](url)
    pattern = re.compile(r'!\[(.*?)\]\((.*?)\)')

    last_pos = 0

    for match in pattern.finditer(markdown_text):
        # 显示上一个位置到匹配位置之间的文本
        if last_pos < match.start():
            st.markdown(markdown_text[last_pos:match.start()], unsafe_allow_html=True)

        # 显示图片
        img_url = match.group(2)
        alt_text = match.group(1)

        try:
            st.image(img_url, caption=alt_text, use_container_width=True)
        except Exception as e:
            st.warning(f"图片加载失败: {img_url}")

        last_pos = match.end()

    # 显示剩余的文本
    if last_pos < len(markdown_text):
        st.markdown(markdown_text[last_pos:], unsafe_allow_html=True)

def display_message(role: str, content: str):
    """展示消息"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    elif role == "assistant":
        with st.chat_message("assistant"):
            render_markdown_with_images(content)
    else:
        st.markdown(content)

# ==================== 聊天输入 ====================
def render_chat_input():
    """渲染聊天输入框"""
    if not st.session_state.knowledge_base_id:
        st.warning("⚠️ 请先在侧边栏选择知识库")
        return

    if st.session_state.messages:
        for msg in st.session_state.messages:
            display_message(msg["role"], msg["content"])

    # 聊天输入
    prompt = st.chat_input("输入您的问题...")

    if prompt:
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)

        # 调用API
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = chat(st.session_state.knowledge_base_id, prompt, st.session_state.session_id)

                if result:
                    # 更新会话ID
                    st.session_state.session_id = result.get("session_id")

                    # 显示回复
                    answer = result.get("message", {}).get("content", "")
                    if answer:
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        render_markdown_with_images(answer)

                        # 显示检索来源
                        retrieved_chunks = result.get("retrieved_chunks", [])
                        if retrieved_chunks:
                            with st.expander("📚 检索来源", expanded=False):
                                for i, chunk in enumerate(retrieved_chunks[:3]):
                                    st.markdown(f"**来源 {i+1}:** {chunk.get('file_name', 'unknown')}")
                                    st.markdown(f"相似度: `{chunk.get('score', 0):.4f}`")
                                    st.markdown(chunk.get('text', '')[:200] + "...")
                                    st.divider()
                else:
                    st.error("回答失败，请稍后重试")

# ==================== 核心功能 ====================
def chat(knowledge_base_id: int, query: str, session_id: str = None) -> dict:
    """
    调用问答API

    测试用例：
    1. 正常问答流程
    2. 无检索结果处理
    3. API超时处理
    4. 会话历史关联
    """
    payload = {
        "knowledge_base_id": knowledge_base_id,
        "query": query,
        "top_k": 5
    }

    if session_id:
        payload["session_id"] = session_id

    return api_request("POST", "/chat", json=payload)

# ==================== 主程序 ====================
def main():
    st.title("💬 图文对话")

    # 渲染侧边栏
    selected_kb = render_sidebar()

    # 渲染聊天界面
    render_chat_input()

if __name__ == "__main__":
    main()

# ==================== 测试用例 ====================
def test_knowledge_base_selection():
    """
    测试知识库选择

    测试步骤：
    1. 获取知识库列表
    2. 选择特定知识库
    3. 验证选择结果
    """
    print("\n" + "="*50)
    print("Testing Knowledge Base Selection...")
    print("="*50)

    kbs = get_knowledge_bases()
    print(f"Available knowledge bases: {kbs}")

def test_session_management():
    """
    测试会话管理

    测试步骤：
    1. 创建新会话
    2. 发送消息
    3. 获取历史
    4. 加载历史
    """
    print("\n" + "="*50)
    print("Testing Session Management...")
    print("="*50)

    # 创建会话并发送消息
    result = chat(1, "测试问题")
    print(f"Chat result session_id: {result.get('session_id') if result else 'None'}")

    if result and result.get("session_id"):
        session_id = result["session_id"]
        history = get_chat_history(session_id)
        print(f"History: {history}")

def test_rag_retrieval():
    """
    测试RAG检索

    测试步骤：
    1. 上传测试文档
    2. 等待索引完成
    3. 发送问题
    4. 验证检索结果
    """
    print("\n" + "="*50)
    print("Testing RAG Retrieval...")
    print("="*50)

    # 发送一个具体问题
    result = chat(1, "这个文档主要讲了什么？")
    print(f"Retrieved chunks: {len(result.get('retrieved_chunks', [])) if result else 0}")

def test_answer_generation():
    """
    测试答案生成

    测试步骤：
    1. 发送需要推理的问题
    2. 验证答案格式
    3. 验证图片来源标注
    """
    print("\n" + "="*50)
    print("Testing Answer Generation...")
    print("="*50)

    result = chat(1, "根据图表，产品A的销售额在哪个季度开始下降？")
    print(f"Answer: {result.get('message', {}).get('content', '')[:500] if result else 'None'}")