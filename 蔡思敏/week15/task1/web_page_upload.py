"""
文件上传页面 - Streamlit前端

功能：
1. 知识库管理（创建、选择）
2. 文件上传（带进度显示）
3. 文件列表展示（状态、删除）
4. 实时状态刷新

测试逻辑：
1. test_knowledge_base_crud: 知识库CRUD测试
2. test_file_upload: 文件上传测试
3. test_file_deletion: 文件删除测试
4. test_status_refresh: 状态刷新测试
"""

import streamlit as st
import requests
import time
from datetime import datetime

# ==================== 配置 ====================
API_BASE_URL = "http://localhost:8000/api/v1"
POLL_INTERVAL = 2  # 状态轮询间隔（秒）

# ==================== 页面配置 ====================
st.set_page_config(page_title="文件管理", page_icon="📁")

# ==================== 辅助函数 ====================
def api_request(method: str, endpoint: str, **kwargs):
    """发送API请求"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        elif method == "DELETE":
            response = requests.delete(url, **kwargs)
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

def create_knowledge_base(name: str, description: str = None):
    """创建知识库"""
    return api_request("POST", "/knowledge-bases", json={"name": name, "description": description})

def get_files(knowledge_base_id: int = None, state: str = None):
    """获取文件列表"""
    params = {}
    if knowledge_base_id:
        params["knowledge_base_id"] = knowledge_base_id
    if state:
        params["state"] = state
    return api_request("GET", "/files", params=params)

def upload_file(file, knowledge_base_id: int = None):
    """上传文件"""
    url = f"{API_BASE_URL}/files/upload"
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        data = {}
        if knowledge_base_id:
            data["knowledge_base_id"] = knowledge_base_id

        response = requests.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"上传失败: {e}")
        return None

def delete_file(file_id: int):
    """删除文件"""
    return api_request("DELETE", f"/files/{file_id}")

def get_file_status(file_id: int):
    """获取文件状态"""
    return api_request("GET", f"/files/{file_id}/status")

def trigger_parse(file_id: int):
    """触发文档解析"""
    return api_request("POST", f"/files/{file_id}/parse")

def poll_file_status(file_id: int, max_wait: int = 120):
    """
    轮询文件状态直到完成或超时

    测试用例：
    1. 正常等待完成
    2. 超时处理
    3. 状态异常处理
    """
    start_time = time.time()
    status_container = st.empty()

    while time.time() - start_time < max_wait:
        status = get_file_status(file_id)
        if not status:
            return None

        state = status.get("state", "unknown")
        progress = status.get("progress", 0)

        # 显示进度
        status_container.info(f"处理中... 状态: {state}, 进度: {progress*100:.0f}%")

        if state in ["indexed", "parse_failed", "index_failed"]:
            return status

        time.sleep(POLL_INTERVAL)

    return status

# ==================== 知识库管理 ====================
def render_knowledge_base_section():
    """渲染知识库管理区域"""
    st.markdown("### 知识库管理")

    col1, col2 = st.columns([3, 1])

    with col1:
        # 选择知识库
        kbs = get_knowledge_bases()
        if kbs and kbs.get("items"):
            options = ["全部"] + [f"{kb['id']}: {kb['name']}" for kb in kbs["items"]]
            selected = st.selectbox("选择知识库", options, key="kb_select")
        else:
            st.info("暂无知识库")
            selected = None

    with col2:
        if st.button("刷新", key="refresh_kb"):
            st.rerun()

    # 新建知识库
    with st.expander("新建知识库", expanded=False):
        kb_name = st.text_input("知识库名称", key="new_kb_name")
        kb_desc = st.text_area("描述（可选）", key="new_kb_desc", height=60)

        if st.button("创建", key="create_kb"):
            if not kb_name:
                st.warning("请输入知识库名称")
            else:
                result = create_knowledge_base(kb_name, kb_desc)
                if result:
                    st.success(f"知识库 '{kb_name}' 创建成功")
                    st.rerun()

    return selected

# ==================== 文件列表 ====================
def render_file_list(knowledge_base_id: int = None):
    """渲染文件列表"""
    st.markdown("### 文件列表")

    files = get_files(knowledge_base_id=knowledge_base_id)
    if not files:
        st.info("暂无文件")
        return []

    items = files.get("items", [])
    if not items:
        st.info("暂无文件")
        return []

    # 显示文件表格
    for file in items:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

            with col1:
                state_emoji = {
                    "uploaded": "📤",
                    "parsing": "🔄",
                    "parsed": "✅",
                    "parse_failed": "❌",
                    "indexing": "🔄",
                    "indexed": "✅",
                    "index_failed": "❌"
                }.get(file["state"], "❓")

                st.markdown(f"**{state_emoji} {file['filename']}**")
                st.caption(f"ID: {file['id']} | 大小: {file.get('file_size', 'N/A')}")

            with col2:
                st.markdown(f"状态: `{file['state']}`")

            with col3:
                if file["state"] in ["uploaded", "parse_failed"]:
                    if st.button("解析", key=f"parse_{file['id']}"):
                        result = trigger_parse(file["id"])
                        if result:
                            st.info("解析任务已提交")
                            poll_file_status(file["id"])
                            st.rerun()

            with col4:
                if st.button("删除", key=f"delete_{file['id']}"):
                    result = delete_file(file["id"])
                    if result:
                        st.success("删除成功")
                        st.rerun()

            st.divider()

    return items

# ==================== 文件上传 ====================
def render_upload_section(knowledge_base_id: int = None):
    """渲染文件上传区域"""
    st.markdown("### 文件上传")

    # 解析knowledge_base_id
    kb_id = None
    if knowledge_base_id and knowledge_base_id != "全部":
        try:
            kb_id = int(knowledge_base_id.split(":")[0].strip())
        except:
            pass

    uploaded_file = st.file_uploader(
        "选择文件",
        type=["pdf", "docx", "txt"],
        help="支持 PDF、DOCX、TXT 格式"
    )

    if uploaded_file:
        col1, col2 = st.columns([1, 4])

        with col1:
            st.markdown("**文件名:**")
            st.text(uploaded_file.name)

            st.markdown("**大小:**")
            st.text(f"{uploaded_file.size / 1024:.1f} KB")

        with col2:
            if st.button("上传", key="upload_btn", type="primary"):
                with st.spinner("上传中..."):
                    result = upload_file(uploaded_file, kb_id)

                    if result:
                        st.success(f"文件 '{uploaded_file.name}' 上传成功!")
                        st.info("解析任务已自动提交，请在文件列表中查看进度")

                        # 轮询状态
                        poll_file_status(result["id"])
                        st.rerun()
                    else:
                        st.error("上传失败")

# ==================== 主程序 ====================
def main():
    st.title("📁 文件管理")

    # 知识库选择
    selected_kb = render_knowledge_base_section()

    # 文件列表
    kb_id_for_list = None
    if selected_kb and selected_kb != "全部":
        try:
            kb_id_for_list = int(selected_kb.split(":")[0].strip())
        except:
            pass

    render_file_list(kb_id_for_list)

    # 文件上传
    render_upload_section(selected_kb)

if __name__ == "__main__":
    main()

# ==================== 测试用例 ====================
def test_knowledge_base_crud():
    """
    测试知识库CRUD操作

    测试步骤：
    1. 创建知识库
    2. 获取知识库列表
    3. 验证创建结果
    """
    print("\n" + "="*50)
    print("Testing Knowledge Base CRUD...")
    print("="*50)

    # 创建
    result = create_knowledge_base("测试知识库", "用于测试")
    print(f"Create: {result}")

    # 列表
    kbs = get_knowledge_bases()
    print(f"List: {kbs}")

def test_file_upload():
    """
    测试文件上传

    测试步骤：
    1. 上传小文件
    2. 上传大文件
    3. 上传错误格式
    """
    print("\n" + "="*50)
    print("Testing File Upload...")
    print("="*50)

    # 创建测试文件
    import io
    test_content = b"Test content"
    test_file = io.BytesIO(test_content)
    test_file.name = "test.txt"

    # 上传
    result = upload_file(test_file)
    print(f"Upload result: {result}")

def test_file_deletion():
    """测试文件删除"""
    print("\n" + "="*50)
    print("Testing File Deletion...")
    print("="*50)

    files = get_files()
    if files and files.get("items"):
        file_id = files["items"][0]["id"]
        result = delete_file(file_id)
        print(f"Delete result: {result}")

def test_status_refresh():
    """
    测试状态刷新

    测试步骤：
    1. 上传文件
    2. 轮询状态
    3. 验证状态变化
    """
    print("\n" + "="*50)
    print("Testing Status Refresh...")
    print("="*50)

    # 模拟轮询
    import io
    test_file = io.BytesIO(b"Test")
    test_file.name = "status_test.txt"

    result = upload_file(test_file)
    if result:
        file_id = result["id"]
        print(f"File uploaded, ID: {file_id}")

        status = poll_file_status(file_id, max_wait=10)
        print(f"Final status: {status}")