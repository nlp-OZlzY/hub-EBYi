"""Streamlit page: multimodal RAG chat — ask questions and see answers with images."""

import logging
import os
import re

import openai
import streamlit as st
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

from config import (
    BGE_MODEL_PATH,
    DASHSCOPE_API_KEY,
    DEFAULT_TOP_K,
    HF_ENDPOINT,
    MILVUS_COLLECTION,
    MILVUS_TOKEN,
    MILVUS_URI,
    QWEN_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# ── RAG prompt ─────────────────────────────────────────────────────
RAG_PROMPT = """基于资料回答用户的问题：{0}

相关资料: {1}

回答要求：
- 回答要客观，有逻辑，要基于只有的资料。
- 如果资料中包含图片链接，则单独一行进行输出，保留图的原始链接，需要将图放在合适的相关内容的位置。
"""

# ── Models & clients ───────────────────────────────────────────────


@st.cache_resource
def load_bge_model() -> SentenceTransformer:
    logger.info("Loading BGE model from %s ...", BGE_MODEL_PATH)
    return SentenceTransformer(BGE_MODEL_PATH)


@st.cache_resource
def get_milvus_client() -> MilvusClient:
    logger.info("Connecting to Milvus ...")
    return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)


@st.cache_resource
def get_qwen_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


bge_model = load_bge_model()
milvus_client = get_milvus_client()
qwen_client = get_qwen_client()

# ── Helpers ────────────────────────────────────────────────────────


def clear_chat_history() -> None:
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是AI助手，可以直接与大模型对话也可以调用内部工具。"}
    ]
    st.session_state.session_id = None


def render_markdown_with_images(markdown_text: str) -> None:
    """Render markdown, displaying image references as actual images."""
    pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    last_pos = 0
    for match in pattern.finditer(markdown_text):
        st.markdown(markdown_text[last_pos : match.start()], unsafe_allow_html=True)
        img_url = match.group(1)
        if os.path.exists(img_url):
            st.image(img_url)
        last_pos = match.end()
    st.markdown(markdown_text[last_pos:], unsafe_allow_html=True)


# ── Session init ───────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是AI助手，可以直接与大模型对话也可以调用内部工具。"}
    ]

# ── Render history ─────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.button("清空当前聊天", on_click=clear_chat_history, use_container_width=True)

# ── Chat input ─────────────────────────────────────────────────────
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("搜索中..."):
            prompt_embedding = bge_model.encode(prompt, normalize_embeddings=True).tolist()
            results = milvus_client.search(
                collection_name=MILVUS_COLLECTION,
                data=[prompt_embedding],
                limit=DEFAULT_TOP_K,
                anns_field="text_vector",
                output_fields=["text", "db_id", "file_name", "file_path"],
            )

            related_parts: list[str] = []
            for hit in results[0]:
                entity = hit["entity"]
                text = entity.get("text", "")
                file_path = entity.get("file_path", "")
                file_dir = os.path.basename(file_path).split(".")[0] if file_path else ""
                if file_dir:
                    text = text.replace("images/", f"./processed/{file_dir}/vlm/images/")
                related_parts.append(text)

        with st.spinner("生成回答中..."):
            completion = qwen_client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": RAG_PROMPT.format(prompt, "\n".join(related_parts))},
                ],
            )
            answer = completion.choices[0].message.content or ""

        st.session_state.messages.append({"role": "assistant", "content": answer})
        render_markdown_with_images(answer)
