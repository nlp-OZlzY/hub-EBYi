"""
问答服务

这是RAG（检索增强生成）的核心服务，负责：
1. 将用户问题编码为向量
2. 在Milvus中检索相关文档
3. 拼接上下文和问题
4. 调用大模型生成回答

RAG流程：
  用户问题 → 编码 → 检索 → 拼接上下文 → LLM生成回答
"""
import logging
import openai
from typing import Dict, List
from config import settings
from services.encode_service import EncodeService
from services.search_service import SearchService

logger = logging.getLogger(__name__)

# ============================================
# RAG提示词模板
# {0} 会被替换为用户问题
# {1} 会被替换为检索到的相关资料
# ============================================
RAG_PROMPT = """基于资料回答的提问精简的回答下面的问题：{0}

相关资料: {1}

回答要求：
- 回答要客观，有逻辑，要基于只有的资料。
- 如果资料中包含图片链接，则单独一行进行输出，保留图的原始链接，不要修改任何连接路径，需要将图放在合适的相关内容的位置。
"""


class ChatService:
    """
    问答服务类

    职责：
    - 接收用户问题
    - 协调编码服务、检索服务、LLM服务
    - 返回答案和来源信息
    """

    def __init__(self):
        """初始化服务，创建编码、检索、LLM客户端"""
        # 编码服务：将文本转为向量
        self.encode_service = EncodeService()

        # 检索服务：在Milvus中搜索相似向量
        self.search_service = SearchService()

        # LLM客户端：调用Qwen-VL大模型
        # 使用OpenAI兼容的API格式
        self.llm_client = openai.OpenAI(
            api_key=settings.QWEN_API_KEY,
            base_url=settings.QWEN_BASE_URL,
        )

    def chat(self, question: str) -> Dict:
        """
        问答主流程

        Args:
            question: 用户问题，例如 "什么是RAG？"

        Returns:
            字典，包含：
            - answer: 生成的答案
            - sources: 来源信息列表

        示例返回：
        {
            "answer": "RAG是检索增强生成...",
            "sources": [
                {"file_name": "doc.pdf", "relevance": 0.85}
            ]
        }
        """
        # ============================================
        # 步骤1：将问题编码为向量
        # bge模型将文本转为512维向量
        # ============================================
        question_vector = self.encode_service.encode_text(question)
        logger.info(f"问题编码完成: {question[:50]}...")

        # ============================================
        # 步骤2：在Milvus中检索相似文档
        # 返回Top-5最相关的文档片段
        # ============================================
        search_results = self.search_service.search(question_vector, limit=5)
        logger.info(f"检索到 {len(search_results)} 条相关资料")

        # ============================================
        # 步骤3：拼接上下文
        # 将检索到的文档片段拼接成一个字符串
        # 同时处理图片路径替换
        # ============================================
        context_parts = []  # 存储文本内容
        sources = []        # 存储来源信息

        for result in search_results:
            # 获取文本内容
            text = result["text"]

            # 替换图片路径
            # 原始路径：images/fig.png
            # 替换后：./processed/文件名/vlm/images/fig.png
            file_dir = result["file_path"].split("/")[-1].split(".")[0]
            text = text.replace("images/", f"./processed/{file_dir}/vlm/images/")

            # 添加到上下文
            context_parts.append(text)

            # 记录来源信息
            # relevance = 1 - distance，距离越小，相关性越高
            sources.append({
                "file_name": result["file_name"],
                "relevance": 1 - result["distance"]
            })

        # 将所有文本片段用换行符连接
        context = "\n".join(context_parts)

        # ============================================
        # 步骤4：调用LLM生成回答
        # 将问题和上下文发送给Qwen-VL
        # ============================================
        try:
            completion = self.llm_client.chat.completions.create(
                model=settings.QWEN_MODEL,  # 使用的模型
                messages=[
                    # 系统消息：定义AI助手的角色
                    {'role': 'system', 'content': 'You are a helpful assistant.'},
                    # 用户消息：包含问题和检索到的资料
                    {'role': 'user', 'content': RAG_PROMPT.format(question, context)}
                ],
            )
            # 提取生成的回答
            answer = completion.choices[0].message.content
            logger.info("LLM回答生成完成")

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            answer = "抱歉，问答服务暂时不可用，请稍后再试。"

        # ============================================
        # 返回结果
        # ============================================
        return {
            "answer": answer,
            "sources": sources
        }
