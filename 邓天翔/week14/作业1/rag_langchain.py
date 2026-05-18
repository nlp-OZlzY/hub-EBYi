import yaml  # type: ignore
from typing import Union, List, Any, Dict

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

import numpy as np
import datetime
import pdfplumber  # 导入pdfplumber模块，用于处理PDF文件

import torch  # type: ignore
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
# from FlagEmbedding import FlagReranker
from es_api import es

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

device = config["device"]

EMBEDDING_MODEL_PARAMS: Dict[Any, Any] = {}

BASIC_QA_TEMPLATE = '''现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
如果问题可以从资料中获得，则请逐步回答。

资料：
{#RELATED_DOCUMENT#}


问题：{#QUESTION#}
'''


def load_embdding_model(model_name: str, model_path: str) -> None:
    """
    加载编码模型
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """
    global EMBEDDING_MODEL_PARAMS
    # sbert模型
    if model_name in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
        EMBEDDING_MODEL_PARAMS["embedding_model"] = SentenceTransformer(model_path)


def load_rerank_model(model_name: str, model_path: str) -> None:
    """
    加载重排序模型
    :param model_name: 模型名称
    :param model_path: 模型路径
    :return:
    """
    global EMBEDDING_MODEL_PARAMS
    if model_name in ["bge-reranker-base"]:
        EMBEDDING_MODEL_PARAMS["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_tokenizer"] = AutoTokenizer.from_pretrained(model_path)
        EMBEDDING_MODEL_PARAMS["rerank_model"].eval()
        EMBEDDING_MODEL_PARAMS["rerank_model"].to(device)


if config["rag"]["use_embedding"]:
    model_name = config["rag"]["embedding_model"]
    model_path = config["models"]["embedding_model"][model_name]["local_url"]

    print(f"Loading embedding model {model_name} from model_path...")
    load_embdding_model(model_name, model_path)

if config["rag"]["use_rerank"]:
    model_name = config["rag"]["rerank_model"]
    model_path = config["models"]["rerank_model"][model_name]["local_url"]

    print(f"Loading rerank model {model_name} from model_path...")
    load_rerank_model(model_name, model_path)


def split_text_with_overlap(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = start + chunk_size - chunk_overlap
    return chunks


class RAG:
    def __init__(self):
        self.embedding_model = config["rag"]["embedding_model"]
        self.rerank_model = config["rag"]["rerank_model"]

        self.use_rerank = config["rag"]["use_rerank"]

        self.embedding_dims = config["models"]["embedding_model"][
            config["rag"]["embedding_model"]
        ]["dims"]

        self.chunk_size = config["rag"]["chunk_size"]
        self.chunk_overlap = config["rag"]["chunk_overlap"]
        self.chunk_candidate = config["rag"]["chunk_candidate"]

        self.llm_model = config["rag"]["llm_model"]

        # LangChain: 初始化Embedding模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config["models"]["embedding_model"][self.embedding_model]["local_url"],
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # LangChain: 初始化向量存储
        self.vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=self.embeddings
        )

        # LangChain: 初始化LLM
        self.llm = ChatOpenAI(
            model=self.llm_model,
            api_key=config["rag"]["llm_api_key"],
            base_url=config["rag"]["llm_base"],
            temperature=0.1,
            max_tokens=1024
        )

        # LangChain: RAG提示词模板
        self.rag_prompt = ChatPromptTemplate.from_template(
            '''现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
            如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
            如果问题可以从资料中获得，则请逐步回答。

            资料：
            {context}

            问题：{question}'''
        )

        # LangChain: 构建RAG链
        self._build_chain()

    def _build_chain(self):
        """构建LangChain RAG链"""
        def get_context(question: str) -> str:
            docs = self.vectorstore.similarity_search(question, k=self.chunk_candidate)
            return "\n\n".join([doc.page_content for doc in docs])

        self.chain = (
            {"context": lambda x: get_context(x), "question": RunnablePassthrough()}
            | self.rag_prompt
            | self.llm
        )

    def _extract_pdf_content(self, knowledge_id, document_id, title, file_path) -> bool:
        try:
            pdf = pdfplumber.open(file_path)
        except:
            print("打开文件失败")
            return False

        print(f"{file_path} pages: ", len(pdf.pages))  # 打印提示信息，显示PDF文件的页数

        abstract = ""

        for page_number in range(len(pdf.pages)):  # 每一页 提取
            current_page_text = pdf.pages[page_number].extract_text()  # 提取图片
            if page_number <= 3:
                abstract = abstract + '\n' + current_page_text

            # 每一页内容的内容
            embedding_vector = self.get_embedding(current_page_text)
            page_data = {
                "document_id": document_id,
                "knowledge_id": knowledge_id,
                "page_number": page_number,
                "chunk_id": 0,  # 先存储每一也所有内容
                "chunk_content": current_page_text,
                "chunk_images": [],
                "chunk_tables": [],
                "embedding_vector": embedding_vector
            }
            response = es.index(index="chunk_info", document=page_data)

            # 划分chunk
            page_chunks = split_text_with_overlap(current_page_text, self.chunk_size, self.chunk_overlap)
            embedding_vector = self.get_embedding(page_chunks)
            for chunk_idx in range(1, len(page_chunks) + 1):
                page_data = {
                    "document_id": document_id,
                    "knowledge_id": knowledge_id,
                    "page_number": page_number,
                    "chunk_id": chunk_idx,
                    "chunk_content": page_chunks[chunk_idx - 1],
                    "chunk_images": [],
                    "chunk_tables": [],
                    "embedding_vector": embedding_vector[chunk_idx - 1]
                }
                response = es.index(index="chunk_info", document=page_data)

        document_data = {
            "document_id": document_id,
            "knowledge_id": knowledge_id,
            "document_name": title,
            "file_path": file_path,
            "abstract": abstract
        }
        response = es.index(index="document_meta", document=document_data)

    def _extract_word_content():
        pass

    def extract_content(self, knowledge_id, document_id, title, file_type, file_path):
        if "pdf" in file_type:
            self._extract_pdf_content(knowledge_id, document_id, title, file_path)
        elif "word" in file_type:
            pass

        print("提取完成", document_id, file_type, file_path)

    def get_embedding(self, text) -> np.ndarray:
        """
        对文本进行编码
        :param text: 待编码文本
        :return: 编码结果
        """
        if self.embedding_model in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
            return EMBEDDING_MODEL_PARAMS["embedding_model"].encode(text, normalize_embeddings=True)

        raise NotImplemented

    def get_rank(self, text_pair) -> np.ndarray:
        """
        对文本对进行重排序
        :param text_pair: 待排序文本
        :return: 匹配打分结果
        """
        if self.rerank_model in ["bge-reranker-base"]:
            with torch.no_grad():
                inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                    text_pair, padding=True, truncation=True,
                    return_tensors='pt', max_length=512,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                scores = EMBEDDING_MODEL_PARAMS["rerank_model"](**inputs, return_dict=True).logits.view(-1, ).float()
                scores = scores.data.cpu().numpy()
                return scores

        raise NotImplemented

    def query_document(self, query: str) -> List[Any]:
        """LangChain: 文档检索（语义检索）"""
        docs = self.vectorstore.similarity_search(query, k=self.chunk_candidate)
        return docs

    def chat_with_rag(
            self,
            knowledge_id: int,  # 知识库 哪一个知识库提问
            messages: List[Dict],
    ):
        # 用户的第一次提问用rag
        if len(messages) == 1:
            query = messages[0]["content"]
            # LangChain: 使用RAG链生成回答
            rag_response = self.chain.invoke(query).content
            messages.append({"role": "system", "content": rag_response})
        # 后序提问 直接大模型回答
        else:
            normal_response = self.llm.invoke(messages).content
            messages.append({"role": "system", "content": normal_response})

        return messages

    def chat(self, messages: List[Dict]) -> Any:
        return self.llm.invoke(messages)

    def query_parse(self, query: str) -> str:
        return ""

    def query_rewrite(self, query: str) -> str:
        return ""
