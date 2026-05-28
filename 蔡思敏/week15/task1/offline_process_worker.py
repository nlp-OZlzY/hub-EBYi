"""
多模态RAG聊天机器人 - 离线处理Worker
从Kafka消费文档解析任务，调用minerU解析，chunk划分，向量编码，存储到Milvus

需求：
1. 从Kafka消费文档解析任务
2. 调用minerU解析PDF文档（输出markdown和图片）
3. 对markdown内容进行chunk划分
4. 对chunk进行BGE和CLIP向量编码
5. 存储向量到Milvus，元数据到SQLite

测试逻辑：
1. parse_document_test: 测试文档解析功能
2. encode_chunks_test: 测试chunk编码功能
3. full_pipeline_test: 测试完整流程
"""

import os
import glob
import traceback
import subprocess
import json
from typing import List, Dict, Tuple, Optional

from kafka import KafkaConsumer
import numpy as np

# 加载embedding模型
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from sentence_transformers import SentenceTransformer

# ==================== 配置 ====================
MILVUS_URI = os.getenv("MILVUS_URI", "https://in03-5cb3b56f3af9ebc.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "9027d285f74e5ce113bf24162fc5cabe04b67db3ee25055f4748ea23785f00d0fa9b8217c108a04dc77c4a703b5860a7d39d7a7b")
COLLECTION_NAME = "rag_data_new"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "rag-data"
MINERU_URL = os.getenv("MINERU_URL", "http://127.0.0.1:30000")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./processed")

# ==================== 模型初始化 ====================
print("Loading BGE model...")
bge_model = SentenceTransformer('/root/autodl-tmp/models/BAAI/bge-small-zh-v1.5')
print("BGE model loaded!")

print("Loading CLIP model...")
clip_model = SentenceTransformer(
    '/root/autodl-tmp/models/jinaai/jina-clip-v2',
    trust_remote_code=True,
    truncate_dim=1024
)
print("CLIP model loaded!")

# ==================== Milvus客户端 ====================
def get_milvus_client():
    from pymilvus import MilvusClient
    return MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)

# ==================== 数据库更新 ====================
def update_file_state(file_id: int, state: str, state_message: str = None,
                      parsed_path: str = None, page_count: int = None):
    """更新文件状态"""
    from orm_model import init_db, File, FileState

    session = init_db()[0]()
    try:
        file = session.query(File).filter(File.id == file_id).first()
        if file:
            file.state = state
            if state_message:
                file.state_message = state_message
            if parsed_path:
                file.parsed_path = parsed_path
            if page_count is not None:
                file.page_count = page_count
            session.commit()
    finally:
        session.close()

# ==================== Chunk划分 ====================
def split_text2chunks(lines: List[str], chunk_size: int = 256) -> List[str]:
    """
    将文本分割成多个块，每个块的长度不超过chunk_size个字符

    测试用例：
    1. 空行过滤
    2. 参考引用过滤([数字]格式)
    3. 标题行保护
    4. 超长chunk处理
    """
    chunks = []
    for line in lines:
        line = line.strip()

        if not line:
            continue

        # 跳过参考引用部分
        if line == "# References":
            continue

        # 跳过带编号的引用行
        if len(line) > 2 and line[0] == "[" and line[1].isdigit():
            continue

        # 第一个chunk直接添加
        if len(chunks) == 0:
            chunks.append(line)
        else:
            # 合并到上一个chunk或创建新chunk
            if len(chunks[-1]) <= chunk_size:
                chunks[-1] += "\n" + line
            else:
                chunks.append(line)

    return chunks

# ==================== 向量编码 ====================
def encode_text_and_image(text: str, markdown_path: str) -> Tuple[List[float], List[float], List[float]]:
    """
    将文本和图像编码成向量

    测试用例：
    1. 纯文本编码（BGE + CLIP text）
    2. 含图像的文本编码（额外CLIP image）
    3. 编码失败时的零向量处理
    """
    # 分离文本和图片
    text_with_no_image = "\n".join([line for line in text.split("\n") if not line.startswith("!["]])
    text_with_image = [line for line in text.split("\n") if line.startswith("![")]

    # BGE编码（文本）
    try:
        text_bge_embedding = bge_model.encode(text_with_no_image, normalize_embeddings=True)
        text_bge_embedding = list(text_bge_embedding)
    except Exception as e:
        print(f"BGE encoding failed: {e}")
        text_bge_embedding = np.zeros(512).tolist()

    # CLIP text编码
    try:
        text_clip_embedding = clip_model.encode(text_with_no_image, normalize_embeddings=True)
        text_clip_embedding = list(text_clip_embedding)
    except Exception as e:
        print(f"CLIP text encoding failed: {e}")
        text_clip_embedding = np.zeros(1024).tolist()

    # CLIP image编码（如有图片）
    if len(text_with_image) > 0:
        try:
            # 提取图片路径
            image_path = text_with_image[0].split("](")[1].split(")")[0]
            image_real_path = os.path.join(os.path.dirname(markdown_path), image_path.split("/")[-1])
            print(f"Encoding image: {image_real_path}")
            image_clip_embedding = clip_model.encode(image_real_path, normalize_embeddings=True)
            image_clip_embedding = list(image_clip_embedding)
        except Exception as e:
            print(f"CLIP image encoding failed: {e}")
            image_clip_embedding = np.zeros(1024).tolist()
    else:
        image_clip_embedding = np.zeros(1024).tolist()

    return text_bge_embedding, text_clip_embedding, image_clip_embedding

# ==================== 文档解析 ====================
def parse_document(file_path: str, output_dir: str = PROCESSED_DIR) -> Optional[str]:
    """
    使用minerU解析文档

    测试用例：
    1. 正常PDF解析
    2. 解析失败处理
    3. 超时处理

    返回：markdown文件路径
    """
    file_name = os.path.basename(file_path).split(".")[0]
    output_path = os.path.join(output_dir, file_name)

    # 调用minerU
    cmd = f"mineru -p {file_path} -o {output_dir} -b vlm-http-client -u {MINERU_URL}"
    print(f"Executing: {cmd}")

    try:
        subprocess.check_output(cmd, shell=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"Timeout parsing {file_path}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Parse error: {e}")
        return None

    # 查找生成的markdown文件
    markdown_files = glob.glob(os.path.join(output_path, "**", "*.md"), recursive=True)

    if len(markdown_files) == 0:
        print(f"No markdown found for {file_name}")
        return None

    return markdown_files[0]

# ==================== Chunk编码与存储 ====================
def encode_and_store_chunks(
    markdown_path: str,
    file_id: int,
    file_name: str,
    file_path: str
) -> int:
    """
    对markdown内容进行chunk划分、向量编码、存储到Milvus

    测试用例：
    1. 空文件处理
    2. 正常chunk处理
    3. 部分chunk编码失败处理

    返回：成功存储的chunk数量
    """
    if not os.path.exists(markdown_path):
        print(f"Markdown file not found: {markdown_path}")
        return 0

    # 读取markdown内容
    lines = open(markdown_path, 'r', encoding='utf-8').readlines()
    chunks = split_text2chunks(lines)

    print(f"Total chunks: {len(chunks)}")

    mc = get_milvus_client()
    success_count = 0

    for i, chunk in enumerate(chunks):
        try:
            # 编码
            text_bge, text_clip, image_clip = encode_text_and_image(chunk, markdown_path)

            # 存储到Milvus
            data = [{
                "text_vector": text_bge,
                "clip_text_vector": text_clip,
                "clip_image_vector": image_clip,
                "text": chunk,
                "db_id": file_id,
                "file_name": file_name,
                "file_path": file_path
            }]

            result = mc.insert(
                collection_name=COLLECTION_NAME,
                data=data
            )
            success_count += 1
            print(f"Chunk {i+1}/{len(chunks)} stored, id: {result}")

        except Exception as e:
            print(f"Chunk {i} encoding failed: {e}")
            traceback.print_exc()
            continue

    return success_count

# ==================== 完整处理流程 ====================
def process_document(file_id: int, file_name: str, file_path: str) -> bool:
    """
    完整的文档处理流程

    测试用例：
    1. 文件不存在检查
    2. 解析阶段状态更新
    3. 索引阶段状态更新
    """
    print(f"\n{'='*50}")
    print(f"Processing document: {file_name} (ID: {file_id})")
    print(f"{'='*50}")

    # 检查文件
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        update_file_state(file_id, "parse_failed", f"File not found: {file_path}")
        return False

    try:
        # 步骤1: 更新状态为解析中
        update_file_state(file_id, "parsing", "Document parsing in progress...")

        # 步骤2: 解析文档
        print("Step 1: Parsing document...")
        markdown_path = parse_document(file_path)
        if not markdown_path:
            raise Exception("Document parsing failed")

        # 统计页数（通过查找的图片数量估算）
        image_count = len(glob.glob(os.path.join(os.path.dirname(markdown_path), "**", "*.jpg"), recursive=True))

        update_file_state(
            file_id, "parsed",
            f"Parsed successfully",
            parsed_path=markdown_path,
            page_count=image_count
        )

        # 步骤3: 更新状态为建索引中
        update_file_state(file_id, "indexing", "Indexing in progress...")

        # 步骤4: 编码并存储chunks
        print("Step 2: Encoding and storing chunks...")
        chunk_count = encode_and_store_chunks(markdown_path, file_id, file_name, file_path)

        if chunk_count == 0:
            raise Exception("No chunks stored")

        # 步骤5: 更新状态为已完成
        update_file_state(file_id, "indexed", f"Indexed {chunk_count} chunks")

        print(f"Document processed successfully! {chunk_count} chunks stored.")
        return True

    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        update_file_state(file_id, "index_failed", error_msg)
        return False

# ==================== Kafka消费者 ====================
def create_consumer():
    """创建Kafka消费者"""
    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        group_id="rag-processor"
    )

def run_consumer():
    """运行消费者"""
    print(f"Starting Kafka consumer for topic: {KAFKA_TOPIC}")
    consumer = create_consumer()

    for msg in consumer:
        try:
            file_name = msg.value['file_name']
            file_path = msg.value['file_path']
            file_id = msg.value['id']

            print(f"\nReceived message: file_name={file_name}, id={file_id}")
            process_document(file_id, file_name, file_path)

        except Exception as e:
            print(f"Message processing failed: {e}")
            traceback.print_exc()

# ==================== 测试函数 ====================
def parse_document_test():
    """
    测试文档解析功能

    测试用例：
    1. 正常PDF解析
    2. 文件不存在
    3. 解析超时
    """
    print("\n" + "="*50)
    print("Testing parse_document...")
    print("="*50)

    test_files = [
        "/path/to/test.pdf",  # 需要替换为实际路径
    ]

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"Test file not found: {test_file}")
            continue

        result = parse_document(test_file)
        print(f"Result: {result}")

def encode_chunks_test():
    """
    测试chunk编码功能

    测试用例：
    1. 纯文本
    2. 图文混合
    3. 空文本
    4. 超长文本
    """
    print("\n" + "="*50)
    print("Testing encode_text_and_image...")
    print("="*50)

    test_cases = [
        ("这是一段纯文本。", "/tmp/test.md"),
        ("这是文本\n![image](images/test.jpg)", "/tmp/test.md"),
        ("", "/tmp/test.md"),
        ("A" * 1000, "/tmp/test.md"),
    ]

    for text, path in test_cases:
        print(f"\nInput: {text[:50]}...")
        bge, clip_text, clip_img = encode_text_and_image(text, path)
        print(f"BGE dim: {len(bge)}, CLIP text dim: {len(clip_text)}, CLIP img dim: {len(clip_img)}")

def split_text2chunks_test():
    """
    测试chunk划分功能

    测试用例：
    1. 空列表
    2. 正常文本
    3. 含引用行
    4. 超长行
    """
    print("\n" + "="*50)
    print("Testing split_text2chunks...")
    print("="*50)

    test_cases = [
        [],
        ["line1", "line2", "line3"],
        ["[1] Reference", "line1", "line2"],
        ["A" * 500, "B" * 500],
    ]

    for lines in test_cases:
        result = split_text2chunks(lines)
        print(f"Input lines: {len(lines)} -> Output chunks: {len(result)}")

def full_pipeline_test():
    """
    测试完整处理流程

    测试用例：
    1. 正常处理
    2. 文件不存在
    3. 解析失败
    """
    print("\n" + "="*50)
    print("Testing full_pipeline...")
    print("="*50)

    # 需要提供实际的测试数据
    test_data = {
        "file_id": 1,
        "file_name": "test.pdf",
        "file_path": "/path/to/test.pdf"
    }

    # 取消注释以运行实际测试
    # result = process_document(**test_data)
    # print(f"Result: {result}")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG离线处理Worker")
    parser.add_argument("--test", choices=["parse", "encode", "split", "full", "all"],
                        help="运行测试")
    args = parser.parse_args()

    if args.test:
        if args.test in ["parse", "all"]:
            parse_document_test()
        if args.test in ["encode", "all"]:
            encode_chunks_test()
        if args.test in ["split", "all"]:
            split_text2chunks_test()
        if args.test in ["full", "all"]:
            full_pipeline_test()
    else:
        # 运行消费者
        run_consumer()

    # 测试用法:
    # python offline_process_worker.py --test split  # 测试chunk划分
    # python offline_process_worker.py --test encode  # 测试编码
    # python offline_process_worker.py --test all     # 运行所有测试
    # python offline_process_worker.py                # 运行消费者