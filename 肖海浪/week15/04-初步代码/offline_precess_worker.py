"""
离线处理Worker服务

负责：
1. 消费Kafka消息
2. 调用mineru解析PDF文档
3. 使用bge/clip模型编码文本和图片
4. 将向量存储到Milvus

运行方式：python offline_precess_worker.py

为什么用Worker？
- 文档解析耗时较长（可能1分钟）
- 同步处理会让用户等待
- Worker在后台异步处理，用户无感知
"""
import os
import glob
import json
import logging
import subprocess
from typing import List
from kafka import KafkaConsumer
from orm_model import File, Session
from services.encode_service import EncodeService
from services.search_service import SearchService
from utils.text_splitter import split_text2chunks
from config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OfflineWorker:
    """
    离线处理Worker类

    职责：
    - 监听Kafka消息队列
    - 处理文档解析任务
    - 更新文件状态
    """

    def __init__(self):
        """
        初始化Worker

        1. 创建编码服务
        2. 创建检索服务
        3. 创建Kafka消费者
        """
        # 编码服务：将文本/图片转为向量
        self.encode_service = EncodeService()

        # 检索服务：操作Milvus向量数据库
        self.search_service = SearchService()

        # Kafka消费者：监听消息队列
        self.consumer = KafkaConsumer(
            settings.KAFKA_TOPIC,                    # 监听的主题
            bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,  # Kafka地址
            enable_auto_commit=True,                 # 自动提交offset
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),  # 反序列化
        )
        logger.info("Worker初始化完成，等待消息...")

    def update_file_state(self, file_id: int, state: str):
        """
        更新文件状态

        Args:
            file_id: 文件ID
            state: 新状态，如 "解析中" / "已解析" / "解析失败"

        状态流转：
        已上传 → 解析中 → 已解析
                        → 解析失败
        """
        with Session() as session:
            file = session.query(File).filter(File.id == file_id).first()
            if file:
                file.filestate = state
                session.commit()
                logger.info(f"文件状态更新: file_id={file_id}, state={state}")

    def parse_document(self, file_path: str) -> str:
        """
        解析文档

        Args:
            file_path: PDF文件路径

        Returns:
            解析后的markdown文件路径

        流程：
        1. 调用mineru命令行工具
        2. 解析PDF为markdown + 图片
        3. 返回markdown文件路径
        """
        logger.info(f"开始解析文档: {file_path}")

        # 调用mineru解析PDF
        # -p: 输入文件路径
        # -o: 输出目录
        # -b: 使用的解析后端
        # -u: VLM服务地址
        cmd = f"mineru -p {file_path} -o ./processed -b vlm-http-client -u http://127.0.0.1:30000"

        # 执行命令，超时600秒
        subprocess.check_output(cmd, shell=True, timeout=600)

        # 查找解析后的markdown文件
        # mineru会在 ./processed/文件名/ 目录下生成markdown
        file_name = os.path.basename(file_path).split(".")[0]
        markdown_pattern = os.path.join("./processed", file_name, "**/**.md")
        markdown_files = glob.glob(markdown_pattern, recursive=True)

        if len(markdown_files) == 0:
            raise Exception(f"未找到解析后的markdown文件: {file_name}")

        logger.info(f"文档解析完成: {markdown_files[0]}")
        return markdown_files[0]

    def encode_and_store(self, markdown_path: str, file_id: int, file_name: str, file_path: str):
        """
        编码文档并存储到Milvus

        Args:
            markdown_path: markdown文件路径
            file_id: 文件ID
            file_name: 文件名
            file_path: 原始文件路径

        流程：
        1. 读取markdown内容
        2. 文本切分为chunks
        3. 对每个chunk编码
        4. 存储到Milvus
        """
        logger.info(f"开始编码文档: {markdown_path}")

        # 读取markdown内容
        with open(markdown_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 文本切分
        # 将长文本切分为多个小块，每块不超过256字符
        chunks = split_text2chunks(lines, chunk_size=settings.CHUNK_SIZE)
        logger.info(f"文本切分完成: {len(chunks)} 个chunks")

        # 对每个chunk编码并存储
        for i, chunk in enumerate(chunks):
            try:
                # 编码：文本 → bge向量 + clip文本向量 + clip图片向量
                text_bge, text_clip, image_clip = self.encode_service.encode_text_and_image(
                    chunk, markdown_path
                )

                # 构造Milvus数据
                data = [{
                    "text_vector": text_bge,         # bge编码的文本向量 (512维)
                    "clip_text_vector": text_clip,   # clip编码的文本向量 (1024维)
                    "clip_image_vector": image_clip, # clip编码的图片向量 (1024维)
                    "text": chunk,                   # 原始文本
                    "db_id": file_id,                # 关联的文件ID
                    "file_name": file_name,          # 文件名
                    "file_path": file_path           # 文件路径
                }]

                # 存储到Milvus
                self.search_service.client.insert(
                    collection_name=settings.MILVUS_COLLECTION,
                    data=data
                )

                logger.info(f"编码完成: chunk {i+1}/{len(chunks)}")

            except Exception as e:
                logger.error(f"编码chunk失败: {e}")

    def process_message(self, message: dict):
        """
        处理Kafka消息

        Args:
            message: 消息内容，包含 file_name, file_path, id

        流程：
        1. 更新状态为"解析中"
        2. 解析文档
        3. 编码并存储
        4. 更新状态为"已解析"
        """
        file_name = message['file_name']
        file_path = message['file_path']
        file_id = message['id']

        logger.info(f"开始处理文件: {file_name} (id={file_id})")

        # 更新状态为解析中
        self.update_file_state(file_id, "解析中")

        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise Exception(f"文件不存在: {file_path}")

            # 解析文档
            markdown_path = self.parse_document(file_path)

            # 编码并存储到Milvus
            self.encode_and_store(markdown_path, file_id, file_name, file_path)

            # 更新状态为已解析
            self.update_file_state(file_id, "已解析")
            logger.info(f"文件处理完成: {file_name}")

        except Exception as e:
            logger.error(f"文件处理失败: {e}")
            # 更新状态为解析失败
            self.update_file_state(file_id, "解析失败")

    def run(self):
        """
        启动Worker，持续监听Kafka消息

        这是一个无限循环，会一直运行
        每收到一条消息，就调用 process_message 处理
        """
        logger.info("Worker启动，等待消息...")

        # 持续消费消息
        for message in self.consumer:
            try:
                # 处理消息
                self.process_message(message.value)
            except Exception as e:
                logger.error(f"处理消息失败: {e}")


def main():
    """主函数"""
    worker = OfflineWorker()
    worker.run()


if __name__ == "__main__":
    main()
