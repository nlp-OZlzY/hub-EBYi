### 项目背景
我们正在构建一个**多模态 RAG（检索增强生成）系统**。这是一个生产级应用，核心需求是允许用户上传复杂的 PDF 文档（包含图表、图片），系统能解析并“看懂”这些图片，最后通过自然语言问答给出准确回答。

### 核心需求 (基于 test1.md 优化)
请基于以下技术栈，帮我实现后端接口和测试逻辑。

**技术栈约束**
- **编程语言**: Python 3.10+
- **Web 框架**: FastAPI (需要支持异步)
- **数据库**: 
    - 向量数据库: Milvus (用于存储文本和图片向量)
    - 关系型数据库: SQLite (使用 SQLAlchemy，用于存储文件元信息)
- **消息队列**: Kafka (用于解耦文件上传和耗时的解析过程)
- **核心模型**: 
    - 文档解析: MinerU (通过 API 调用，参考提供的代码片段)
    - 向量检索: BGE (文本), Jina-CLIP (图文)
    - 问答模型: Qwen-VL / Qwen-Plus
    - 对象存储: 七牛云 (Qiniu)

### 模块一：接口实现 (Implementation)

请帮我生成 `main.py` 或 `api/routes.py` 的代码。需要实现以下两个核心接口：

**1. 文档上传接口 (POST /upload/document)**
- **功能逻辑**:
    1. 接收用户上传的 PDF 文件。
    2. 将文件保存到本地临时目录，并生成唯一 ID。
    3. 将文件元信息（文件名、路径、状态“上传中”）写入 SQLite 数据库。
    4. 将文件上传到七牛云对象存储。
    5. 向 Kafka Topic (`document_parse_topic`) 发送一条消息，通知后台 Worker 进行解析，有一个后台服务会消费 Kafka 消息，调用 MinerU，然后把结果存入 Milvus。
    6. 返回成功响应给前端。

**2. 聊天问答接口 (POST /chat)**
- **功能逻辑**:
    1. 接收用户提问 (query) 和 知识库 ID。
    2. 使用 `BGE` 模型对提问进行文本 Embedding。
    3. 在 Milvus 中进行混合检索（既要检索相关文本 Chunk，也要检索相关图片）。
    4. 将检索到的 Top-K 文本和图片（Base64 或 URL）与用户提问拼接，形成 Prompt。
    5. 调用 `Qwen-VL` 进行多模态推理，生成最终答案。
    6. 答案中必须包含信息来源（例如：来自哪个 PDF 的哪一页）。

### 模块二：测试逻辑 (Testing Logic) - 重点

我非常看重代码的可测试性。请按照以下要求为上述接口编写单元测试和集成测试：

**1. 测试策略**
- 使用 `pytest` 框架。
- 使用 `unittest.mock` 来 Mock 外部依赖（不要在测试中真的调用 MinerU API 或 七牛云，除非是集成测试）。

**2. 具体测试用例**
- **测试上传服务**:
    - 模拟一个 PDF 文件上传，验证是否成功写入了数据库（状态是否为“上传中”）。
    - 验证是否成功向 Kafka 发送了消息（Mock Kafka Producer，验证 send 方法被调用）。
- **测试问答服务**:
    - Mock Milvus 的 `search` 方法，让它返回预设的模拟数据（例如：一段关于“汽车保养”的文本和一张“发动机图表”的图片）。
    - 验证构造的 Prompt 是否包含了这些模拟的图文信息。
    - 验证最终返回的答案是否包含“来源：汽车手册.pdf, 第 5 页”。

### 补充信息 (参考你提供的代码片段)
- **MinerU 调用方式**: 参考 `02.md` 中的 Python requests 代码。Worker 服务会消费 Kafka 消息来调用它。
- **七牛云配置**: 访问密钥已提供，但在代码中请使用环境变量或配置类来管理，不要硬编码。

```python

'''MinerU'''
import requests

token = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI0ODgwMDQ1NyIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc3OTI2ODcxMSwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiOWY5NjRjNDMtZTRhNC00MTgyLWFlMDUtMWY1NGVkYjNmNmQ2IiwiZW1haWwiOiIiLCJleHAiOjE3ODcwNDQ3MTF9.GE684bFMtTSP43uQfOdXVweq6w5N24Fjat-AK1f5-0VoiaAE_DR4n_QTZVBc-TMkXTxZWSjB7DJ_xYePBRxy3A"
url = "https://mineru.net/api/v4/extract/task"
header = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {token}"
}
data = {
  "url": "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
  "model_version": "vlm"
}

res = requests.post(url, headers=header, json=data)
print(res.status_code)
print(res.json())
print(res.json()["data"])

'''QiNiuYun'''
from qiniu import Auth, put_file, etag
import qiniu.config

# 1. 初始化Auth状态
access_key = 'fmelP1qnrKlVuAGAdcRXVPb-eBSgcKz8kDDRjbQ6'
secret_key = 'JI0tlniy62avkHU0aN9iiSDZeJs41m2mKaYJHJ-C'
q = Auth(access_key, secret_key)

# 2. 生成上传 Token
bucket_name = 'duixiangcunchu001'
token = q.upload_token(bucket_name, key=None)  # key为None时由七牛云生成文件名

# 3. 上传文件
local_file_path = '汽车知识手册.pdf'  # 本地文件路径
key_in_qiniu = 'remote_file_dk'  # 上传后在七牛云保存的文件名

ret, info = put_file(token, key_in_qiniu, local_file_path)
print(ret)  # 打印上传成功后的文件信息，如hash值

'''Milvus'''
from pymilvus import MilvusClient

client = MilvusClient("http://localhost:19530")

client = MilvusClient(
  uri="http://localhost:19530",
  token="root:Milvus",
  db_name="default"
)

client = MilvusClient(
  uri="http://localhost:19530",
  token="user:a6acc7196c5690f17a6a2a577a79cfb048b4bef270cd5f8748b1430a63778af6d480daab2202791f8e747079d15db54e2f3f7dba",
  # replace this with your token
  db_name="default"
)

```

