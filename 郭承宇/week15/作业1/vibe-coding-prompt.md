# 多模态 RAG 聊天机器人 — Vibe Coding 提示词

> 将以下内容完整复制到 Claude Code 对话中，即可启动 vibe coding 流程。

---

## 角色与目标

你是一个资深全栈 Python 工程师，擅长 FastAPI + Streamlit + 向量数据库 + 消息队列架构。
你的任务是根据下方需求规格，**从零 vibe coding 一个多模态 RAG 聊天机器人系统**，实现所有接口、编写测试逻辑，确保代码可运行。

---

## 项目概述

构建一个多模态 RAG 系统，用户可上传 PDF 文档（含图文混排），系统自动解析文档内容（文本+图片），将内容向量化存入 Milvus，用户提问时进行多模态检索并调用 Qwen 大模型生成答案。

### 核心流程

```
用户上传PDF → Kafka消息队列 → Worker离线解析(mineru) → chunk切分 → 向量编码(bge+clip) → 存入Milvus
用户提问 → 提问embedding → Milvus检索(文本+图像) → 检索结果+问题 → Qwen生成答案
```

### 技术栈

| 类别 | 技术选型 |
|------|---------|
| Web框架 | FastAPI (API) + Streamlit (UI) |
| 文档解析 | mineru (PDF→Markdown+图片) |
| 文本编码 | BAAI/bge-small-zh-v1.5 (文本向量, 512维) |
| 多模态编码 | jinaai/jina-clip-v2 (文本+图像向量, 1024维) |
| 向量数据库 | Milvus (Zilliz Cloud Serverless) |
| 消息队列 | Kafka |
| 元信息存储 | SQLite (SQLAlchemy ORM) |
| LLM | Qwen (DashScope API, qwen-plus) |

---

## 需求规格

### 1. 数据模型 (orm_model.py)

**File 表**：文件元信息
- `id`: Integer, 主键, 自增
- `filename`: String(255), 原始文件名, 非空
- `filepath`: String(1000), 文件存储路径, 非空
- `filestate`: String(20), 处理状态, 非空，取值: "已上传" | "解析中" | "已完成" | "解析失败"

**数据库**：SQLite，文件 `db.db`，使用 SQLAlchemy ORM。

### 2. 文件上传接口 (POST /upload/document)

**功能**：向指定知识库上传文档

**请求**：
- multipart/form-data
- `file`: PDF/DOCX/TXT 文件
- `knowledge_base_id`: 可选，知识库标识（当前版本默认为 "default"）

**处理步骤**：
1. 接收上传文件，保存到 `uploads/` 目录（UUID 重命名，保留原始扩展名）
2. 在 SQLite 中创建 File 记录，状态设为 "已上传"
3. 向 Kafka topic `rag-data` 发送消息：`{"file_name": 原始文件名, "file_path": 保存路径, "id": 数据库记录ID}`
4. 返回 `{"status": "success", "file_id": id, "message": "文件上传成功，等待解析"}`

**错误处理**：
- 文件为空 → 400
- 不支持的文件类型 → 400
- Kafka 不可达 → 文件仍保存，状态标记为 "待发送"，返回 200 + 警告信息

### 3. 文档解析 Worker (offline_process_worker.py)

**功能**：离线消费 Kafka 消息，解析文档并编码存入 Milvus

**处理步骤**：
1. 从 Kafka topic `rag-data` 消费消息
2. 更新 File 状态为 "解析中"
3. 调用 mineru 解析 PDF：`mineru -p {file_path} -o ./processed -b vlm-http-client -u http://127.0.0.1:30000`
4. 读取解析后的 Markdown 文件
5. 文本切分：按 chunk_size=256 字符切分，保留图片引用
6. 对每个 chunk 编码：
   - `text_vector`: bge-small-zh-v1.5 编码文本(去除图片行), 512维
   - `clip_text_vector`: jina-clip-v2 编码文本(去除图片行), 1024维
   - `clip_image_vector`: jina-clip-v2 编码图片(如有), 1024维; 无图则存零向量
7. 将编码结果插入 Milvus collection `rag_data_new`
8. 更新 File 状态为 "已完成"；失败则标记 "解析失败"

**Milvus Collection 结构** (rag_data_new)：
- `id`: 自增主键
- `text_vector`: FloatVector(512) — bge 文本向量
- `clip_text_vector`: FloatVector(1024) — clip 文本向量
- `clip_image_vector`: FloatVector(1024) — clip 图像向量
- `text`: VarChar — chunk 原文
- `db_id`: Int64 — 对应 File 表的 id
- `file_name`: VarChar — 文件名
- `file_path`: VarChar — 文件路径

**文本切分逻辑**：
- 按行读取 Markdown
- 跳过空行、"# References" 行、引用编号行 (如 "[1...")
- 当前 chunk 累积字符数 <= chunk_size 时拼接到当前 chunk，否则新开 chunk

### 4. 多模态问答接口 (POST /chat)

**功能**：接收用户提问，执行 RAG 检索并生成答案

**请求**：
```json
{
  "question": "用户问题",
  "knowledge_base_id": "default",
  "top_k": 5
}
```

**处理步骤**：
1. 使用 bge-small-zh-v1.5 对用户问题编码 (512维)
2. 在 Milvus 中搜索 `text_vector` 字段，返回 top_k 条结果
3. 输出字段：`text`, `db_id`, `file_name`, `file_path`
4. 将检索结果中的图片相对路径转换为实际路径：
   - 原始: `images/xxx.jpg` → 替换为: `./processed/{file_dir}/vlm/images/xxx.jpg`
   - 其中 file_dir = os.path.basename(file_path).split(".")[0]
5. 拼接检索结果为上下文
6. 调用 Qwen (DashScope qwen-plus) 生成答案

**RAG Prompt 模板**：
```
基于资料回答的提问提问问题：{question}

相关资料: {related_content}

回答要求：
- 回答要客观，有逻辑，要基于只有的资料。
- 如果资料中包含图片链接，则单独一行进行输出，保留图的原始链接，需要将图放在合适的相关内容的位置。
```

**响应**：
```json
{
  "answer": "生成的答案(Markdown格式)",
  "sources": [
    {"file_name": "xxx.pdf", "db_id": 1, "text": "相关片段..."}
  ]
}
```

### 5. 文件管理接口 (GET /files, DELETE /files/{file_id})

**GET /files**：返回所有文件列表
```json
[
  {"id": 1, "filename": "test.pdf", "filepath": "uploads/xxx.pdf", "filestate": "已完成"}
]
```

**DELETE /files/{file_id}**：
1. 从 SQLite 删除 File 记录
2. 删除本地文件
3. 从 Milvus 删除 db_id == file_id 的所有向量记录
4. 返回 `{"status": "success", "message": "文件删除成功"}`

### 6. Streamlit UI

**文件管理页** (web_page_upload.py)：
- 展示所有已上传文件列表，每个文件旁有删除按钮
- 文件上传组件，支持 pdf/docx/txt 格式
- 上传后自动保存到数据库并发送 Kafka 消息

**图文对话页** (web_page_chat.py)：
- 聊天界面，支持文字输入
- 用户提问后：编码→检索→调用 Qwen 生成答案
- 支持渲染 Markdown 答案中的图片（正则匹配 `![alt](url)` 并展示为图片）
- 侧边栏清空聊天按钮
- 系统提示词："你好，我是AI助手，可以直接与大模型对话也可以调用内部工具。"

**入口** (web_demo.py)：Streamlit 多页面导航，包含"文件管理"和"图文对话"两个页面。

---

## 测试逻辑

### 单元测试

使用 pytest 编写，放在 `tests/` 目录下。

**test_orm_model.py** — 数据模型测试：
- 创建 File 记录，验证字段正确写入
- 查询 File 记录，验证返回结果
- 更新 filestate 字段
- 删除 File 记录，验证已删除
- 使用临时数据库 (fixture)，不污染正式数据

**test_text_split.py** — 文本切分测试：
- 普通文本切分，验证 chunk 大小不超过 256
- 包含图片行 `![img](path)` 的文本，图片行保留在 chunk 中
- 空行被跳过
- "# References" 行被跳过
- 引用编号行 "[1..." 被跳过
- 超长单行（>256）作为一个独立 chunk

**test_encode.py** — 编码逻辑测试：
- 纯文本编码：验证 bge 输出维度为 512，clip 文本输出维度为 1024
- 含图片 chunk 编码：验证 clip 图像向量维度为 1024
- 无图片 chunk：clip_image_vector 应为 1024 维零向量
- 编码异常时返回零向量（不抛出异常）

### 接口测试

**test_upload_api.py** — 上传接口测试：
- 正常上传 PDF 文件，验证返回 file_id 和 success
- 上传空文件，验证返回 400
- 上传不支持的文件类型 (.exe)，验证返回 400
- 上传后验证数据库有对应记录，filestate 为 "已上传"
- 验证 uploads 目录下生成了对应文件

**test_chat_api.py** — 问答接口测试：
- 发送正常问题，验证返回 answer 字段非空
- 验证 sources 字段包含检索来源
- 发送空问题，验证返回 400
- top_k 参数验证：默认5，可自定义

**test_file_management_api.py** — 文件管理接口测试：
- GET /files 返回文件列表
- DELETE /files/{id} 删除成功
- DELETE /files/{id} 删除不存在的 id，验证返回 404

### 集成测试

**test_integration.py** — 端到端流程测试（需启动所有服务）：
- 上传文件 → 等待 Worker 解析 → 提问 → 验证答案包含相关内容
- 上传 → 删除文件 → 验证 Milvus 中向量已清除

---

## 项目结构

```
05-multimodal-rag-chatbot/
├── web_demo.py                 # Streamlit 入口
├── web_page_upload.py          # 文件管理页面
├── web_page_chat.py            # 图文对话页面
├── offline_process_worker.py   # 文档解析 Worker
├── orm_model.py                # 数据模型
├── api_server.py               # FastAPI 接口服务
├── config.py                   # 配置项（模型路径、API Key、连接地址等）
├── requirements.txt            # 依赖
├── doc1/                       # 示例文档
│   ├── content.md
│   └── demo.png
├── tests/
│   ├── test_orm_model.py
│   ├── test_text_split.py
│   ├── test_encode.py
│   ├── test_upload_api.py
│   ├── test_chat_api.py
│   ├── test_file_management_api.py
│   └── test_integration.py
├── uploads/                    # 上传文件存储
└── processed/                  # 解析后文件存储
```

---

## 编码要求

1. **config.py 统一管理所有配置**：模型路径、API Key、Milvus 地址、Kafka 地址等，不硬编码在业务代码中
2. **API Key 不提交到 Git**：使用 `.env` 文件 + python-dotenv 加载，提供 `.env.example` 模板
3. **错误处理**：所有外部调用（Milvus、Kafka、LLM）要有 try-except，失败时给出明确错误信息
4. **日志**：使用 logging 模块，关键步骤记录日志
5. **类型注解**：所有函数参数和返回值加类型注解
6. **测试可运行**：单元测试不依赖外部服务（用 mock 替代 Milvus/Kafka/LLM），接口测试用 FastAPI TestClient

---

## 环境搭建文档

在项目根目录生成 `SETUP.md`，包含：
1. Python 版本要求 (3.10+)
2. pip install -r requirements.txt
3. Milvus 安装与配置（Docker 方式 + Zilliz Cloud 方式）
4. Kafka 安装与配置（Docker 方式）
5. mineru 安装与 VLM 服务配置
6. 模型下载（bge-small-zh-v1.5、jina-clip-v2）
7. .env 配置说明
8. 启动命令：
   - `python api_server.py` — 启动 API 服务
   - `streamlit run web_demo.py` — 启动 UI
   - `python offline_process_worker.py` — 启动解析 Worker
   - `pytest tests/` — 运行测试

---

## 执行顺序

请按以下顺序 vibe coding：

1. 先创建 `config.py` + `orm_model.py` + `requirements.txt` + `.env.example`
2. 再创建 `api_server.py`（FastAPI，实现所有 HTTP 接口）
3. 再创建 `offline_process_worker.py`（Kafka 消费 + 文档解析 + 编码）
4. 再创建 Streamlit 页面 (`web_demo.py`, `web_page_upload.py`, `web_page_chat.py`)
5. 编写所有测试文件 (`tests/`)
6. 最后生成 `SETUP.md`

每完成一个步骤，简要说明做了什么，再继续下一步。
