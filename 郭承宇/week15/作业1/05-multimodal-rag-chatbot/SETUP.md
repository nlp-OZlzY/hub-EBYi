# 环境搭建指南

## 1. Python 版本要求

- Python >= 3.10

```bash
python --version  # 确认 >= 3.10
```

## 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

## 3. Milvus 配置

### 方式一：Zilliz Cloud（推荐，无需本地部署）

1. 注册 [Zilliz Cloud](https://cloud.zilliz.com/) 账号
2. 创建 Serverless 集群，获取 URI 和 Token
3. 填入 `.env` 文件：
   ```
   MILVUS_URI=https://in03-xxxxx.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn
   MILVUS_TOKEN=your_token_here
   MILVUS_COLLECTION=rag_data_new
   ```

### 方式二：本地 Docker

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  milvusdb/milvus:latest
```

然后 `.env` 配置：
```
MILVUS_URI=http://localhost:19530
MILVUS_TOKEN=
MILVUS_COLLECTION=rag_data_new
```

## 4. Kafka 配置

### Docker 方式

```bash
docker run -d --name kafka \
  -p 9092:9092 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  bitnami/kafka:latest
```

`.env` 配置：
```
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=rag-data
```

## 5. mineru-api 安装与启动

### 安装 mineru

```bash
pip install mineru
```

### 启动 mineru API 服务

```bash
mineru-api --host 0.0.0.0 --port 8000 --enable-vlm-preload true
```

确保 mineru API 服务运行在 `http://127.0.0.1:8000`，提供 `/parse` 端点。

`.env` 配置：
```
MINERU_BACKEND=vlm-http-client
MINERU_ENDPOINT=http://127.0.0.1:8000
MINERU_TIMEOUT=600
```

## 6. 模型下载

项目需要两个本地模型，放到 `./models/` 目录下：

### BGE 文本嵌入模型 (bge-small-zh-v1.5)

```bash
# 方式一：自动下载（设置好 HF_ENDPOINT 镜像后首次运行自动下载）
# 方式二：手动下载
git lfs install
git clone https://huggingface.co/BAAI/bge-small-zh-v1.5 models/bge-small-zh-v1.5
```

模型来源: [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)  
大小约 100MB，路径对应 `.env` 中 `BGE_MODEL_PATH=./models/bge-small-zh-v1.5`

### Jina CLIP 多模态模型 (jina-clip-v2)

```bash
# 方式一：自动下载
# 方式二：手动下载（模型较大，约 14GB）
git lfs install
git clone https://huggingface.co/jinaai/jina-clip-v2 models/jina-clip-v2
```

模型来源: [jinaai/jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2)  
大小约 14GB，路径对应 `.env` 中 `CLIP_MODEL_PATH=./models/jina-clip-v2`

### 国内镜像

如果下载缓慢，设置 HuggingFace 镜像：
```
HF_ENDPOINT=https://hf-mirror.com
```

## 7. DashScope (Qwen) API Key

1. 注册 [阿里云 DashScope](https://dashscope.aliyun.com/)
2. 获取 API Key
3. 填入 `.env`：
   ```
   DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
   QWEN_MODEL=qwen-plus
   ```

## 8. .env 配置说明

复制 `.env.example` 为 `.env`，按实际情况填写：

```bash
cp .env.example .env
# 编辑 .env 填入真实的 API Key 和连接地址
```

各配置项含义：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MILVUS_URI` | Milvus/Zilliz 连接地址 | Zilliz Cloud 地址 |
| `MILVUS_TOKEN` | Milvus/Zilliz 认证 Token | — |
| `MILVUS_COLLECTION` | Milvus 集合名称 | rag_data_new |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka 地址 | localhost:9092 |
| `KAFKA_TOPIC` | Kafka 主题名称 | rag-data |
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key | — |
| `QWEN_MODEL` | Qwen 模型名称 | qwen-plus |
| `BGE_MODEL_PATH` | BGE 模型本地路径 | ./models/bge-small-zh-v1.5 |
| `CLIP_MODEL_PATH` | CLIP 模型本地路径 | ./models/jina-clip-v2 |
| `MINERU_BACKEND` | mineru 后端模式 | vlm-http-client |
| `MINERU_ENDPOINT` | mineru API 服务地址 | http://127.0.0.1:8000 |
| `MINERU_TIMEOUT` | 文档解析超时（秒） | 600 |
| `HF_ENDPOINT` | HuggingFace 镜像 | https://hf-mirror.com |
| `CHUNK_SIZE` | 文本切分大小（字符） | 256 |
| `DEFAULT_TOP_K` | 默认检索返回条数 | 5 |

## 9. 启动命令

### 启动 API 服务

```bash
python src/api_server.py
# API 运行在 http://localhost:8000
# Swagger 文档: http://localhost:8000/docs
```

### 启动 Streamlit UI

```bash
streamlit run src/web_demo.py
# UI 运行在 http://localhost:8501
```

### 启动文档解析 Worker

```bash
python src/offline_process_worker.py
# Worker 持续监听 Kafka，消费消息并解析文档
```

## 10. 启动顺序

推荐按以下顺序启动：

1. **基础设施**：Kafka + Milvus + mineru API 服务
2. **Worker**：`python src/offline_process_worker.py`
3. **API 服务**：`python src/api_server.py`
4. **UI**：`streamlit run src/web_demo.py`

## 11. 常见问题

**Q: Kafka 连接失败？**
确保 Kafka 已启动：`docker ps | grep kafka`

**Q: 模型下载慢？**
设置 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像。

**Q: mineru 解析超时？**
大文件可能需要更长时间，可调整 `MINERU_TIMEOUT` 环境变量。

**Q: Milvus 连接失败？**
检查 `.env` 中的 `MILVUS_URI` 和 `MILVUS_TOKEN` 是否正确。

**Q: 导入错误 `ModuleNotFoundError: No module named 'config'`？**
确保从项目根目录运行命令（`python src/xxx.py`），Python 会自动把脚本所在目录加入搜索路径。
