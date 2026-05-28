# 作业2：MinerU 与 pdfplumber 文档解析效果对比

## 一、MinerU 论文阅读笔记

### 1.1 论文基本信息

- **标题**: MinerU: An Open-Source Solution for Precise Document Content Extraction
- **作者**: Bin Wang, Chao Xu, Xiaomeng Zhao 等（上海人工智能实验室 OpenDataLab 团队）
- **发表时间**: 2024 年 9 月（arXiv: 2409.18839）
- **最新版本**: MinerU 2.5（2025 年 9 月，arXiv: 2509.22186）

### 1.2 核心问题

现有开源文档解析方案（如 PyPDF2、pdfplumber、pdfminer 等）在处理多样化文档时，难以一致地输出高质量解析结果，主要问题包括：

1. **布局分析缺失**：无法识别多栏、图文混排等复杂布局，导致阅读顺序错乱
2. **公式无法识别**：数学公式被当作图片丢弃，或识别为乱码
3. **表格解析能力弱**：跨页表格、无线表格、合并单元格等场景处理差
4. **扫描版 PDF 支持差**：缺乏 OCR 能力，扫描文档返回空文本

### 1.3 技术架构

MinerU 采用**多模块流水线**（Pipeline）架构，基于 PDF-Extract-Kit 模型库，分三个阶段处理：

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  阶段1: 预处理   │───▶│  阶段2: 内容解析     │───▶│  阶段3: 后处理       │
│                 │    │                     │    │                     │
│ - 语言检测       │    │ - 布局检测           │    │ - 无效区域删除       │
│ - 页数/尺寸提取  │    │   (DocLayout-YOLO)   │    │ - 区域内容拼接       │
│ - 扫描/文本判断  │    │ - 公式检测           │    │ - 排序信息生成       │
│ - 乱码检测       │    │   (YOLOv8-ft)        │    │ - Markdown 输出      │
│                 │    │ - OCR (PaddleOCR)    │    │                     │
│                 │    │ - 公式识别           │    │                     │
│                 │    │   (UniMERNet)        │    │                     │
│                 │    │ - 表格识别           │    │                     │
│                 │    │   (StructEqTable)    │    │                     │
└─────────────────┘    └─────────────────────┘    └─────────────────────┘
```

#### 阶段 1：文档预处理

- 提取元信息：语言类型（中/英）、页数、页面尺寸
- 判断文档类型：文本型 PDF 直接提取文字；扫描版 PDF 触发 OCR
- 检测乱码文本：基于视觉线索（大面积图片区域、极少可提取文本）识别扫描内容

#### 阶段 2：内容解析（核心）

基于 PDF-Extract-Kit，集成 5 个专用模型：

| 任务 | 模型 | 说明 |
|------|------|------|
| 布局检测 | DocLayout-YOLO / LayoutLMv3-ft | 识别标题、段落、图片、表格、公式等区域 |
| 公式检测 | YOLOv8-ft | 定位行内公式和独立公式块 |
| OCR | PaddleOCR | 识别文本区域中的文字（支持 109 种语言）|
| 公式识别 | UniMERNet | 将公式图像转为 LaTeX 代码 |
| 表格识别 | StructEqTable | 将表格图像转为 HTML/Markdown |

关键设计：**先做布局检测，再在细分区域内做 OCR**，避免多栏文本被错误合并为单栏。

#### 阶段 3：后处理

- 删除无效区域（页眉、页脚、页码）
- 按区域定位信息拼接内容
- 输出 Markdown / JSON 格式

### 1.4 MinerU 2.5 的演进

MinerU 2.5 引入了**端到端视觉语言模型（VLM）**替代传统流水线：

- **旧版**（v0.x）：多模块 Pipeline → 各模块独立优化，但存在错误传播问题
- **新版**（v2.5）：单一 VLM 模型 → 统一处理布局、OCR、公式、表格，更鲁棒
- 支持 VLM HTTP Client 模式，可调用云端 VLM 推理服务
- 输出质量接近商业方案（如 Mathpix）

### 1.5 性能指标

| 任务 | 指标 | MinerU | 对比方案 |
|------|------|--------|----------|
| 布局检测 | mAP（学术论文）| 77.6% | DocXchain: 52.8% |
| 布局检测 | AP50（学术论文）| 93.3% | - |
| 公式检测 | AP50（学术论文）| 87.7% | Pix2Text-MFD: 60.1% |
| 公式识别 | CDM | 0.968 | Mathpix: 0.951 |

### 1.6 MinerU 使用方法

#### 命令行（离线）

```bash
# 基础用法
mineru -p <input_path> -o <output_path>

# 使用 VLM 模式
mineru -p <file_path> -o ./processed -b vlm-http-client -u http://127.0.0.1:30000
```

#### FastAPI 部署（在线服务）

```bash
mineru-api --host 0.0.0.0 --port 8000
```

#### Python SDK

```python
from mineru import MinerU

mineru = MinerU()
result = mineru.parse("input.pdf")
# result 包含 markdown 文本 + 图片路径
```

---

## 二、pdfplumber 简介

### 2.1 基本原理

pdfplumber 基于 pdfminer.six 构建，主要能力包括：

- **文本提取**：逐字符解析 PDF 底层文本层，返回字符级位置信息
- **表格提取**：基于线条和文本对齐检测表格结构
- **可视化调试**：支持渲染页面对象用于调试

### 2.2 核心方法

```python
import pdfplumber

with pdfplumber.open("example.pdf") as pdf:
    for page in pdf.pages:
        # 提取文本
        text = page.extract_text()

        # 提取表格
        tables = page.extract_tables()

        # 提取字符级信息
        chars = page.chars  # 位置、字体、文本等
```

### 2.3 局限性

| 局限 | 说明 |
|------|------|
| 无布局分析 | 无法识别多栏布局，文本按内部坐标顺序输出 |
| 无 OCR 能力 | 扫描版 PDF 返回空文本 |
| 无公式识别 | 公式被当作图片丢弃或识别为乱码 |
| 无图片提取 | 不支持提取嵌入图片 |
| 速度较慢 | 逐字符处理，大文档处理慢 |

---

## 三、效果对比

### 3.1 对比维度总览

| 维度 | MinerU | pdfplumber |
|------|--------|------------|
| **文本提取** | ✅ 先布局检测再 OCR，阅读顺序正确 | ⚠️ 直接读取文本层，多栏布局顺序可能错乱 |
| **扫描版 PDF** | ✅ 自动检测并启用 OCR | ❌ 无 OCR，返回空文本 |
| **布局分析** | ✅ 多模型联合检测（标题/段落/图/表/公式）| ❌ 无布局分析能力 |
| **公式处理** | ✅ 检测 + 识别为 LaTeX | ❌ 公式作为图片丢弃 |
| **表格提取** | ✅ 支持旋转/跨页/合并单元格，输出 HTML/Markdown | ⚠️ 仅支持有线条边框的简单表格 |
| **图片提取** | ✅ 提取图片并保留引用关系 | ❌ 不支持图片提取 |
| **输出格式** | Markdown / JSON / LaTeX | 纯文本 / Python 列表 |
| **多语言** | ✅ OCR 支持 109 种语言 | ⚠️ 取决于 PDF 内嵌字体编码 |
| **运行速度** | 较慢（需 GPU，约 1 min/文件）| 较快（纯 CPU，秒级）|
| **部署复杂度** | 高（需安装模型、可选 GPU）| 低（pip install 即可）|

### 3.2 具体场景对比

#### 场景 1：纯文本 PDF（单栏）

```python
# pdfplumber
import pdfplumber
with pdfplumber.open("simple_text.pdf") as pdf:
    text = pdf.pages[0].extract_text()
# 结果：文本提取完整，顺序正确

# MinerU
# mineru -p simple_text.pdf -o ./output
# 结果：文本提取完整，额外保留段落结构标记
```

**结论**：对于简单文本 PDF，两者效果相当。pdfplumber 速度更快。

#### 场景 2：多栏排版 PDF（学术论文）

```
# pdfplumber 输出：
左栏第1行 右栏第1行
左栏第2行 右栏第2行   ← 顺序错乱！两栏内容交错

# MinerU 输出：
左栏第1行
左栏第2行
右栏第1行             ← 顺序正确！按阅读顺序排列
右栏第2行
```

**结论**：MinerU 先做布局检测再 OCR，能正确处理多栏；pdfplumber 按内部坐标顺序输出，多栏错乱。

#### 场景 3：包含数学公式的 PDF

```
# pdfplumber 输出：
"对于任意实数 x，有  成立"  ← 公式丢失！

# MinerU 输出：
"对于任意实数 $x$，有 $f(x) = \sum_{i=1}^{n} a_i x^i$ 成立"  ← 公式转为 LaTeX
```

**结论**：MinerU 能识别公式并转为 LaTeX；pdfplumber 完全无法处理公式。

#### 场景 4：扫描版 PDF

```
# pdfplumber 输出：
""  ← 空文本，无内容

# MinerU 输出：
自动检测扫描版 → 启用 OCR → 输出完整文本
```

**结论**：MinerU 自动检测并启用 OCR；pdfplumber 对扫描版完全失效。

#### 场景 5：复杂表格

```
# pdfplumber 输出（有边框的简单表格）：
[['A', 'B', 'C'], ['1', '2', '3']]  ← 正常

# pdfplumber 输出（无线/合并单元格表格）：
[[None, None], [None, None]]  ← 识别失败

# MinerU 输出（所有类型表格）：
| A | B | C |
|---|---|---|
| 1 | 2 | 3 |  ← 结构化 Markdown，支持复杂表格
```

**结论**：MinerU 的表格识别更鲁棒，支持无线表格和合并单元格；pdfplumber 仅适合有清晰边框的简单表格。

### 3.3 对比代码示例

```python
# ============ pdfplumber 提取 ============
import pdfplumber

def extract_with_pdfplumber(pdf_path):
    results = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            tables = page.extract_tables() or []
            results.append({
                "page": page.page_number,
                "text": text,
                "tables": tables,
            })
    return results

# ============ MinerU 提取 ============
# CLI 方式:
# mineru -p input.pdf -o ./output
# 结果在 ./output/<filename>/auto/<filename>.md

# API 方式（需先启动 mineru-api）:
import requests

def extract_with_mineru_api(pdf_path, api_url="http://localhost:8000"):
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{api_url}/extract",
            files={"file": f}
        )
    return response.json()
```

### 3.4 在 RAG 场景下的差异

对于多模态 RAG 对话系统（如作业 1 的项目），文档解析质量直接影响检索和问答效果：

| 环节 | pdfplumber | MinerU |
|------|-----------|--------|
| **文本切分** | 无段落标记，切分粗糙 | Markdown 标题/段落结构清晰，切分精准 |
| **向量编码** | 公式丢失 → 检索不到公式相关内容 | 公式保留 → 支持公式相关查询 |
| **图片检索** | 无图片 → 无法做图像编码 | 图片提取 → 支持 CLIP 图像编码 |
| **来源溯源** | 无页码/位置标记 | 保留区域定位信息，可溯源到具体页面 |
| **表格问答** | 简单表格可提取，复杂表格丢失 | 表格转 Markdown/HTML，信息完整 |

**结论**：在多模态 RAG 场景下，MinerU 是更好的选择。它输出的 Markdown 包含结构化文本、公式、图片引用，可以直接用于文本分块、多模态编码和检索。pdfplumber 仅适合简单的纯文本提取场景。

---

## 四、总结

### 4.1 选择建议

| 场景 | 推荐工具 | 理由 |
|------|----------|------|
| 简单文本 PDF 提取 | pdfplumber | 快速、轻量、无需 GPU |
| 多栏/复杂布局 PDF | MinerU | 布局检测保证阅读顺序 |
| 包含公式的学术论文 | MinerU | 公式识别为 LaTeX |
| 扫描版 PDF | MinerU | 自动 OCR |
| 多模态 RAG 系统 | MinerU | 输出 Markdown+图片，支持跨模态编码 |
| 快速原型/数据探索 | pdfplumber | 部署简单，秒级出结果 |
| 生产级文档处理流水线 | MinerU | 质量高，但需要 GPU 资源 |

### 4.2 核心差异一句话总结

**pdfplumber** 是一个**基于规则的文本提取库**，擅长从结构化 PDF 中提取文本和简单表格；**MinerU** 是一个**基于深度学习的文档解析引擎**，能理解文档布局、识别公式和表格、处理扫描件，输出结构化 Markdown，是构建 RAG 系统的首选。

### 4.3 互补使用

在实际项目中，可以组合使用两者：

```python
def smart_parse(pdf_path):
    """智能选择解析策略"""
    # 1. 先用 pdfplumber 快速判断文档类型
    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text() or ""

    # 2. 如果文本丰富 → 简单文档，pdfplumber 足够
    if len(text.strip()) > 100:
        return extract_with_pdfplumber(pdf_path)

    # 3. 如果文本稀少 → 扫描版或复杂布局，使用 MinerU
    return extract_with_mineru(pdf_path)
```

这样可以在保证解析质量的同时，减少对 GPU 资源的依赖。
