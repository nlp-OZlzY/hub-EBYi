# Week17 作业设计思路：基于 redis-vl-python 的教学版实现

## 1. 作业目标

作业要求是参考 `redis-vl-python`，手写 4 个与 LLM 应用常见能力相关的模块：

- `EmbeddingsCache.py`：Embedding 向量缓存
- `SemanticCache.py`：语义问答缓存
- `SemanticMessageHistory.py`：对话历史管理
- `SemanticRouter.py`：语义路由

答案文件放在 `答案/` 目录中，根目录同名文件保留为原始题目/半成品。

## 2. redis-vl-python 做了什么

`redis-vl-python` 是 Redis 官方面向 AI 应用的 Python 工具库。它把 Redis Stack 的能力封装起来，让开发者更方便地完成：

- 向量存储
- 向量相似度检索
- 语义缓存
- 对话历史管理
- 语义路由

普通 Redis 只能做精确 key-value 查询：

```text
GET user:1
```

而向量检索可以做语义相似查询：

```text
"我要退货" -> embedding -> 找到和 "如何退货" 最接近的向量 -> 返回 refund
```

## 3. 本作业的技术取舍

真实的 redis-vl-python 常用 Redis Stack/RediSearch 做向量索引。本作业为了便于理解，把能力拆成两层：

```text
文本
  -> embedding_method 生成向量
  -> FAISS 做向量相似度检索
  -> Redis 保存 key-value、列表、JSON 等业务数据
```

也就是说：

- FAISS 负责“找相似”。
- Redis 负责“存数据、查数据、设置过期时间”。

这样可以避开 Redis Stack 索引语法，把注意力放在语义缓存和路由的核心流程上。

## 4. EmbeddingsCache 设计

### 场景

Embedding 模型调用有成本。同一个文本如果已经生成过向量，就没有必要重复调用模型。

### 数据结构

```text
Redis Key:   {name}:{md5(text)}
Redis Value: embedding 的 float32 bytes
TTL:         默认 24 小时
```

### 核心流程

```text
store(text, embedding)
  -> text 转成 list
  -> embedding 统一成二维 float32
  -> 每条文本生成 MD5 key
  -> pipeline 批量 setex

call(text)
  -> 生成同样的 key
  -> mget 批量读取
  -> bytes 转回 np.ndarray

delete(text)
  -> 生成 key
  -> delete 批量删除
```

### 关键修正点

- 单条文本的 embedding 常见形状是 `(768,)`，必须先 reshape 成 `(1, 768)`。
- 存储前统一转 `float32`，否则读取时用 `float32` 还原会错位。
- 即使传入单条文本，`call()` 也返回列表，便于和批量调用保持一致。

## 5. SemanticCache 设计

### 场景

用户问过一个问题并得到 LLM 回答后，后续语义相近的问题可以直接复用缓存回答。

例子：

```text
"hello world"  -> 缓存回答 "hello answer"
"hello there"  -> 语义相近 -> 直接返回 "hello answer"
```

### 数据结构

```text
FAISS IndexFlatL2:
  保存所有 prompt 的 embedding

Redis List:
  {name}:prompts
  按 FAISS add 的顺序保存原始 prompt

Redis Key:
  {name}:response:{md5(prompt)}
  保存 prompt 对应的 response

本地文件:
  {name}.index
  FAISS 索引持久化文件
```

### 核心流程

```text
store(prompt, response)
  -> prompt 生成 embedding
  -> embedding add 到 FAISS
  -> prompt 用 rpush 写入 Redis list
  -> response 用 setex 写入 Redis
  -> FAISS 索引写入本地文件

call(prompt)
  -> prompt 生成 embedding
  -> FAISS search 找 top_k 近邻
  -> 根据 distance_threshold 过滤
  -> 用 FAISS 返回的 index 去 Redis list 找原始 prompt
  -> 再用 prompt 的 response key 找回答
```

### 关键修正点

- Redis prompt 列表使用 `rpush`，保证列表顺序和 FAISS 索引顺序一致。
- FAISS 返回的是实际索引 `indices`，不能用 `enumerate(distances)` 的位置代替。
- Redis 回答可能因 TTL 过期而消失，所以 `call()` 会过滤 `None`。
- `clear_cache()` 同时删除 Redis 数据和本地 `.index` 文件。

## 6. SemanticMessageHistory 设计

### 场景

LLM 对话需要保存上下文，但不能每次都把全部历史塞进模型。常见做法是：

- 取最近 N 条
- 按角色取消息
- 按相关性取消息

### 数据结构

```text
Redis Key:   semantic_history:{session_id}
Redis Value: JSON 数组
```

JSON 示例：

```json
[
  {"role": "user", "content": "你好"},
  {"role": "llm", "content": "你好，有什么可以帮你？"}
]
```

### 核心流程

```text
add_message(message)
  -> 读取旧历史
  -> 单条 dict 用 append 语义
  -> 多条 list 用 extend 语义
  -> JSON 写回 Redis，并刷新 TTL

get_recent(role, top_k)
  -> 可按 role 过滤
  -> 返回最近 top_k 条

get_relevant(content, top_k)
  -> 对每条消息计算 Levenshtein.ratio
  -> 按相似度从高到低排序
  -> 返回 top_k 条
```

### 关键修正点

- 原始写法如果对 dict 直接 `extend()`，会把 `"role"`、`"content"` 这些 key 写入历史。
- `role` 支持字符串和字符串列表。
- `top_k <= 0` 时返回空列表，避免 `history[-0:]` 等于全量历史。
- 如果没安装 `python-Levenshtein`，代码会用标准库 `difflib` 兜底。

## 7. SemanticRouter 设计

### 场景

把用户输入自动分发到业务意图：

```text
"你好呀"       -> greeting
"我要退货"     -> refund
"帮我查物流"   -> logistics
```

### 数据结构

```text
FAISS IndexFlatL2:
  保存所有 route 示例问题的 embedding

Redis Key:
  {name}:route_names  -> route 名称列表
  {name}:examples     -> 示例问题列表
  {name}:emb:{i}      -> 第 i 个示例问题的 embedding bytes
```

`route_names` 的顺序必须和 FAISS 索引一致：

```text
FAISS index 0 -> "Hi, good morning" -> greeting
FAISS index 1 -> "Hi, good afternoon" -> greeting
FAISS index 2 -> "如何退货" -> refund
```

### 核心流程

```text
add_route(questions, target)
  -> questions 生成 embedding
  -> embedding add 到 FAISS
  -> route_names 追加 target
  -> examples 追加原始问题
  -> embedding bytes 写入 Redis

route(question)
  -> question 生成 embedding
  -> FAISS search(k=1)
  -> 如果距离超过 distance_threshold，返回 None
  -> 否则根据 index 返回 route_names[index]
```

### 扩展能力

当前答案还实现了几个便于验证的方法：

- `route_with_score(question)`：返回 `(route_name, distance)`。
- `get_routes()`：返回去重后的 route 列表。
- `remove_route(target)`：删除 route，并重建 FAISS 索引。
- `clear_routes()`：清空路由，方便重复运行测试。
- `__call__(question)`：支持 `router(question)` 的写法。

## 8. 4 个模块的关系

```text
EmbeddingsCache
  解决：同一文本不要重复生成 embedding

SemanticCache
  解决：语义相近的问题不要重复请求 LLM

SemanticMessageHistory
  解决：对话历史可保存、可裁剪、可检索

SemanticRouter
  解决：用户输入自动分发到不同业务意图
```

它们组合起来就是一个简化版 LLM 应用基础设施：

```text
用户输入
  -> SemanticRouter 判断业务意图
  -> SemanticCache 尝试命中历史回答
  -> 未命中时调用 LLM
  -> EmbeddingsCache 缓存 embedding
  -> SemanticMessageHistory 保存对话上下文
```

## 9. 一句话总结

本作业不是直接复刻 redis-vl-python 的源码，而是手写一个教学版：

```text
Redis 保存业务数据 + FAISS 做向量检索 + Python 代码串起缓存、历史、路由流程
```

这样能更清楚地理解 redis-vl-python 背后的核心设计。
