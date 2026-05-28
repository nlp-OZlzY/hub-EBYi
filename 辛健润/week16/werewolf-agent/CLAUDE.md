# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AI 狼人杀多智能体协作与博弈系统。基于 LangGraph/LangChain 构建，集成 DeepSeek 模型实现智能博弈。

## 技术栈

- **语言**: Python 3.10+
- **框架**: LangGraph / LangChain
- **LLM**: DeepSeek API
- **依赖管理**: pip

## 关键约定

### 代码组织
- `core/` - 核心引擎模块（对局引擎、角色系统、消息总线）
- `agents/` - Agent 模块（通用 Agent 基类、角色化 Agent）
- `llm/` - LLM 接口抽象层
- `utils/` - 工具函数（日志、配置）

### Agent 设计
- 每个角色 Agent 继承自 `BaseAgent`
- 通过 `MessageBus` 实现信息隔离
- 使用结构化 JSON 日志输出

### 游戏流程
1. 初始化角色配置
2. 夜晚阶段（狼人刀人、预言家验人等）
3. 白天阶段（发言、投票、死亡判定）
4. 胜负裁决
5. 输出日志

## 命令

```bash
# 运行游戏
python main.py

# 安装依赖
pip install -r requirements.txt
```

## 配置

在 `.env` 或 `utils/config.py` 中配置 DeepSeek API Key：

```
DEEPSEEK_API_KEY=your_api_key
```