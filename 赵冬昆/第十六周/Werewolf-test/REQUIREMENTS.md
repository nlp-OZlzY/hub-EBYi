# 🐺 AI 狼人杀：多智能体协作与博弈 Agent Team 终极实战指南

> **文档用途**：本文档是 AI 辅助编程（Vibe Coding）的唯一真理来源。所有代码生成、架构设计、Prompt 工程均需严格遵循此规范。
> **版本**：v2.0 (Enhanced)

## 1. 项目核心定义

### 1.1 愿景
构建一个**高可信、可观测、可进化**的多智能体狼人杀博弈平台。不仅是游戏模拟，更是研究 LLM 在信息不对称、多方博弈、社会推理场景下的行为实验室。

### 1.2 核心差异化特征
-   🧠 **结构化思维链**：拒绝黑盒决策，所有行动必有 CoT 支撑。
-   🎭 **动态人格系统**：Agent 具备独立性格参数，拒绝同质化发言。
-   🔌 **插件化角色架构**：角色逻辑与引擎解耦，支持热插拔与自生成。
-   👁️ **信念可视化**：前端实时渲染 Agent 内心的信任/怀疑矩阵。
-   🛡️ **元认知监控**：内置防死循环与信息熵监测，保障对局流畅。

---

## 2. 系统架构蓝图

### 2.1 分层架构
| 层级 | 组件 | 关键职责 | 技术栈推荐 |
| :--- | :--- | :--- | :--- |
| **表现层** | Vue3 Watchtower | 实时看板、信念热力图、人机交互、日志流 | Vue3 + Pinia + ECharts |
| **接口层** | FastAPI Gateway | RESTful API、WebSocket 推送、鉴权、限流 | FastAPI + Uvicorn |
| **引擎层** | Game Orchestrator | 状态机流转、规则校验、元监控、插件调度 | Python + StateMachine |
| **智能层** | Agent Runtime | LLM 调用、CoT 解析、记忆管理、人格注入 | LangChain / AutoGen |
| **数据层** | Structured Store | 对局日志、评测指标、配置模板 | SQLite / PostgreSQL + JSONL |

### 2.2 目录结构规范
```text
ai-werewolf/
├── agent/                # 智能体核心
│   ├── core/             # BaseAgent, Memory, Persona
│   ├── prompts/          # 分角色 Prompt 模板 (Jinja2)
│   └── strategies/       # 高阶策略模块 (悍跳, 倒钩)
├── engine/               # 游戏引擎
│   ├── state_machine.py  # 核心状态流转
│   ├── meta_monitor.py   # ⭐ 元认知监控器
│   ├── plugins/          # ⭐ 角色插件目录
│   └── rules/            # 胜负判定、动作合法性校验
├── api/                  # 后端服务
├── frontend/             # 前端观战台
├── evaluation/           # 评测与Leaderboard
├── configs/              # 游戏板子、人格预设
└── docs/                 # 设计文档与API Spec