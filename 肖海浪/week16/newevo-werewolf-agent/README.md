# 自演化狼人杀 Agent（newevo-werewolf-agent）

基于多 Agent 的狼人杀对局引擎，实现「读懂自己 → 修改自己 → 运行自己」的自演化循环：每局结束后根据指标与日志由 LLM 反思并改写各角色策略 prompt，在多局迭代中逐步专精。

## 核心特性

- **多 Agent 协作对局**：每个玩家由独立的 LLM Agent 控制，支持狼人、预言家、女巫、猎人、村民五种角色
- **信息隔离**：严格按角色过滤可见信息，狼人知道同伴身份，预言家查验结果仅本人可见
- **自演化系统**：每局结束后自动反思 → 改进 prompt → 下局验证，策略在迭代中逐步专精
- **经验积累**：SummaryAgent 生成结构化复盘，经验跨游戏持久化并注入后续对局
- **多决策风格**：支持谨慎、大胆、随机、平衡四种风格，模拟不同玩家性格
- **前端观战 UI**：Vue 3 + Vite 单页应用，支持 AI 观战与自演化控制台

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      自演化主循环                            │
│                                                             │
│  ┌───────────┐    ┌──────────┐    ┌──────────────┐          │
│  │ GameEngine │───→│ Metrics  │───→│ SelfReflector│          │
│  │  (对局引擎) │    │Collector │    │  (LLM 反思)  │          │
│  └───────────┘    └──────────┘    └──────┬───────┘          │
│       ↑                                  │                  │
│       │              改进后的 prompt       │                  │
│       │         ┌────────────────────────┘                  │
│       │         ↓                                           │
│  ┌────┴─────┐  ┌──────────────┐  ┌───────────────┐          │
│  │PlayerAgent│←─│ PromptStore  │←─│   Experience  │          │
│  │ (LLM决策) │  │ (版本管理)    │  │  (经验系统)    │          │
│  └──────────┘  └──────────────┘  └───────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

| 模块 | 路径 | 职责 |
|------|------|------|
| LLMClient | [llm/client.py](llm/client.py) | OpenAI 兼容 API 客户端，内置限流与重试 |
| PromptStore | [prompt_store/store.py](prompt_store/store.py) | 角色 prompt 读写与版本管理 |
| PlayerAgent | [agent/player_agent.py](agent/player_agent.py) | 从 prompt 文件加载策略并决策（夜间/发言/投票） |
| SummaryAgent | [agent/summary_agent.py](agent/summary_agent.py) | 游戏结束后生成结构化复盘总结 |
| MetricsCollector | [metrics/collector.py](metrics/collector.py) | 胜负、投票正确率、击杀效率等量化指标 |
| SelfReflector | [agent/reflector.py](agent/reflector.py) | LLM 驱动的 prompt 自反思与改进 |
| GameEngine | [engine/game_engine.py](engine/game_engine.py) | 游戏裁判：流程控制、规则判定、对话记录 |
| GameState | [engine/state.py](engine/state.py) | 游戏状态管理与信息隔离 |
| Experience | [memory/experience.py](memory/experience.py) | 跨游戏经验持久化与注入 |
| evolve.py | [evolve.py](evolve.py) | 自演化主入口 CLI |
| API Server | [api/server.py](api/server.py) | FastAPI 服务端（游戏 + 演化 + Prompt 浏览） |

## 角色说明

| 角色 | 阵营 | 夜间能力 | 胜利条件 |
|------|------|----------|----------|
| 狼人 | 邪恶 | 与同伴投票击杀一人 | 所有神职死亡 或 所有村民死亡 |
| 预言家 | 善良 | 查验一名玩家身份 | 消灭所有狼人 |
| 女巫 | 善良 | 解药救人 / 毒药杀人（各一次，同夜不双用） | 消灭所有狼人 |
| 猎人 | 善良 | 被杀/被投时开枪带走一人（毒杀除外） | 消灭所有狼人 |
| 村民 | 善良 | 无 | 消灭所有狼人 |

## 角色配置

| 配置名 | 说明 | 人数 |
|--------|------|------|
| `simple_4` | 1 狼、1 预言家、1 女巫、1 村民 | 4 |
| `standard_6` | 2 狼、预言家、女巫、猎人、村民 | 6 |
| `big_9` | 3 狼、预言家、女巫、猎人、2 村民、白痴（白痴暂未实现） | 9 |

## 环境准备

```bash
cd newevo-werewolf-agent
pip install -r requirements.txt
```

设置 API Key（二选一）：

```bash
# 方式 1：环境变量
# Windows PowerShell
$env:MIMO_API_KEY = "your-api-key"
# Linux / macOS
export MIMO_API_KEY="your-api-key"

# 方式 2：配置文件 config/llm_config.json 中的 api_key 字段
```

配置文件说明：
- `config/llm_config.json` — LLM 连接配置（`base_url`、`model`、`temperature` 等）
- `config/system_config.json` — 系统级配置（默认模型、API 地址）

## 使用方式

### 自演化 CLI

```bash
# 运行 5 局自演化（默认 standard_6）
python evolve.py --rounds 5

# 4 人简易局，2 局
python evolve.py --rounds 2 --config simple_4

# 查看狼人 prompt 演化历史
python evolve.py --history werewolf

# 回滚到指定版本
python evolve.py --rollback werewolf v001_20260528_1430
```

### 单局演示

```bash
# 默认 6 人局，自动推进
python main_demo.py

# 4 人简易局
python main_demo.py -c simple_4

# 逐天模式（每天暂停等待用户确认）
python main_demo.py --day-by-day

# 交互模式（每步暂停）
python main_demo.py -i

# 自定义玩家风格
python main_demo.py -s '{"0":"bold","1":"cautious","2":"balanced"}'
```

### 前端观战 UI

Vue 3 + Vite 单页应用，支持 **AI 观战**（单局逐步/自动推进）与 **自演化控制台**（多局演化 + Prompt 预览）。

**开发模式（前后端分离）：**

```bash
# 终端 1 — 启动 API
python run_server.py

# 终端 2 — 启动前端
cd frontend
npm install
npm run dev
```

浏览器打开 http://localhost:5173

**生产模式（单端口）：**

```bash
cd frontend && npm run build
python run_server.py
```

浏览器打开 http://localhost:8000

## 自演化工作原理

```
第 1 局                    第 2 局                    第 3 局
┌─────────┐              ┌─────────┐              ┌─────────┐
│ 初始     │   反思改进   │ v2 策略  │   反思改进   │ v3 策略  │
│ Prompt  │────────────→│ Prompt  │────────────→│ Prompt  │
└────┬────┘              └────┬────┘              └────┬────┘
     │                       │                       │
     ▼                       ▼                       ▼
  运行对局                 运行对局                 运行对局
     │                       │                       │
     ▼                       ▼                       ▼
  收集指标                 收集指标                 收集指标
  (投票准确率、             (改进后指标               (进一步
   击杀效率等)               应有提升)                专精)
```

1. **运行对局**：GameEngine 驱动完整对局流程，PlayerAgent 根据当前 prompt 做决策
2. **收集指标**：MetricsCollector 提取各角色的量化表现（投票正确率、击杀效率等）
3. **LLM 反思**：SelfReflector 将当前 prompt + 指标 + 日志发给 LLM，生成改进版 prompt
4. **版本存档**：旧 prompt 保存到 `prompt_versions/`，新 prompt 写入 `prompts/agents/`
5. **经验积累**：SummaryAgent 为每个玩家生成复盘总结，保存到 `memory/experiences/`
6. **迭代验证**：下一轮使用更新后的 prompt + 历史经验，验证策略是否改进

## 测试

```bash
# 运行全部单元测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_llm_client.py tests/test_prompt_store.py \
  tests/test_metrics_collector.py tests/test_self_reflector.py -v

# 端到端自演化测试（需有效 MIMO_API_KEY）
python evolve.py --rounds 2 --config simple_4
```

## 目录结构

```
newevo-werewolf-agent/
├── evolve.py                # 自演化主入口 CLI
├── main_demo.py             # 单局演示入口
├── run_server.py            # API 服务启动
│
├── engine/                  # 游戏引擎
│   ├── game_engine.py       #   核心裁判（流程控制、规则判定）
│   ├── state.py             #   游戏状态管理（信息隔离）
│   ├── player.py            #   玩家数据结构（桥接 Role 与 Agent）
│   └── phase.py             #   游戏阶段枚举
│
├── roles/                   # 角色实现
│   ├── base.py              #   角色基类（类型、阵营、技能接口）
│   ├── werewolf.py          #   狼人（夜间击杀、同伴感知）
│   ├── seer.py              #   预言家（夜间查验）
│   ├── witch.py             #   女巫（解药/毒药）
│   ├── hunter.py            #   猎人（死亡开枪）
│   └── villager.py          #   村民（无特殊能力）
│
├── agent/                   # AI 代理
│   ├── player_agent.py      #   玩家代理（LLM 决策 + JSON 解析）
│   ├── summary_agent.py     #   总结代理（游戏复盘生成）
│   └── reflector.py         #   自反思器（prompt 改进）
│
├── llm/                     # LLM 客户端
│   └── client.py            #   OpenAI 兼容 API（限流 + 重试）
│
├── prompt_store/            # Prompt 存储
│   └── store.py             #   版本管理（读写、历史、回滚）
│
├── memory/                  # 经验系统
│   └── experience.py        #   跨游戏经验持久化与注入
│
├── metrics/                 # 指标收集
│   └── collector.py         #   量化表现指标提取
│
├── schema/                  # 数据模型
│   ├── game_record.py       #   游戏记录（Pydantic）
│   ├── game_logger.py       #   游戏日志记录器
│   └── system_config.py     #   系统配置
│
├── api/                     # Web API
│   ├── server.py            #   FastAPI 服务端
│   ├── models.py            #   请求/响应模型
│   └── evolve_service.py    #   自演化后台任务服务
│
├── prompts/                 # Prompt 文件
│   ├── agents/              #   各角色当前策略 Agent.md
│   └── meta/                #   反思器元提示词
│
├── prompt_versions/         # Prompt 历史版本（自演化产物）
├── memory/experiences/      # 经验文件（按角色 JSON 存储）
├── config/                  # 配置文件
├── frontend/                # Vue 3 前端
└── tests/                   # 单元与集成测试
```

## 作业说明

本项目对应第 16 周作业：**AI 狼人杀 — 多智能体协作与博弈**，进阶方向 ① 通用 Agent 自演化（从通用策略演化为多角色专精 Agent）。

核心创新点：
- **Prompt 自演化**：LLM 自动评估策略效果并改写 prompt，无需人工调参
- **经验 + 策略双驱动**：PromptStore 管理策略指令，Experience 系统积累实战教训
- **信息隔离**：严格按角色过滤可见信息，模拟真实狼人杀的信息不对称
- **四层 JSON 容错**：LLM 输出格式不完美时仍能提取有效决策
