# AI 狼人杀 - 多智能体博弈系统

基于 Qwen 大模型的狼人杀 AI 对战系统，支持多智能体博弈、游戏评测和前端观战。

## 项目结构

```
ai-werewolf/
├── agent/                    # AI Agent 模块
│   ├── core/                 # 核心Agent实现
│   │   ├── __init__.py
│   │   └── base_agent.py     # 基础Agent类（思考、发言、投票）
│   ├── evolution/            # 自我进化模块
│   │   └── self_evolution.py
│   ├── llm/                  # 大模型客户端
│   │   └── qwen_client.py    # Qwen API客户端（DashScope）
│   ├── prompts/              # 提示词模板
│   │   ├── __init__.py
│   │   └── template_loader.py
│   └── strategies/           # 策略模块
│       ├── __init__.py
│       ├── base_strategy.py  # 基础策略类
│       └── simple_strategy.py # 简单策略实现
│
├── api/                      # API服务
│   └── server.py             # FastAPI服务器
│
├── configs/                  # 配置模块
│   └── game_configs.py       # 游戏配置
│
├── engine/                   # 游戏引擎
│   ├── plugins/              # 角色插件
│   │   ├── base_role.py      # 基础角色类
│   │   ├── werewolf.py       # 狼人
│   │   ├── seer.py           # 预言家
│   │   ├── witch.py          # 女巫
│   │   ├── hunter.py         # 猎人
│   │   └── villager.py       # 平民
│   ├── rules/                # 游戏规则
│   │   └── win_conditions.py # 胜利条件
│   ├── game_manager.py       # 游戏管理器
│   ├── state_machine.py      # 状态机
│   └── types.py              # 类型定义
│
├── evaluation/               # 评测模块
│   ├── api.py                # 评测API
│   ├── leaderboard.py        # 排行榜
│   ├── metrics.py            # 评测指标
│   └── replay_analyzer.py    # 回放分析
│
├── frontend/                 # 前端观战UI（Vue3 + Vite）
│   ├── src/
│   │   ├── components/       # Vue组件
│   │   │   ├── GameBoard.vue # 游戏面板
│   │   │   ├── Header.vue    # 头部导航
│   │   │   └── Leaderboard.vue # 排行榜
│   │   ├── App.vue           # 主应用
│   │   ├── main.js           # 入口文件
│   │   └── style.css         # 样式文件
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js        # Vite配置
│   └── tailwind.config.js    # Tailwind CSS配置
│
├── tests/                    # 测试模块
│   ├── test_game_manager.py
│   └── test_role_plugins.py
│
├── demo_evolution/           # 进化演示数据
│   └── agent_0_evolution.json
│
├── example.py                # 基础示例
├── example_qwen.py           # Qwen模型演示（主要入口）
├── example_evaluation.py     # 评测示例
├── cli.py                    # 命令行工具
├── requirements.txt          # Python依赖
└── pyproject.toml            # 项目配置
```

## 快速开始

### 1. 安装依赖

```bash
cd ai-werewolf
pip install -r requirements.txt
```

### 2. 配置 API Key

在 `example_qwen.py` 中设置你的 DashScope API Key：

```python
api_key = "your-dashscope-api-key"
```

### 3. 运行游戏

```bash
python example_qwen.py
```

### 4. 启动前端观战UI

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173 查看游戏界面。

## 核心功能

### 1. AI Agent

- **思考 (think)**: 分析当前局势，推理行动策略
- **发言 (speak)**: 根据角色身份生成策略性发言
- **投票 (vote)**: 基于分析结果投票给嫌疑玩家

### 2. 游戏引擎

- 完整的狼人杀游戏流程（夜晚→白天→投票）
- 支持多种角色：狼人、预言家、女巫、猎人、平民
- 自动判定胜负条件

### 3. 评测系统

- 玩家得分计算
- MVP评选
- 游戏回放分析
- 排行榜系统

### 4. 前端观战

- 实时游戏状态展示
- 玩家存活状态
- 对话记录
- 阵营统计

## 技术栈

- **后端**: Python 3.10+, FastAPI, Pydantic
- **AI模型**: 阿里云 DashScope (Qwen3.6-35b-a3b)
- **前端**: Vue 3, Vite, Tailwind CSS
- **测试**: pytest

## API 配置

使用阿里云 DashScope OpenAI 兼容接口：

```python
from agent.llm.qwen_client import QwenClient

client = QwenClient(
    api_key="your-api-key",
    model="qwen3.6-35b-a3b",
    region="cn-beijing"  # 可选: us-virginia, singapore, eu-frankfurt
)
```

## 游戏配置

```python
from engine.types import GameConfig

config = GameConfig(
    name="测试局",
    total_players=6,
    werewolf_count=2,
    seer_count=1,
    witch_count=1,
    villager_count=2
)
```

## 许可证

MIT License
