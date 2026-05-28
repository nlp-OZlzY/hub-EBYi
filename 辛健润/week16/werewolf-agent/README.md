# AI 狼人杀 - 多智能体协作与博弈系统

基于多智能体协作框架，构建能够自主完成信息不对称博弈的狼人杀 Agent Team 系统。

## 核心特性

- **多智能体协作/对抗**: 每个 Agent 根据角色（狼人、预言家、女巫等）拥有独立目标、策略与行动空间
- **严格信息隔离**: 在信息不对称约束下进行推理、发言与决策
- **完整对局引擎**: 驱动回合流转与胜负裁决
- **全程可观测**: 输出结构化日志
- **自演化系统**: 从「通用 Agent」演化为「狼人杀多角色 Agent」

## 技术栈

- **语言**: Python 3.10+
- **框架**: LangGraph / LangChain
- **LLM**: DeepSeek (支持国产模型)
- **日志**: 结构化日志输出

## 角色系统

| 角色 | 阵营 | 能力 |
|------|------|------|
| 狼人 | 狼人 | 每晚刀人，知道其他狼人 |
| 预言家 | 好人 | 每晚验人，知道目标阵营 |
| 女巫 | 好人 | 救人/毒人 |
| 守卫 | 好人 | 守人，不能连守同一人 |
| 村民 | 好人 | 无特殊能力 |
| 猎人 | 好人 | 死亡可开枪 |

## 项目结构

```
werewolf-agent/
├── core/                    # 核心引擎
│   ├── game_engine.py      # 对局引擎
│   ├── role_system.py      # 角色系统
│   ├── message_bus.py      # 消息总线
│   └── event_system.py     # 事件系统
├── agents/                  # Agent层
│   ├── base_agent.py       # 通用Agent基类
│   └── role_agents/        # 角色化Agent
├── llm/                     # LLM接口层
├── utils/                   # 工具函数
└── main.py                  # 入口文件
```

## 环境要求

- Python 3.10+
- DeepSeek API Key

## 安装

```bash
pip install langchain langgraph deepseek
```

## 使用方法

```python
from core.game_engine import GameEngine

engine = GameEngine()
engine.start()
```

## License

MIT