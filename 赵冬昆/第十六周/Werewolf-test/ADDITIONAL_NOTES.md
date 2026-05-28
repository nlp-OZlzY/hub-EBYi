这是一个非常棒的项目构思。为了让这个项目在“Vibe Coding”过程中更加顺畅，且最终成品具备更高的技术深度与可玩性，我在您原有需求的基础上，补充了以下专家级建议。这些建议旨在解决多智能体博弈中常见的“幻觉”、“死循环”和“策略单一”问题，并提升系统的工程化水平。

💡 专家级增强建议

引入“思维链（CoT）强制结构化”
    痛点： LLM 直接输出动作容易导致逻辑跳跃或忘记规则。
    方案： 在 System Prompt 中强制要求 Agent 在输出 JSON Action 之前，必须先输出一段  标签包裹的思考过程。思考过程必须包含：当前局势分析 -> 我的角色目标 -> 可用策略枚举 -> 风险评估 -> 最终决策。
    价值： 显著提升决策质量，且为后续的“复盘归因”提供完美数据源。

构建“动态人格与风格系统”
    痛点： AI 说话千篇一律，缺乏“人味”，导致博弈体验枯燥。
    方案： 为每个 Agent 实例化时注入独立的 Persona Card（如：激进型、保守型、逻辑怪、情绪流）。将人格参数作为 Prompt 的顶层约束，影响发言长度、用词习惯及风险偏好。
    价值： 增加对局的多样性与观赏性，是评测“拟人度”的关键维度。

实现“上帝视角的元认知监控器”
    痛点： Agent 陷入无效争吵或重复发言，游戏无法推进。
    方案： 在游戏引擎层增加一个轻量级 Meta-Monitor（可由小模型或规则驱动）。当检测到连续 N 轮信息熵过低或重复模式时，自动触发“法官干预”机制（如：强制跳过发言、提示玩家总结观点、或注入随机事件打破僵局）。
    价值： 保证系统鲁棒性，避免 Vibe Coding 时因边界情况导致程序卡死。

设计“模块化插件式角色架构”
    痛点： 新增角色（如白痴、守卫）需要修改核心引擎代码，耦合度高。
    方案： 采用 RolePlugin 接口设计。每个角色是一个独立插件，通过钩子函数注册到引擎（如 on_night_start, on_vote_cast, on_death）。引擎只负责调度，不感知具体角色逻辑。
    价值： 极大扩展性，方便后续做“自演化”时动态生成新角色能力。

前端增加“信念状态可视化”
    痛点： 观众只能看到发言，看不到 Agent “心里怎么想的”。
    方案： 在前端为每个玩家增加一个“信任度矩阵”热力图。实时展示该 Agent 对其他玩家的怀疑/信任分数变化。
    价值： 将抽象的推理过程具象化，这是区别于普通游戏Demo的核心亮点。

以下是整合了上述建议及您原始需求的终极版项目规格文档。您可以直接保存为 AI_WEREWOLF_MASTER_SPEC.md。

🐺 AI 狼人杀：多智能体协作与博弈 Agent Team 终极实战指南

文档用途：本文档是 AI 辅助编程（Vibe Coding）的唯一真理来源。所有代码生成、架构设计、Prompt 工程均需严格遵循此规范。
版本：v2.0 (Enhanced)

项目核心定义

1.1 愿景
构建一个高可信、可观测、可进化的多智能体狼人杀博弈平台。不仅是游戏模拟，更是研究 LLM 在信息不对称、多方博弈、社会推理场景下的行为实验室。

1.2 核心差异化特征
🧠 结构化思维链：拒绝黑盒决策，所有行动必有 CoT 支撑。
🎭 动态人格系统：Agent 具备独立性格参数，拒绝同质化发言。
🔌 插件化角色架构：角色逻辑与引擎解耦，支持热插拔与自生成。
👁️ 信念可视化：前端实时渲染 Agent 内心的信任/怀疑矩阵。
🛡️ 元认知监控：内置防死循环与信息熵监测，保障对局流畅。

系统架构蓝图

2.1 分层架构
层级   组件   关键职责   技术栈推荐
表现层   Vue3 Watchtower   实时看板、信念热力图、人机交互、日志流   Vue3 + Pinia + ECharts

接口层   FastAPI Gateway   RESTful API、WebSocket 推送、鉴权、限流   FastAPI + Uvicorn

引擎层   Game Orchestrator   状态机流转、规则校验、元监控、插件调度   Python + StateMachine

智能层   Agent Runtime   LLM 调用、CoT 解析、记忆管理、人格注入   LangChain / AutoGen

数据层   Structured Store   对局日志、评测指标、配置模板   SQLite / PostgreSQL + JSONL

2.2 目录结构规范
text
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

核心机制详解

3.1 强制结构化输出协议
所有 Agent 的输出必须严格符合以下 Schema，引擎层进行 JSON Schema 校验，失败则触发重试或兜底。

json
{
  "inner_monologue": "string // 标签包裹的思考过程",
  "action": {
    "type": "speak | vote | skill | pass",
    "target": "player_id | null",
    "content": "string // 发言内容或技能参数"
  },
  "belief_update": {
    "player_id": 0.8 // ⭐ 本轮对该玩家的信任度更新 (-1.0 ~ 1.0)
  }
}

3.2 动态人格系统
每个 Agent 初始化时加载 PersonaCard：
yaml
name: "老谋深算的村长"
traits: ["conservative", "logical", "verbose"]
risk_tolerance: 0.2
speech_style: "使用谚语，语速慢，喜欢引用历史对局"
hidden_agenda: "优先保护预言家，即使牺牲自己"
Vibe Coding 提示：将 Persona 作为 System Prompt 的第一段，确保人格贯穿始终。

3.3 元认知监控器
在每轮发言结束后执行检查：
信息熵检测：若连续3轮发言相似度 > 0.9，触发“法官提醒：请提供新信息”。
死锁检测：若投票平票超过2次，触发“加时赛”或“随机放逐”机制。
违规检测：若 Agent 试图说出夜间信息，立即拦截并替换为“我昨晚睡得很香”。

3.4 插件化角色接口
python
class RolePlugin(ABC):
    @abstractmethod
    def on_night_start(self, context: GameContext) -> Optional[Action]: ...
    
    @abstractmethod
    def on_speech_phase(self, context: GameContext) -> str: ...
    
    @abstractmethod
    def get_valid_actions(self, phase: Phase) -> List[str]: ...
    
    @property
    def win_condition(self) -> Callable[[GameState], bool]: ...

游戏规则基准 (6人屠边制)

4.1 阵营与胜利条件
阵营   角色   胜利条件
好人   预言家、女巫、猎人、村民   所有狼人死亡

狼人   基础狼人   屠边成功（全神死 OR 全民死）

4.2 关键约束
女巫：单局限一次解药/毒药；不可同夜双药；不可自救；被毒者不能开枪。
猎人：被刀/被投可开枪；被毒不可开枪。
狼人：夜间可团队密聊；白天可自爆直接进入黑夜。
信息隔离：严禁将全局状态传入任何非上帝视角的 Agent Prompt。

4.3 状态机流转
mermaid
graph TD
    A[天黑] --> B[狼人刀人]
    B --> C[预言家查验]
    C --> D[女巫用药]
    D --> E[结算伤亡]
    E --> F{首日?}
    F -- 是 --> G[警长竞选]
    F -- 否 --> H[公布死讯]
    G --> H
    H --> I[轮流发言]
    I --> J[公投放逐]
    J --> K{胜负判定}
    K -- 未结束 --> A
    K -- 结束 --> L[终局复盘]

进阶研究方向（三选一深化）

🧬 A. 通用 Agent 自演化
核心：Read-Eval-Modify Loop
实现：Agent 在对局后自动生成 Self-Critique Report，基于报告重写自己的 Strategy Module 或 Prompt Template，下一局加载新版本。
验证：记录每代版本的胜率曲线与策略复杂度。

📊 B. 多维评测与 Leaderboard
核心：Process + Outcome Evaluation
指标体系：
   结果指标：胜率、MVP率、挡刀成功率
   过程指标：发言信息量、逻辑一致性得分、伪装成功率、投票准确率
产出：自动化复盘报告 + 跨模型竞技天梯。

🔄 C. RLAIF 自进化闭环
核心：Gameplay as Data
流程：高质量对局 → 自动标注关键决策点 → 构造 SFT/DPO 数据 → 微调专用小模型 → 部署回系统。
目标：让 Agent 自发习得“倒钩”、“悍跳”等人类高阶战术。

Vibe Coding 执行路线图

Phase 1: 最小可运行内核
实现 engine/state_machine.py + 单元测试
实现 agent/core/base_agent.py + JSON 输出解析器
实现基础角色插件（狼人、村民）
CLI 跑通一局纯 AI 对战

Phase 2: 完整博弈体验
补全所有角色插件 + 元认知监控器
实现 FastAPI 后端 + WebSocket 推送
实现 Vue3 前端基础看板 + 日志流
接入真实 LLM API，调试 Prompt

Phase 3: 增强与进阶
实现动态人格系统 + 信念可视化
选择并实现一个进阶方向
完善文档、测试覆盖率、部署脚本

快速启动命令

bash
后端
pip install -r requirements.txt
uvicorn api.server:app --reload --port 8000

前端
cd frontend && npm i && npm run dev

创建对局
curl -X POST http://localhost:8000/games \
  -H 'Content-Type: application/json' \
  -d '{"config":"standard_6","personas":["aggressive","logical","emotional"]}'

⚠️ 给 AI 编程助手的特别指令
永远先写测试：任何角色逻辑或状态转换，必须先有对应的 pytest 用例。
严格遵守 Schema：不要为了“聪明”而绕过 JSON 校验，鲁棒性优于灵活性。
日志即产品：每个关键决策点都必须写入结构化日志，这是后续所有进阶功能的基础。
人格不是装饰：Persona 必须影响决策权重，而不仅仅是改变说话语气。
信息隔离是红线：在 Code Review 时，重点检查是否有全局状态泄露到 Agent Context 中。

📥 如何使用这份文档

保存：将上述 Markdown 内容保存为 AI_WEREWOLF_MASTER_SPEC.md。
初始化项目：在您的 IDE（Cursor/Windsurf/Claude Code）中打开一个新文件夹。
首次对话：将文件拖入对话框，并发送以下 Prompt：
    > “请阅读 AI_WEREWOLF_MASTER_SPEC.md。我们现在开始 Phase 1 的开发。请先帮我搭建项目骨架，并实现 engine/state_machine.py 及其完整的单元测试。记住遵守文档中的所有约束。”
迭代开发：每完成一个模块，都引用该文档进行 Code Review：“请根据 Master Spec 检查这段代码是否符合‘强制结构化输出’和‘信息隔离’的要求。”

