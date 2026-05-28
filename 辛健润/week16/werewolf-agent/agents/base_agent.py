"""通用 Agent 基类 - 实现自演化系统"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any
import json
from core.role_system import RoleType, Role, Camp
from core.game_engine import GameEngine, GameState, Player
from llm.base import BaseLLM


@dataclass
class AgentMemory:
    """Agent 记忆"""
    observations: list[dict] = field(default_factory=list)  # 观察记录
    decisions: list[dict] = field(default_factory=list)    # 决策记录
    outcomes: list[dict] = field(default_factory=list)     # 结果记录
    strategy_weights: dict[str, float] = field(default_factory=dict)  # 策略权重


@dataclass
class EvolutionProfile:
    """演化档案"""
    agent_id: str
    role_type: RoleType
    generation: int = 1
    win_rate: float = 0.0
    avg_survival: float = 0.0
    performance_history: list[dict] = field(default_factory=list)
    prompt_template: str = ""
    strategy_params: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """通用 Agent 基类

    实现「读懂自己→修改自己→运行自己」的自演化逻辑
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        llm: BaseLLM,
        role: Optional[Role] = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.llm = llm
        self.role = role
        self.memory = AgentMemory()
        self.evolution_profile: Optional[EvolutionProfile] = None

        # 如果有角色，创建演化档案
        if role:
            self.evolution_profile = EvolutionProfile(
                agent_id=agent_id,
                role_type=role.role_type,
                prompt_template=self._get_default_prompt()
            )

    def _get_default_prompt(self) -> str:
        """获取默认 Prompt 模板"""
        return """你是一个理性的人工智能玩家，在狼人杀游戏中扮演 {role_name}。

游戏规则：
- 好人阵营：预言家、女巫、守卫、猎人、村民
- 狼人阵营：狼人
- 每晚狼人可以刀人，好人可以行动
- 白天发言、投票放逐

你的能力：{abilities}
{role_description}

请根据当前局势和你的身份，做出最优决策。
"""

    def understand_self(self, game_state: GameState) -> dict:
        """第一阶段：读懂自己

        分析自己的角色、能力、当前处境
        """
        player = game_state.players.get(self.agent_id)
        if not player:
            return {"error": "player not found"}

        # 构建自我认知
        self_understanding = {
            "role": player.role.name,
            "role_type": player.role.role_type.value,
            "camp": player.role.camp.value,
            "abilities": player.role.abilities,
            "alive": player.alive,
            "day": game_state.day,
            "survival_status": self._evaluate_survival(game_state),
            "goals": self._define_goals(player.role),
        }

        # 记忆观察
        self.memory.observations.append({
            "phase": game_state.phase.value,
            "self_understanding": self_understanding,
            "day": game_state.day
        })

        return self_understanding

    def _evaluate_survival(self, game_state: GameState) -> dict:
        """评估存活状态"""
        player = game_state.players[self.agent_id]
        alive_players = [p for p in game_state.players.values() if p.alive]
        total_alive = len(alive_players)

        return {
            "total_alive": total_alive,
            "wolves_alive": len([p for p in alive_players if p.role.camp == Camp.WOLF]),
            "good_alive": len([p for p in alive_players if p.role.camp == Camp.GOOD]),
            "my_position": self._calculate_position(game_state)
        }

    def _calculate_position(self, game_state: GameState) -> int:
        """计算玩家位置"""
        return game_state.alive_players.index(self.agent_id)

    def _define_goals(self, role: Role) -> list[str]:
        """定义角色目标"""
        goals_map = {
            RoleType.WEREWOLF: [
                "隐藏身份，避免被投票出局",
                "引导舆论消灭好人",
                "保护同伴狼人",
                "扰乱好人视线"
            ],
            RoleType.SEER: [
                "找出狼人并公之于众",
                "保护自己的查验结果",
                "存活到关键轮次"
            ],
            RoleType.WITCH: [
                "合理使用药水",
                "隐藏身份",
                "关键时刻救人"
            ],
            RoleType.GUARD: [
                "守护关键好人",
                "保护自己"
            ],
            RoleType.VILLAGER: [
                "通过发言判断狼人",
                "配合好人阵营"
            ],
            RoleType.HUNTER: [
                "寻找狼人",
                "死亡时带走狼人"
            ]
        }
        return goals_map.get(role.role_type, [])

    def reason(self, game_state: GameState, context: dict) -> dict:
        """第二阶段：推理决策

        基于当前局势进行推理
        """
        self_understanding = self.understand_self(game_state)

        # 构建推理上下文
        reasoning_context = {
            "my_info": self_understanding,
            "game_status": self._get_game_status_summary(game_state),
            "public_info": context.get("public_info", {}),
            "private_info": context.get("private_info", {}),
            "recent_events": self._get_recent_events()
        }

        # 使用 LLM 进行推理
        reasoning_prompt = self._build_reasoning_prompt(reasoning_context)

        reasoning = self.llm.chat([
            {"role": "system", "content": reasoning_prompt},
            {"role": "user", "content": "请分析当前局势并给出你的推理。"}
        ])

        return {
            "reasoning": reasoning,
            "context": reasoning_context
        }

    def _get_game_status_summary(self, game_state: GameState) -> dict:
        """获取游戏状态摘要"""
        alive = [p for p in game_state.players.values() if p.alive]
        dead = [p for p in game_state.players.values() if not p.alive]

        return {
            "day": game_state.day,
            "phase": game_state.phase.value,
            "alive_players": [{"id": p.id, "name": p.name} for p in alive],
            "dead_players": [{"id": p.id, "name": p.name} for p in dead],
            "wolf_count": len([p for p in alive if p.role.camp == Camp.WOLF]),
            "good_count": len([p for p in alive if p.role.camp == Camp.GOOD])
        }

    def _get_recent_events(self) -> list[dict]:
        """获取最近事件"""
        return self.memory.observations[-3:] if self.memory.observations else []

    def _build_reasoning_prompt(self, context: dict) -> str:
        """构建推理 Prompt"""
        return f"""你是 {self.name}，扮演 {context['my_info']['role']}。

当前局势：
- 第 {context['game_status']['day']} 天
- {context['game_status']['phase']} 阶段
- 存活玩家：{len(context['game_status']['alive_players'])}
- 狼人数量：{context['game_status']['wolf_count']}
- 好人数量：{context['game_status']['good_count']}

你的身份：
- 角色：{context['my_info']['role']} ({context['my_info']['camp']})
- 能力：{', '.join(context['my_info']['abilities'])}

你的目标：
{chr(10).join('- ' + g for g in context['my_info']['goals'])}

请分析局势并进行推理。
"""

    @abstractmethod
    def act(self, game_state: GameState, context: dict) -> dict:
        """第三阶段：执行行动

        根据推理结果执行具体行动
        """
        pass

    def evolve(self, game_result: dict) -> dict:
        """第四阶段：演化优化

        分析对局结果，优化策略
        """
        # 记录结果
        self.memory.outcomes.append(game_result)

        # 更新演化档案
        if self.evolution_profile:
            self.evolution_profile.performance_history.append(game_result)

            # 计算胜率
            total_games = len(self.evolution_profile.performance_history)
            wins = sum(1 for g in self.evolution_profile.performance_history
                      if g.get("won", False))
            self.evolution_profile.win_rate = wins / total_games if total_games > 0 else 0

            # 分析并优化策略
            optimization = self._analyze_and_optimize(game_result)

            return {
                "analysis": optimization["analysis"],
                "improvements": optimization["improvements"],
                "new_generation": self.evolution_profile.generation + 1
            }

        return {}

    def _analyze_and_optimize(self, game_result: dict) -> dict:
        """分析与优化"""
        analysis_prompt = f"""分析这局游戏的得失：

结果：{'胜利' if game_result.get('won') else '失败'}
存活轮次：{game_result.get('survived_days', 0)}
死因：{game_result.get('death_reason', 'N/A')}

决策记录：
{json.dumps(game_result.get('decisions', []), ensure_ascii=False, indent=2)}

请分析：
1. 决策是否合理？
2. 有哪些可以改进的地方？
3. 提出具体的优化建议。
"""

        analysis = self.llm.chat([{"role": "user", "content": analysis_prompt}])

        # 根据分析结果更新策略
        improvements = self._extract_improvements(analysis)

        return {
            "analysis": analysis,
            "improvements": improvements
        }

    def _extract_improvements(self, analysis: str) -> list[str]:
        """提取优化建议"""
        # 简单的启发式提取
        improvements = []
        lines = analysis.split('\n')
        for line in lines:
            if '改进' in line or '优化' in line or '建议' in line:
                improvements.append(line.strip())

        return improvements

    def get_prompt(self) -> str:
        """获取当前 Prompt（用于注入角色）"""
        if not self.evolution_profile:
            return self._get_default_prompt()

        return self.evolution_profile.prompt_template or self._get_default_prompt()

    def update_prompt(self, new_prompt: str):
        """更新 Prompt（自演化）"""
        if self.evolution_profile:
            self.evolution_profile.prompt_template = new_prompt

    def get_memory(self) -> AgentMemory:
        """获取记忆"""
        return self.memory

    def export_profile(self) -> dict:
        """导出演化档案"""
        if not self.evolution_profile:
            return {}

        return {
            "agent_id": self.evolution_profile.agent_id,
            "role_type": self.evolution_profile.role_type.value,
            "generation": self.evolution_profile.generation,
            "win_rate": self.evolution_profile.win_rate,
            "avg_survival": self.evolution_profile.avg_survival,
            "performance_history": self.evolution_profile.performance_history,
            "strategy_params": self.evolution_profile.strategy_params
        }