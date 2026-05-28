"""自演化管理器"""
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import json
from agents.base_agent import BaseAgent, EvolutionProfile
from core.role_system import RoleType


@dataclass
class EvolutionConfig:
    """演化配置"""
    max_generations: int = 10
    min_games_for_evolution: int = 5
    win_rate_threshold: float = 0.6
    improvement_threshold: float = 0.1


@dataclass
class EvolutionRecord:
    """演化记录"""
    agent_id: str
    role_type: RoleType
    from_generation: int
    to_generation: int
    before_win_rate: float
    after_win_rate: float
    improvements: list[str]
    timestamp: str


class EvolutionManager:
    """自演化管理器

    实现 "对局→分析→优化→再对局" 的自进化循环
    """

    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        self.agents: dict[str, BaseAgent] = {}
        self.evolution_records: list[EvolutionRecord] = []
        self.game_history: list[dict] = []

    def register_agent(self, agent: BaseAgent):
        """注册 Agent"""
        self.agents[agent.agent_id] = agent

    def record_game(self, game_result: dict):
        """记录游戏结果"""
        self.game_history.append(game_result)

        # 通知各 Agent 分析结果
        for agent in self.agents.values():
            agent.evolve(game_result)

    def should_evolve(self, agent: BaseAgent) -> bool:
        """判断是否应该演化"""
        if not agent.evolution_profile:
            return False

        profile = agent.evolution_profile

        # 游戏数量不足
        if len(profile.performance_history) < self.config.min_games_for_evolution:
            return False

        # 胜率低于阈值，考虑优化
        return profile.win_rate < self.config.win_rate_threshold

    def evolve_agent(self, agent: BaseAgent) -> EvolutionRecord:
        """演化单个 Agent"""
        if not agent.evolution_profile:
            raise ValueError("Agent has no evolution profile")

        profile = agent.evolution_profile
        before_win_rate = profile.win_rate

        # 分析并生成改进建议
        evolution_result = agent.evolve({
            "decisions": [d for d in agent.get_memory().decisions],
            "won": profile.win_rate >= 0.5,
            "survived_days": profile.avg_survival
        })

        improvements = evolution_result.get("improvements", [])

        # 更新演化档案
        profile.generation += 1

        # 应用优化建议到 Prompt
        if improvements:
            new_prompt = self._apply_improvements(
                agent.get_prompt(),
                improvements
            )
            agent.update_prompt(new_prompt)

        # 记录演化
        record = EvolutionRecord(
            agent_id=agent.agent_id,
            role_type=profile.role_type,
            from_generation=profile.generation - 1,
            to_generation=profile.generation,
            before_win_rate=before_win_rate,
            after_win_rate=profile.win_rate,
            improvements=improvements,
            timestamp=datetime.now().isoformat()
        )
        self.evolution_records.append(record)

        return record

    def _apply_improvements(self, current_prompt: str, improvements: list[str]) -> str:
        """应用改进建议到 Prompt"""
        improvement_text = "\n".join(f"- {imp}" for imp in improvements)

        # 简化处理：直接在末尾添加优化建议
        new_prompt = current_prompt + f"\n\n## 策略优化\n{improvement_text}"

        return new_prompt

    def auto_evolve_all(self) -> list[EvolutionRecord]:
        """自动演化所有需要进化的 Agent"""
        evolved = []

        for agent in self.agents.values():
            if self.should_evolve(agent):
                try:
                    record = self.evolve_agent(agent)
                    evolved.append(record)
                except Exception as e:
                    print(f"Evolution failed for {agent.agent_id}: {e}")

        return evolved

    def run_evolution_cycle(self, num_games: int = 10) -> dict:
        """运行一轮演化循环：多局对局 + 分析 + 优化"""
        print(f"\n=== 开始第 {len(self.game_history) // num_games + 1} 轮演化 ===")

        # 记录当前各角色胜率
        before_stats = self.get_stats_by_role()

        # 收集演化记录
        records = []

        # 自动演化
        evolved = self.auto_evolve_all()
        records.extend(evolved)

        print(f"本轮演化了 {len(evolved)} 个 Agent")
        for record in evolved:
            print(f"  - {record.agent_id}: 胜率 {record.before_win_rate:.2%} -> {record.after_win_rate:.2%}")

        # 返回统计
        after_stats = self.get_stats_by_role()

        return {
            "evolved_count": len(evolved),
            "before_stats": before_stats,
            "after_stats": after_stats,
            "evolution_records": records
        }

    def get_stats_by_role(self) -> dict:
        """按角色统计胜率"""
        stats = {}
        for agent in self.agents.values():
            if not agent.evolution_profile:
                continue

            role = agent.evolution_profile.role_type.value
            if role not in stats:
                stats[role] = {"total": 0, "wins": 0, "agents": []}

            stats[role]["total"] += 1
            stats[role]["wins"] += agent.evolution_profile.win_rate
            stats[role]["agents"].append(agent.agent_id)

        return {
            role: {
                "count": data["total"],
                "avg_win_rate": data["wins"] / data["total"] if data["total"] > 0 else 0
            }
            for role, data in stats.items()
        }

    def get_leaderboard(self) -> list[dict]:
        """获取排行榜"""
        leaderboard = []

        for agent in self.agents.values():
            if agent.evolution_profile:
                profile = agent.evolution_profile
                leaderboard.append({
                    "agent_id": agent.agent_id,
                    "role": profile.role_type.value,
                    "generation": profile.generation,
                    "win_rate": profile.win_rate,
                    "avg_survival": profile.avg_survival,
                    "total_games": len(profile.performance_history)
                })

        # 按胜率排序
        leaderboard.sort(key=lambda x: x["win_rate"], reverse=True)

        return leaderboard

    def export_evolution_log(self) -> list[dict]:
        """导出演化日志"""
        return [
            {
                "agent_id": r.agent_id,
                "role": r.role_type.value,
                "from_gen": r.from_generation,
                "to_gen": r.to_generation,
                "win_rate_change": r.after_win_rate - r.before_win_rate,
                "improvements": r.improvements
            }
            for r in self.evolution_records
        ]

    def get_agent_stats(self, agent_id: str) -> dict:
        """获取 Agent 统计"""
        agent = self.agents.get(agent_id)
        if not agent or not agent.evolution_profile:
            return {}

        profile = agent.evolution_profile

        return {
            "agent_id": agent_id,
            "role": profile.role_type.value,
            "generation": profile.generation,
            "win_rate": profile.win_rate,
            "total_games": len(profile.performance_history),
            "avg_survival": profile.avg_survival
        }