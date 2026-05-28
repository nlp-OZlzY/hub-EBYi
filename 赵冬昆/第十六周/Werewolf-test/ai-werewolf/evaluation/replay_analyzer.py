"""
复盘分析器 - 深度分析游戏过程
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

class CriticalMomentType(str, Enum):
    KEY_VOTE = "key_vote"           # 关键投票
    ROLE_REVEAL = "role_reveal"     # 身份暴露
    MISJUDGMENT = "misjudgment"     # 误判时刻
    TEAMWORK = "teamwork"           # 团队协作
    TURNING_POINT = "turning_point" # 转折点

@dataclass
class CriticalMoment:
    """关键时刻"""
    moment_type: CriticalMomentType
    round_number: int
    phase: str
    description: str
    involved_players: List[int]
    impact_score: float  # 影响程度 (0-10)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.moment_type.value,
            "round": self.round_number,
            "phase": self.phase,
            "description": self.description,
            "players": self.involved_players,
            "impact": self.impact_score
        }

@dataclass
class PlayerDecision:
    """玩家决策分析"""
    player_id: int
    round_number: int
    phase: str
    decision: str
    rationale: str
    outcome: str
    was_optimal: bool
    alternative_actions: List[str]

class ReplayAnalyzer:
    """复盘分析器"""
    
    def __init__(self):
        self.critical_moments: List[CriticalMoment] = []
        self.player_decisions: List[PlayerDecision] = []
    
    def analyze_game(self, game_state: Any) -> Dict[str, Any]:
        """分析整局游戏"""
        self.critical_moments = []
        self.player_decisions = []
        
        # 分析关键时刻
        self._analyze_voting_patterns(game_state)
        self._analyze_role_reveals(game_state)
        self._analyze_turning_points(game_state)
        
        # 分析玩家决策
        self._analyze_player_decisions(game_state)
        
        return {
            "game_id": game_state.game_id,
            "critical_moments_count": len(self.critical_moments),
            "critical_moments": [m.to_dict() for m in self.critical_moments],
            "decision_analysis": self._summarize_decisions(),
            "key_insights": self._generate_insights(game_state)
        }
    
    def _analyze_voting_patterns(self, game_state: Any):
        """分析投票模式"""
        try:
            from engine.types import Team
        except ImportError:
            from ..engine.types import Team
        
        for dialogue in game_state.dialogues:
            if dialogue.get("phase") == "vote":
                votes = dialogue.get("votes", {})
                eliminated = dialogue.get("eliminated")
                
                if eliminated is not None:
                    eliminated_player = game_state.get_player_by_id(eliminated)
                    
                    # 检查是否投出了关键狼人
                    if eliminated_player and eliminated_player.role.value == "werewolf":
                        moment = CriticalMoment(
                            moment_type=CriticalMomentType.KEY_VOTE,
                            round_number=game_state.day_number,
                            phase="vote",
                            description=f"成功投票放逐狼人玩家 {eliminated}",
                            involved_players=[int(v) for v in votes.keys()],
                            impact_score=8.0
                        )
                        self.critical_moments.append(moment)
                    
                    # 检查是否误投好人
                    elif eliminated_player and eliminated_player.team == Team.GOOD:
                        moment = CriticalMoment(
                            moment_type=CriticalMomentType.MISJUDGMENT,
                            round_number=game_state.day_number,
                            phase="vote",
                            description=f"误投好人玩家 {eliminated}",
                            involved_players=[int(v) for v in votes.keys()],
                            impact_score=6.0
                        )
                        self.critical_moments.append(moment)
    
    def _analyze_role_reveals(self, game_state: Any):
        """分析身份暴露时刻"""
        for dialogue in game_state.dialogues:
            content = dialogue.get("content", "")
            speaker_id = dialogue.get("speaker_id")
            
            # 检测身份宣称
            if "我是预言家" in content or "我是女巫" in content:
                moment = CriticalMoment(
                    moment_type=CriticalMomentType.ROLE_REVEAL,
                    round_number=game_state.day_number,
                    phase=dialogue.get("phase", "unknown"),
                    description=f"玩家 {speaker_id} 宣称身份",
                    involved_players=[speaker_id] if speaker_id else [],
                    impact_score=7.0
                )
                self.critical_moments.append(moment)
    
    def _analyze_turning_points(self, game_state: Any):
        """分析转折点"""
        # 分析死亡顺序对局势的影响
        deaths_by_round: Dict[int, List[int]] = {}
        
        for dialogue in game_state.dialogues:
            if dialogue.get("phase") == "night_result":
                deaths = dialogue.get("deaths", [])
                round_num = dialogue.get("round", game_state.day_number)
                deaths_by_round[round_num] = deaths
        
        # 检查关键角色死亡
        for round_num, deaths in deaths_by_round.items():
            for death_id in deaths:
                player = game_state.get_player_by_id(death_id)
                if player and player.role.value in ["seer", "witch"]:
                    moment = CriticalMoment(
                        moment_type=CriticalMomentType.TURNING_POINT,
                        round_number=round_num,
                        phase="night",
                        description=f"关键角色 {player.role.value} (玩家 {death_id}) 死亡",
                        involved_players=[death_id],
                        impact_score=9.0
                    )
                    self.critical_moments.append(moment)
    
    def _analyze_player_decisions(self, game_state: Any):
        """分析玩家决策"""
        # 简化版本：记录主要决策点
        for dialogue in game_state.dialogues:
            phase = dialogue.get("phase")
            speaker_id = dialogue.get("speaker_id")
            content = dialogue.get("content", "")
            
            if phase == "speech" and speaker_id is not None:
                decision = PlayerDecision(
                    player_id=speaker_id,
                    round_number=game_state.day_number,
                    phase=phase,
                    decision="发言",
                    rationale=content[:50] + "..." if len(content) > 50 else content,
                    outcome="待评估",
                    was_optimal=True,  # 简化
                    alternative_actions=["保持沉默", "悍跳", "自爆"]
                )
                self.player_decisions.append(decision)
    
    def _summarize_decisions(self) -> Dict[str, Any]:
        """总结决策分析"""
        if not self.player_decisions:
            return {}
        
        total_decisions = len(self.player_decisions)
        optimal_decisions = sum(1 for d in self.player_decisions if d.was_optimal)
        
        return {
            "total_decisions": total_decisions,
            "optimal_decisions": optimal_decisions,
            "optimal_rate": round(optimal_decisions / total_decisions * 100, 2) if total_decisions > 0 else 0
        }
    
    def _generate_insights(self, game_state: Any) -> List[str]:
        """生成关键洞察"""
        insights = []
        
        # 分析获胜方
        if game_state.winner:
            winner_text = "好人阵营" if game_state.winner.value == "good" else "狼人阵营"
            insights.append(f"本局 {winner_text} 获胜")
        
        # 分析关键时刻数量
        if len(self.critical_moments) > 5:
            insights.append("本局游戏非常激烈，出现了多个关键时刻")
        elif len(self.critical_moments) < 2:
            insights.append("本局游戏相对平淡，缺乏关键转折点")
        
        # 分析投票模式
        vote_moments = [m for m in self.critical_moments if m.moment_type == CriticalMomentType.KEY_VOTE]
        if len(vote_moments) >= 2:
            insights.append("好人阵营投票准确率较高")
        
        misjudge_moments = [m for m in self.critical_moments if m.moment_type == CriticalMomentType.MISJUDGMENT]
        if len(misjudge_moments) >= 2:
            insights.append("好人阵营出现多次误判")
        
        return insights
    
    def generate_narrative(self, game_state: Any) -> str:
        """生成叙事性复盘"""
        analysis = self.analyze_game(game_state)
        
        narrative = f"""
=== 游戏叙事复盘 ===
游戏ID: {game_state.game_id}

【游戏概况】
本局游戏共进行了 {game_state.day_number} 天
获胜方: {game_state.winner.value if game_state.winner else '未知'}

【关键时刻】
"""
        
        for i, moment in enumerate(self.critical_moments[:5], 1):  # 前5个关键时刻
            narrative += f"""
{i}. {moment.moment_type.value} (第{moment.round_number}天)
   {moment.description}
   影响程度: {'★' * int(moment.impact_score)}
"""
        
        narrative += """
【关键洞察】
"""
        for insight in analysis["key_insights"]:
            narrative += f"- {insight}\n"
        
        narrative += """
【决策分析】
"""
        decision_summary = analysis["decision_analysis"]
        if decision_summary:
            narrative += f"""
总决策数: {decision_summary['total_decisions']}
最优决策: {decision_summary['optimal_decisions']} ({decision_summary['optimal_rate']}%)
"""
        
        return narrative
