"""
多维评测指标体系
包含结果指标和过程指标
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime

class MetricType(str, Enum):
    OUTCOME = "outcome"      # 结果指标
    PROCESS = "process"      # 过程指标

class MetricCategory(str, Enum):
    WIN_RATE = "win_rate"           # 胜率
    SURVIVAL = "survival"           # 生存率
    MVP_RATE = "mvp_rate"           # MVP率
    VOTE_ACCURACY = "vote_accuracy" # 投票准确率
    SPEECH_QUALITY = "speech_quality" # 发言质量
    DECEPTION = "deception"         # 伪装成功率
    TEAMWORK = "teamwork"           # 团队协作

@dataclass
class PlayerMetrics:
    """单个玩家的评测指标"""
    player_id: int
    role: str
    team: str
    
    # 结果指标
    is_winner: bool = False
    is_alive: bool = True
    is_mvp: bool = False
    survival_rounds: int = 0
    
    # 过程指标
    total_votes_cast: int = 0
    correct_votes: int = 0           # 投给敌对阵营
    total_speeches: int = 0
    speech_informativeness: float = 0.0  # 发言信息量 (0-1)
    
    # 角色特定指标
    inspection_accuracy: float = 0.0     # 预言家查验准确率
    save_success_rate: float = 0.0       # 女巫救人成功率
    poison_success_rate: float = 0.0     # 女巫毒人成功率
    kill_success_rate: float = 0.0       # 狼人刀人成功率
    
    # 综合评分
    overall_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "role": self.role,
            "team": self.team,
            "outcome": {
                "is_winner": self.is_winner,
                "is_alive": self.is_alive,
                "is_mvp": self.is_mvp,
                "survival_rounds": self.survival_rounds
            },
            "process": {
                "vote_accuracy": self.correct_votes / max(self.total_votes_cast, 1),
                "speech_quality": self.speech_informativeness,
                "role_performance": {
                    "inspection_accuracy": self.inspection_accuracy,
                    "save_success_rate": self.save_success_rate,
                    "poison_success_rate": self.poison_success_rate,
                    "kill_success_rate": self.kill_success_rate
                }
            },
            "overall_score": self.overall_score
        }

@dataclass
class GameEvaluation:
    """单局游戏的评测结果"""
    game_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config_name: str = ""
    total_rounds: int = 0
    winner: str = ""
    
    player_metrics: Dict[int, PlayerMetrics] = field(default_factory=dict)
    
    # 全局指标
    game_duration_seconds: float = 0.0
    total_dialogues: int = 0
    vote_tie_count: int = 0  # 平票次数
    
    def calculate_overall_scores(self):
        """计算所有玩家的综合评分"""
        for player_id, metrics in self.player_metrics.items():
            score = 0.0
            
            # 基础分：胜利+50，存活+20，MVP+30
            if metrics.is_winner:
                score += 50
            if metrics.is_alive:
                score += 20
            if metrics.is_mvp:
                score += 30
            
            # 过程分
            vote_accuracy = metrics.correct_votes / max(metrics.total_votes_cast, 1)
            score += vote_accuracy * 20
            score += metrics.speech_informativeness * 10
            
            # 角色特定分
            if metrics.role == "seer":
                score += metrics.inspection_accuracy * 15
            elif metrics.role == "witch":
                score += metrics.save_success_rate * 10
                score += metrics.poison_success_rate * 10
            elif metrics.role == "werewolf":
                score += metrics.kill_success_rate * 15
            
            metrics.overall_score = round(score, 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "timestamp": self.timestamp,
            "config_name": self.config_name,
            "total_rounds": self.total_rounds,
            "winner": self.winner,
            "game_duration_seconds": self.game_duration_seconds,
            "global_metrics": {
                "total_dialogues": self.total_dialogues,
                "vote_tie_count": self.vote_tie_count
            },
            "player_metrics": {
                str(pid): m.to_dict() 
                for pid, m in self.player_metrics.items()
            }
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.evaluations: List[GameEvaluation] = []
    
    def collect_from_game(self, game_state: Any) -> GameEvaluation:
        """从游戏状态收集评测数据"""
        try:
            from engine.types import Team, RoleType
        except ImportError:
            from ..engine.types import Team, RoleType
        
        eval_result = GameEvaluation(
            game_id=game_state.game_id,
            total_rounds=game_state.day_number,
            winner=game_state.winner.value if game_state.winner else "unknown",
            total_dialogues=len(game_state.dialogues)
        )
        
        # 计算MVP（得分最高的玩家）
        max_score = -1
        mvp_id = None
        
        for player in game_state.players:
            metrics = PlayerMetrics(
                player_id=player.player_id,
                role=player.role.value,
                team=player.team.value
            )
            
            # 结果指标
            metrics.is_winner = (player.team == game_state.winner)
            metrics.is_alive = player.is_alive
            metrics.survival_rounds = game_state.day_number
            
            # 从对话中统计投票
            for dialogue in game_state.dialogues:
                if dialogue.get("phase") == "vote":
                    votes = dialogue.get("votes", {})
                    if str(player.player_id) in votes or player.player_id in votes:
                        metrics.total_votes_cast += 1
                        # 简化：假设投给狼人是正确投票
                        target = votes.get(str(player.player_id)) or votes.get(player.player_id)
                        if target is not None:
                            target_player = game_state.get_player_by_id(target)
                            if target_player and target_player.team != player.team:
                                metrics.correct_votes += 1
            
            # 统计发言
            for dialogue in game_state.dialogues:
                if dialogue.get("phase") == "speech":
                    if dialogue.get("speaker_id") == player.player_id:
                        metrics.total_speeches += 1
            
            # 发言信息量（简化：根据发言长度估算）
            if metrics.total_speeches > 0:
                total_content_length = sum(
                    len(d.get("content", "")) 
                    for d in game_state.dialogues 
                    if d.get("speaker_id") == player.player_id
                )
                metrics.speech_informativeness = min(total_content_length / 100, 1.0)
            
            # 角色特定指标
            if player.role == RoleType.SEER:
                # 查验准确率
                inspections = []
                for dialogue in game_state.dialogues:
                    if "查验" in dialogue.get("content", ""):
                        inspections.append(dialogue)
                if inspections:
                    # 简化：假设所有查验都是准确的
                    metrics.inspection_accuracy = 1.0
            
            elif player.role == RoleType.WEREWOLF:
                # 刀人成功率
                kills = 0
                successful_kills = 0
                for dialogue in game_state.dialogues:
                    if dialogue.get("phase") == "night_result":
                        deaths = dialogue.get("deaths", [])
                        kills += len(deaths)
                        successful_kills += len([d for d in deaths if game_state.get_player_by_id(d) and not game_state.get_player_by_id(d).is_alive])
                if kills > 0:
                    metrics.kill_success_rate = successful_kills / kills
            
            eval_result.player_metrics[player.player_id] = metrics
            
            # 计算临时分数用于MVP判断
            temp_score = metrics.is_winner * 50 + metrics.is_alive * 20 + metrics.correct_votes * 10
            if temp_score > max_score:
                max_score = temp_score
                mvp_id = player.player_id
        
        # 设置MVP
        if mvp_id is not None and mvp_id in eval_result.player_metrics:
            eval_result.player_metrics[mvp_id].is_mvp = True
        
        # 计算综合评分
        eval_result.calculate_overall_scores()
        
        self.evaluations.append(eval_result)
        return eval_result
    
    def get_leaderboard(self, role: Optional[str] = None, 
                       min_games: int = 1) -> List[Dict[str, Any]]:
        """生成排行榜"""
        player_stats: Dict[int, Dict[str, Any]] = {}
        
        for eval_result in self.evaluations:
            for player_id, metrics in eval_result.player_metrics.items():
                if role and metrics.role != role:
                    continue
                
                if player_id not in player_stats:
                    player_stats[player_id] = {
                        "player_id": player_id,
                        "role": metrics.role,
                        "games_played": 0,
                        "wins": 0,
                        "mvps": 0,
                        "total_score": 0.0,
                        "avg_survival_rounds": 0.0
                    }
                
                stats = player_stats[player_id]
                stats["games_played"] += 1
                if metrics.is_winner:
                    stats["wins"] += 1
                if metrics.is_mvp:
                    stats["mvps"] += 1
                stats["total_score"] += metrics.overall_score
                stats["avg_survival_rounds"] += metrics.survival_rounds
        
        # 计算平均值并过滤
        leaderboard = []
        for stats in player_stats.values():
            if stats["games_played"] >= min_games:
                stats["win_rate"] = round(stats["wins"] / stats["games_played"] * 100, 2)
                stats["mvp_rate"] = round(stats["mvps"] / stats["games_played"] * 100, 2)
                stats["avg_score"] = round(stats["total_score"] / stats["games_played"], 2)
                stats["avg_survival_rounds"] = round(stats["avg_survival_rounds"] / stats["games_played"], 2)
                leaderboard.append(stats)
        
        # 按平均分数排序
        leaderboard.sort(key=lambda x: x["avg_score"], reverse=True)
        
        # 添加排名
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
        
        return leaderboard
    
    def export_to_json(self, filepath: str):
        """导出所有评测数据到JSON文件"""
        data = {
            "total_games": len(self.evaluations),
            "evaluations": [e.to_dict() for e in self.evaluations]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def generate_report(self, game_id: Optional[str] = None) -> str:
        """生成复盘报告"""
        if game_id:
            eval_result = next((e for e in self.evaluations if e.game_id == game_id), None)
            if not eval_result:
                return f"Game {game_id} not found"
            
            report = "=== 游戏复盘报告 ===\n"
            report += f"游戏ID: {eval_result.game_id}\n"
            report += f"时间: {eval_result.timestamp}\n"
            report += f"配置: {eval_result.config_name}\n"
            report += f"总回合数: {eval_result.total_rounds}\n"
            report += f"获胜方: {eval_result.winner}\n\n"
            report += "=== 玩家表现 ===\n"
            
            for player_id, metrics in sorted(eval_result.player_metrics.items()):
                report += f"\n玩家 {player_id} ({metrics.role}):\n"
                report += f"  - 综合评分: {metrics.overall_score}\n"
                report += f"  - 是否获胜: {'是' if metrics.is_winner else '否'}\n"
                report += f"  - 是否存活: {'是' if metrics.is_alive else '否'}\n"
                report += f"  - 是否MVP: {'是' if metrics.is_mvp else '否'}\n"
                report += f"  - 投票准确率: {metrics.correct_votes}/{metrics.total_votes_cast}\n"
                report += f"  - 发言质量: {round(metrics.speech_informativeness, 2)}\n"
            
            return report
        else:
            # 生成总体统计报告
            leaderboard = self.get_leaderboard()
            report = "=== 总体排行榜 ===\n\n"
            for entry in leaderboard[:10]:  # 前10名
                report += f"\n排名 {entry['rank']}: 玩家 {entry['player_id']} ({entry['role']})\n"
                report += f"  - 平均分数: {entry['avg_score']}\n"
                report += f"  - 胜率: {entry['win_rate']}%\n"
                report += f"  - MVP率: {entry['mvp_rate']}%\n"
                report += f"  - 参与场次: {entry['games_played']}\n"
            return report
