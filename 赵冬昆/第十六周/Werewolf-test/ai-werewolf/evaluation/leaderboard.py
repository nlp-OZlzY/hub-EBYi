"""
Leaderboard 系统 - 跨模型竞技天梯
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

@dataclass
class AgentVersion:
    """Agent 版本信息"""
    agent_id: str
    version: str
    model_name: str  # 使用的LLM模型
    prompt_version: str
    strategy_version: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "version": self.version,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "strategy_version": self.strategy_version,
            "created_at": self.created_at
        }

@dataclass
class LeaderboardEntry:
    """排行榜条目"""
    rank: int
    agent_id: str
    agent_version: str
    model_name: str
    
    # 统计指标
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    mvps: int = 0
    
    # 评分指标
    avg_score: float = 0.0
    win_rate: float = 0.0
    mvp_rate: float = 0.0
    avg_survival_rounds: float = 0.0
    
    # 角色特定表现
    role_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 趋势
    recent_form: List[float] = field(default_factory=list)  # 最近10场得分
    trend: str = "stable"  # up, down, stable
    
    def calculate_rates(self):
        """计算比率"""
        if self.games_played > 0:
            self.win_rate = round(self.wins / self.games_played * 100, 2)
            self.mvp_rate = round(self.mvps / self.games_played * 100, 2)
    
    def update_trend(self):
        """更新趋势"""
        if len(self.recent_form) >= 5:
            recent_avg = sum(self.recent_form[-5:]) / 5
            older_avg = sum(self.recent_form[:-5]) / max(len(self.recent_form[:-5]), 1)
            
            if recent_avg > older_avg * 1.1:
                self.trend = "up"
            elif recent_avg < older_avg * 0.9:
                self.trend = "down"
            else:
                self.trend = "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "agent_id": self.agent_id,
            "version": self.agent_version,
            "model_name": self.model_name,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "mvps": self.mvps,
            "win_rate": self.win_rate,
            "mvp_rate": self.mvp_rate,
            "avg_score": self.avg_score,
            "avg_survival_rounds": self.avg_survival_rounds,
            "role_performance": self.role_performance,
            "trend": self.trend,
            "recent_form": self.recent_form[-5:]  # 只显示最近5场
        }

class LeaderboardManager:
    """排行榜管理器"""
    
    def __init__(self, storage_path: str = "./leaderboard_data.json"):
        self.storage_path = storage_path
        self.entries: Dict[str, LeaderboardEntry] = {}  # agent_id -> entry
        self.agent_versions: Dict[str, List[AgentVersion]] = {}  # agent_id -> versions
        self.load_data()
    
    def register_agent(self, agent_id: str, version: str, model_name: str,
                      prompt_version: str = "1.0", strategy_version: str = "1.0"):
        """注册新Agent版本"""
        if agent_id not in self.agent_versions:
            self.agent_versions[agent_id] = []
        
        agent_version = AgentVersion(
            agent_id=agent_id,
            version=version,
            model_name=model_name,
            prompt_version=prompt_version,
            strategy_version=strategy_version
        )
        
        self.agent_versions[agent_id].append(agent_version)
        
        # 创建排行榜条目
        if agent_id not in self.entries:
            self.entries[agent_id] = LeaderboardEntry(
                rank=0,
                agent_id=agent_id,
                agent_version=version,
                model_name=model_name
            )
        else:
            # 更新版本信息
            self.entries[agent_id].agent_version = version
            self.entries[agent_id].model_name = model_name
        
        self.save_data()
    
    def record_game_result(self, agent_id: str, game_result: Dict[str, Any]):
        """记录游戏结果"""
        if agent_id not in self.entries:
            print(f"Warning: Agent {agent_id} not registered")
            return
        
        entry = self.entries[agent_id]
        
        # 更新基本统计
        entry.games_played += 1
        if game_result.get("is_winner"):
            entry.wins += 1
        else:
            entry.losses += 1
        
        if game_result.get("is_mvp"):
            entry.mvps += 1
        
        # 更新平均分
        score = game_result.get("score", 0)
        entry.avg_score = (entry.avg_score * (entry.games_played - 1) + score) / entry.games_played
        entry.avg_score = round(entry.avg_score, 2)
        
        # 更新生存回合
        survival_rounds = game_result.get("survival_rounds", 0)
        entry.avg_survival_rounds = (entry.avg_survival_rounds * (entry.games_played - 1) + survival_rounds) / entry.games_played
        entry.avg_survival_rounds = round(entry.avg_survival_rounds, 2)
        
        # 更新角色表现
        role = game_result.get("role", "unknown")
        if role not in entry.role_performance:
            entry.role_performance[role] = {
                "games": 0,
                "wins": 0,
                "avg_score": 0.0
            }
        
        role_stats = entry.role_performance[role]
        role_stats["games"] += 1
        if game_result.get("is_winner"):
            role_stats["wins"] += 1
        role_stats["avg_score"] = (role_stats["avg_score"] * (role_stats["games"] - 1) + score) / role_stats["games"]
        
        # 更新最近表现
        entry.recent_form.append(score)
        if len(entry.recent_form) > 20:  # 保留最近20场
            entry.recent_form = entry.recent_form[-20:]
        
        # 计算比率和趋势
        entry.calculate_rates()
        entry.update_trend()
        
        # 重新排序
        self._recalculate_ranks()
        
        self.save_data()
    
    def _recalculate_ranks(self):
        """重新计算排名"""
        # 按平均分数排序
        sorted_entries = sorted(
            self.entries.values(),
            key=lambda x: (x.avg_score, x.win_rate, x.mvp_rate),
            reverse=True
        )
        
        for i, entry in enumerate(sorted_entries, 1):
            entry.rank = i
    
    def get_leaderboard(self, role: Optional[str] = None, 
                       min_games: int = 5,
                       model_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取排行榜"""
        filtered_entries = []
        
        for entry in self.entries.values():
            if entry.games_played < min_games:
                continue
            
            if model_filter and entry.model_name != model_filter:
                continue
            
            if role and role not in entry.role_performance:
                continue
            
            filtered_entries.append(entry.to_dict())
        
        # 按排名排序
        filtered_entries.sort(key=lambda x: x["rank"])
        
        return filtered_entries
    
    def get_agent_stats(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取特定Agent的统计"""
        if agent_id not in self.entries:
            return None
        
        entry = self.entries[agent_id]
        versions = self.agent_versions.get(agent_id, [])
        
        return {
            "agent_id": agent_id,
            "current_version": entry.agent_version,
            "model_name": entry.model_name,
            "stats": entry.to_dict(),
            "version_history": [v.to_dict() for v in versions],
            "role_breakdown": entry.role_performance
        }
    
    def compare_agents(self, agent_ids: List[str]) -> Dict[str, Any]:
        """对比多个Agent"""
        comparison = {
            "agents": [],
            "head_to_head": {},
            "summary": {}
        }
        
        for agent_id in agent_ids:
            if agent_id in self.entries:
                comparison["agents"].append(self.entries[agent_id].to_dict())
        
        # 计算统计摘要
        if comparison["agents"]:
            avg_scores = [a["avg_score"] for a in comparison["agents"]]
            win_rates = [a["win_rate"] for a in comparison["agents"]]
            
            comparison["summary"] = {
                "best_avg_score": max(avg_scores),
                "best_win_rate": max(win_rates),
                "avg_games_played": sum(a["games_played"] for a in comparison["agents"]) / len(comparison["agents"]),
                "total_games": sum(a["games_played"] for a in comparison["agents"])
            }
        
        return comparison
    
    def get_top_performers(self, category: str = "avg_score", 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """获取顶尖表现者"""
        valid_entries = [e for e in self.entries.values() if e.games_played >= 5]
        
        if category == "avg_score":
            sorted_entries = sorted(valid_entries, key=lambda x: x.avg_score, reverse=True)
        elif category == "win_rate":
            sorted_entries = sorted(valid_entries, key=lambda x: x.win_rate, reverse=True)
        elif category == "mvp_rate":
            sorted_entries = sorted(valid_entries, key=lambda x: x.mvp_rate, reverse=True)
        else:
            sorted_entries = sorted(valid_entries, key=lambda x: x.avg_score, reverse=True)
        
        return [e.to_dict() for e in sorted_entries[:limit]]
    
    def save_data(self):
        """保存数据到文件"""
        data = {
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
            "agent_versions": {k: [v.to_dict() for v in vs] for k, vs in self.agent_versions.items()},
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_data(self):
        """从文件加载数据"""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复entries
            for agent_id, entry_data in data.get("entries", {}).items():
                entry = LeaderboardEntry(
                    rank=entry_data.get("rank", 0),
                    agent_id=entry_data["agent_id"],
                    agent_version=entry_data.get("version", "1.0"),
                    model_name=entry_data.get("model_name", "unknown")
                )
                entry.games_played = entry_data.get("games_played", 0)
                entry.wins = entry_data.get("wins", 0)
                entry.losses = entry_data.get("losses", 0)
                entry.mvps = entry_data.get("mvps", 0)
                entry.avg_score = entry_data.get("avg_score", 0.0)
                entry.win_rate = entry_data.get("win_rate", 0.0)
                entry.mvp_rate = entry_data.get("mvp_rate", 0.0)
                entry.avg_survival_rounds = entry_data.get("avg_survival_rounds", 0.0)
                entry.role_performance = entry_data.get("role_performance", {})
                entry.trend = entry_data.get("trend", "stable")
                entry.recent_form = entry_data.get("recent_form", [])
                
                self.entries[agent_id] = entry
            
            # 恢复agent_versions
            for agent_id, versions_data in data.get("agent_versions", {}).items():
                self.agent_versions[agent_id] = [
                    AgentVersion(**v) for v in versions_data
                ]
                
        except Exception as e:
            print(f"Error loading leaderboard data: {e}")
    
    def export_report(self, filepath: str):
        """导出完整报告"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_agents": len(self.entries),
            "total_games": sum(e.games_played for e in self.entries.values()),
            "leaderboard": self.get_leaderboard(),
            "top_by_role": {}
        }
        
        # 按角色统计
        roles = ["werewolf", "seer", "witch", "hunter", "villager"]
        for role in roles:
            role_leaderboard = self.get_leaderboard(role=role, min_games=1)
            report["top_by_role"][role] = role_leaderboard[:5]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Leaderboard report exported to {filepath}")
