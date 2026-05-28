"""
通用 Agent 自演化系统
实现"读懂自己→修改自己→运行自己"的循环
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import re

@dataclass
class SelfCritique:
    """自我批评报告"""
    agent_id: str
    version: str
    game_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 表现分析
    overall_score: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    # 具体失误
    mistakes: List[Dict[str, Any]] = field(default_factory=list)
    
    # 改进建议
    improvement_suggestions: List[str] = field(default_factory=list)
    
    # 策略调整
    strategy_adjustments: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "version": self.version,
            "game_id": self.game_id,
            "timestamp": self.timestamp,
            "overall_score": self.overall_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "mistakes": self.mistakes,
            "improvement_suggestions": self.improvement_suggestions,
            "strategy_adjustments": self.strategy_adjustments
        }

@dataclass
class EvolutionRecord:
    """演化记录"""
    agent_id: str
    from_version: str
    to_version: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 演化类型
    evolution_type: str = ""  # prompt_update, strategy_update, config_update
    
    # 变更内容
    changes: Dict[str, Any] = field(default_factory=dict)
    
    # 触发原因
    trigger_reason: str = ""
    
    # 效果评估（后续填充）
    performance_delta: float = 0.0
    is_successful: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "timestamp": self.timestamp,
            "evolution_type": self.evolution_type,
            "changes": self.changes,
            "trigger_reason": self.trigger_reason,
            "performance_delta": self.performance_delta,
            "is_successful": self.is_successful
        }

class SelfEvolutionEngine:
    """自演化引擎"""
    
    def __init__(self, agent_id: str, storage_path: str = "./evolution_data"):
        self.agent_id = agent_id
        self.storage_path = storage_path
        self.current_version = "1.0.0"
        
        self.critiques: List[SelfCritique] = []
        self.evolution_history: List[EvolutionRecord] = []
        
        # 当前配置
        self.current_prompt: str = ""
        self.current_strategy_config: Dict[str, Any] = {}
        
        os.makedirs(storage_path, exist_ok=True)
        self.load_data()
    
    def generate_self_critique(self, game_state: Any, 
                               metrics: Dict[str, Any]) -> SelfCritique:
        """生成自我批评报告"""
        critique = SelfCritique(
            agent_id=self.agent_id,
            version=self.current_version,
            game_id=game_state.game_id,
            overall_score=metrics.get("overall_score", 0.0)
        )
        
        # 分析优势
        if metrics.get("is_winner"):
            critique.strengths.append("成功带领阵营获胜")
        
        if metrics.get("is_mvp"):
            critique.strengths.append("获得MVP，表现突出")
        
        vote_accuracy = metrics.get("correct_votes", 0) / max(metrics.get("total_votes", 1), 1)
        if vote_accuracy > 0.7:
            critique.strengths.append(f"投票准确率高 ({vote_accuracy:.0%})")
        
        # 分析劣势
        if not metrics.get("is_winner"):
            critique.weaknesses.append("未能带领阵营获胜")
        
        if vote_accuracy < 0.5:
            critique.weaknesses.append(f"投票准确率较低 ({vote_accuracy:.0%})")
            critique.improvement_suggestions.append("需要提高判断力，更准确地识别狼人")
        
        if not metrics.get("is_alive"):
            critique.weaknesses.append("过早死亡")
            critique.improvement_suggestions.append("需要更好地保护自己，避免成为目标")
        
        # 角色特定分析
        role = metrics.get("role", "")
        if role == "seer":
            if metrics.get("inspection_accuracy", 1.0) < 1.0:
                critique.mistakes.append({
                    "type": "inspection_error",
                    "description": "查验结果有误",
                    "severity": "high"
                })
                critique.improvement_suggestions.append("仔细分析查验结果，避免被误导")
        
        elif role == "werewolf":
            if metrics.get("kill_success_rate", 1.0) < 0.5:
                critique.mistakes.append({
                    "type": "kill_failure",
                    "description": "刀人成功率低",
                    "severity": "medium"
                })
        
        # 生成策略调整建议
        critique.strategy_adjustments = self._suggest_strategy_adjustments(
            metrics, critique.weaknesses
        )
        
        self.critiques.append(critique)
        self.save_data()
        
        return critique
    
    def _suggest_strategy_adjustments(self, metrics: Dict[str, Any], 
                                     weaknesses: List[str]) -> Dict[str, Any]:
        """建议策略调整"""
        adjustments = {
            "risk_tolerance_delta": 0.0,
            "speech_style_changes": [],
            "voting_strategy_changes": [],
            "role_specific_changes": []
        }
        
        # 根据表现调整风险承受度
        if "过早死亡" in weaknesses:
            adjustments["risk_tolerance_delta"] = -0.1  # 更保守
        elif metrics.get("is_winner") and metrics.get("is_alive"):
            adjustments["risk_tolerance_delta"] = 0.05  # 稍微激进一点
        
        # 投票策略调整
        vote_accuracy = metrics.get("correct_votes", 0) / max(metrics.get("total_votes", 1), 1)
        if vote_accuracy < 0.5:
            adjustments["voting_strategy_changes"].append("更加谨慎地分析发言")
            adjustments["voting_strategy_changes"].append("多关注细节和逻辑漏洞")
        
        return adjustments
    
    def evolve(self, critique: SelfCritique) -> Optional[EvolutionRecord]:
        """根据自我批评进行演化"""
        # 决定是否演化
        if critique.overall_score > 80 and len(critique.weaknesses) < 2:
            print(f"[Evolution] Agent {self.agent_id} 表现良好，跳过演化")
            return None
        
        # 确定演化类型
        evolution_type = self._determine_evolution_type(critique)
        
        # 生成新版本号
        new_version = self._increment_version(self.current_version)
        
        # 执行演化
        changes = {}
        
        if evolution_type == "prompt_update":
            changes = self._evolve_prompt(critique)
        elif evolution_type == "strategy_update":
            changes = self._evolve_strategy(critique)
        elif evolution_type == "config_update":
            changes = self._evolve_config(critique)
        
        # 记录演化
        record = EvolutionRecord(
            agent_id=self.agent_id,
            from_version=self.current_version,
            to_version=new_version,
            evolution_type=evolution_type,
            changes=changes,
            trigger_reason=f"Score: {critique.overall_score}, Weaknesses: {len(critique.weaknesses)}"
        )
        
        self.evolution_history.append(record)
        self.current_version = new_version
        
        self.save_data()
        
        print(f"[Evolution] Agent {self.agent_id} evolved from {record.from_version} to {record.to_version}")
        print(f"[Evolution] Type: {evolution_type}, Changes: {list(changes.keys())}")
        
        return record
    
    def _determine_evolution_type(self, critique: SelfCritique) -> str:
        """确定演化类型"""
        # 简单策略：根据弱点类型选择演化方向
        has_speech_issues = any("发言" in w or "沟通" in w for w in critique.weaknesses)
        has_strategy_issues = any("策略" in w or "判断" in w for w in critique.weaknesses)
        
        if has_speech_issues:
            return "prompt_update"
        elif has_strategy_issues:
            return "strategy_update"
        else:
            return "config_update"
    
    def _increment_version(self, version: str) -> str:
        """递增版本号"""
        parts = version.split(".")
        if len(parts) == 3:
            major, minor, patch = parts
            new_patch = str(int(patch) + 1)
            return f"{major}.{minor}.{new_patch}"
        return version + ".1"
    
    def _evolve_prompt(self, critique: SelfCritique) -> Dict[str, Any]:
        """演化Prompt"""
        # 这里可以接入LLM来优化Prompt
        changes = {
            "modified_sections": [],
            "added_instructions": [],
            "removed_instructions": []
        }
        
        # 根据弱点添加特定指令
        for weakness in critique.weaknesses:
            if "投票" in weakness:
                changes["added_instructions"].append(
                    "在投票前，仔细分析每个玩家的发言逻辑，寻找矛盾点"
                )
            if "发言" in weakness:
                changes["added_instructions"].append(
                    "发言时提供更多有价值的信息，避免过于模糊"
                )
        
        return changes
    
    def _evolve_strategy(self, critique: SelfCritique) -> Dict[str, Any]:
        """演化策略"""
        changes = {
            "risk_tolerance_adjustment": critique.strategy_adjustments.get("risk_tolerance_delta", 0),
            "voting_strategy": critique.strategy_adjustments.get("voting_strategy_changes", []),
            "speech_style": critique.strategy_adjustments.get("speech_style_changes", [])
        }
        
        # 更新当前策略配置
        self.current_strategy_config["risk_tolerance"] = self.current_strategy_config.get(
            "risk_tolerance", 0.5
        ) + changes["risk_tolerance_adjustment"]
        
        return changes
    
    def _evolve_config(self, critique: SelfCritique) -> Dict[str, Any]:
        """演化配置参数"""
        changes = {
            "parameter_adjustments": {}
        }
        
        # 调整各种参数
        if "过早死亡" in critique.weaknesses:
            changes["parameter_adjustments"]["defensiveness"] = "increase"
        
        if "误判" in str(critique.weaknesses):
            changes["parameter_adjustments"]["analysis_depth"] = "increase"
        
        return changes
    
    def evaluate_evolution_success(self, evolution_id: int, 
                                   new_performance: float) -> bool:
        """评估演化是否成功"""
        if evolution_id >= len(self.evolution_history):
            return False
        
        record = self.evolution_history[evolution_id]
        
        # 查找演化前的表现
        pre_evolution_critiques = [
            c for c in self.critiques 
            if c.timestamp < record.timestamp
        ]
        
        if pre_evolution_critiques:
            pre_performance = pre_evolution_critiques[-1].overall_score
            delta = new_performance - pre_performance
            
            record.performance_delta = delta
            record.is_successful = delta > 0
            
            self.save_data()
            
            return record.is_successful
        
        return False
    
    def get_evolution_report(self) -> str:
        """获取演化报告"""
        report = f"""
=== Agent {self.agent_id} 自演化报告 ===
当前版本: {self.current_version}
演化次数: {len(self.evolution_history)}
自我批评次数: {len(self.critiques)}

【演化历史】
"""
        for i, record in enumerate(self.evolution_history[-5:], 1):  # 最近5次
            report += f"""
{i}. {record.from_version} → {record.to_version}
   类型: {record.evolution_type}
   原因: {record.trigger_reason}
   效果: {'+' if record.performance_delta > 0 else ''}{record.performance_delta:.1f}
   成功: {'是' if record.is_successful else '否'}
"""
        
        report += """
【最新自我批评】
"""
        if self.critiques:
            latest = self.critiques[-1]
            report += f"""
游戏: {latest.game_id}
得分: {latest.overall_score}
优势: {', '.join(latest.strengths) if latest.strengths else '无'}
劣势: {', '.join(latest.weaknesses) if latest.weaknesses else '无'}
建议: {', '.join(latest.improvement_suggestions[:3]) if latest.improvement_suggestions else '无'}
"""
        
        return report
    
    def save_data(self):
        """保存数据"""
        data = {
            "agent_id": self.agent_id,
            "current_version": self.current_version,
            "current_strategy_config": self.current_strategy_config,
            "critiques": [c.to_dict() for c in self.critiques],
            "evolution_history": [e.to_dict() for e in self.evolution_history]
        }
        
        filepath = os.path.join(self.storage_path, f"{self.agent_id}_evolution.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_data(self):
        """加载数据"""
        filepath = os.path.join(self.storage_path, f"{self.agent_id}_evolution.json")
        if not os.path.exists(filepath):
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_version = data.get("current_version", "1.0.0")
            self.current_strategy_config = data.get("current_strategy_config", {})
            
            # 恢复critiques
            for c_data in data.get("critiques", []):
                critique = SelfCritique(**c_data)
                self.critiques.append(critique)
            
            # 恢复evolution_history
            for e_data in data.get("evolution_history", []):
                record = EvolutionRecord(**e_data)
                self.evolution_history.append(record)
                
        except Exception as e:
            print(f"Error loading evolution data: {e}")
