"""信息处理器 - 统一处理游戏中的公开信息"""
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, Counter
from core.role_system import Camp


@dataclass
class TrustScore:
    """信任分数"""
    player_id: str
    trust_score: float = 0.0    # -1 到 1，负数表示可疑
    vote_against_count: int = 0  # 被投票次数
    suspicious_keywords: list[str] = field(default_factory=list)
    defense_keywords: list[str] = field(default_factory=list)


class InformationProcessor:
    """信息处理器

    统一处理和分析游戏中的公开信息，为各 Agent 提供决策支持
    """

    # 可疑关键词
    SUSPICIOUS_KEYWORDS = [
        "可疑", "不像", "紧张", "躲闪", "心虚", "可疑",
        "逻辑不对", "矛盾", "回避", "带节奏", "跟风",
        "狼", "骗", "演"
    ]

    # 好人倾向关键词
    GOOD_KEYWORDS = [
        "好人", "村民", "配合", "逻辑", "分析", "推理",
        "谨慎", "观察", "稳", "合理", "清晰"
    ]

    def __init__(self):
        self.trust_scores: dict[str, TrustScore] = {}
        self.vote_history: list[dict] = []  # 投票历史
        self.speech_history: list[dict] = []  # 发言历史
        self.death_history: list[dict] = []   # 死亡历史

    def process_speech(self, player_id: str, speech: str):
        """处理发言，分析信任度"""
        if player_id not in self.trust_scores:
            self.trust_scores[player_id] = TrustScore(player_id=player_id)

        score = self.trust_scores[player_id]
        speech_lower = speech.lower()

        # 检测可疑关键词
        for kw in self.SUSPICIOUS_KEYWORDS:
            if kw in speech_lower:
                score.suspicious_keywords.append(kw)
                score.trust_score -= 0.1

        # 检测好人倾向关键词
        for kw in self.GOOD_KEYWORDS:
            if kw in speech_lower:
                score.defense_keywords.append(kw)
                score.trust_score += 0.05

        # 发言长度过短可能是划水
        if len(speech) < 20:
            score.trust_score -= 0.05

        # 限制范围
        score.trust_score = max(-1.0, min(1.0, score.trust_score))

        self.speech_history.append({
            "player_id": player_id,
            "speech": speech,
            "timestamp": len(self.speech_history)
        })

    def process_vote(self, voter_id: str, target_id: str):
        """处理投票"""
        self.vote_history.append({
            "voter": voter_id,
            "target": target_id
        })

        # 目标被投票次数增加
        if target_id not in self.trust_scores:
            self.trust_scores[target_id] = TrustScore(player_id=target_id)
        self.trust_scores[target_id].vote_against_count += 1
        self.trust_scores[target_id].trust_score -= 0.1

    def process_death(self, player_id: str, reason: str):
        """处理死亡"""
        self.death_history.append({
            "player_id": player_id,
            "reason": reason
        })

    def get_trust_score(self, player_id: str) -> float:
        """获取玩家信任分数"""
        if player_id not in self.trust_scores:
            return 0.0
        return self.trust_scores[player_id].trust_score

    def get_vote_target_count(self, player_id: str) -> int:
        """获取被投票次数"""
        if player_id not in self.trust_scores:
            return 0
        return self.trust_scores[player_id].vote_against_count

    def get_vote_pattern(self, player_id: str) -> list[str]:
        """获取某玩家投票过的目标"""
        return [v["target"] for v in self.vote_history if v["voter"] == player_id]

    def get_most_suspicious(self, exclude: list[str] = None) -> Optional[str]:
        """获取最可疑玩家"""
        exclude = exclude or []
        candidates = [
            (pid, score.trust_score)
            for pid, score in self.trust_scores.items()
            if pid not in exclude and score.trust_score < 0
        ]

        if not candidates:
            # 如果没有可疑的，返回被投票最多的
            candidates = [
                (pid, -score.vote_against_count)
                for pid, score in self.trust_scores.items()
                if pid not in exclude
            ]

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]

    def get_most_trusted(self, exclude: list[str] = None) -> Optional[str]:
        """获取最可信玩家"""
        exclude = exclude or []
        candidates = [
            (pid, score.trust_score)
            for pid, score in self.trust_scores.items()
            if pid not in exclude and score.trust_score > 0
        ]

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def get_vote_target_recommendation(
        self,
        agent_id: str,
        alive_players: dict,
        known_wolves: list[str] = None,
        known_good: list[str] = None
    ) -> Optional[str]:
        """获取投票目标推荐

        综合信任分数和已知的狼人/好人信息
        """
        known_wolves = known_wolves or []
        known_good = known_good or []
        exclude = known_wolves + [agent_id]

        # 优先投已知狼人
        for wolf in known_wolves:
            if wolf in alive_players and wolf not in exclude:
                return wolf

        # 其次投最可疑的
        suspicious = self.get_most_suspicious(exclude=exclude)
        if suspicious:
            return suspicious

        # 投被投票最多但不是好人的
        candidates = [
            pid for pid in alive_players.keys()
            if pid not in exclude and pid not in known_good
        ]

        if not candidates:
            return None

        # 按投票次数排序
        candidates.sort(
            key=lambda x: self.get_vote_target_count(x),
            reverse=True
        )
        return candidates[0]

    def get_camp_analysis(self, players: dict) -> dict:
        """获取阵营分析"""
        return {
            "total_players": len(players),
            "alive_wolves": len([p for p in players.values()
                                if p.role.camp == Camp.WOLF and p.alive]),
            "alive_good": len([p for p in players.values()
                              if p.role.camp == Camp.GOOD and p.alive])
        }

    def summarize(self) -> dict:
        """生成信息摘要"""
        return {
            "total_speeches": len(self.speech_history),
            "total_votes": len(self.vote_history),
            "total_deaths": len(self.death_history),
            "trust_scores": {
                pid: {"score": s.trust_score, "votes_against": s.vote_against_count}
                for pid, s in self.trust_scores.items()
            }
        }