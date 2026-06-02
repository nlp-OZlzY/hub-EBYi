"""游戏指标收集器

从 GameEngine 的游戏结果中提取量化指标，供 SelfReflector 反思使用。
包含胜负判定、投票正确率、击杀效率、查验准确率等维度。
"""

from typing import Dict, Any, List


class MetricsCollector:
    """游戏指标收集器

    从已结束的 GameEngine 中提取各角色的量化表现指标，供 SelfReflector 使用。
    按角色类型计算不同维度的指标：
    - 通用：胜负、存活天数、是否被投票出局
    - 好人：投票正确率（是否投中狼人）
    - 狼人：击杀效率（是否刀中神职）
    - 预言家：查验正确率
    - 女巫：救人/毒人是否成功
    - 猎人：开枪是否带走狼人
    """

    @staticmethod
    def collect(engine) -> Dict[str, Dict]:
        """从游戏引擎收集所有角色的指标

        Args:
            engine: GameEngine 实例（游戏已结束）

        Returns:
            {role_type: metrics_dict} 字典
        """
        winner = engine.game_state.get_winner()
        all_players_data = [p.to_dict() for p in engine.game_state.players]
        dialogues = engine.game_state.dialogues
        death_records = engine.death_records

        results = {}
        for player in engine.game_state.players:
            role_type = player.role.role_type.value
            camp = player.role.camp.value
            player_id = player.player_id

            # 计算存活天数
            survival_days = engine.game_state.day_number
            was_voted_out = False
            for death in death_records:
                if death["player_id"] == player_id:
                    survival_days = death.get("day", survival_days)
                    if death.get("cause") == "vote":
                        was_voted_out = True
                    break

            metrics = MetricsCollector._compute_role_metrics(
                role=role_type,
                camp=camp,
                winner=winner,
                player_id=player_id,
                survival_days=survival_days,
                was_voted_out=was_voted_out,
                dialogues=dialogues,
                death_records=death_records,
                all_players=all_players_data,
            )

            results[role_type] = {
                "role": role_type,
                "player_id": player_id,
                "metrics": metrics,
                "highlights": MetricsCollector._extract_highlights(
                    player_id=player_id,
                    dialogues=dialogues,
                    death_records=death_records,
                    survival_days=survival_days,
                    was_voted_out=was_voted_out,
                ),
            }

        return results

    @staticmethod
    def _compute_role_metrics(
        role: str,
        camp: str,
        winner: str,
        player_id: int,
        survival_days: int,
        was_voted_out: bool,
        dialogues: List[Dict],
        death_records: List[Dict],
        all_players: List[Dict],
    ) -> Dict[str, Any]:
        """计算单个角色的指标"""
        win = (camp == winner) if winner else False

        # 构建 player_id -> role 映射
        player_roles = {p["player_id"]: p.get("role", "") for p in all_players}
        player_camps = {p["player_id"]: p.get("camp", "") for p in all_players}

        # 该玩家的对话
        player_dialogues = [d for d in dialogues if d.get("player_id") == player_id]

        # 投票正确率（好人阵营：投中狼人）
        vote_accuracy = 0.0
        if camp == "good":
            votes = [d for d in player_dialogues if d.get("action") == "vote"]
            if votes:
                correct = sum(1 for v in votes if player_camps.get(v.get("target")) == "evil")
                vote_accuracy = correct / len(votes)

        # 击杀效率（狼人：刀中神职）
        kill_efficiency = 0.0
        if role == "werewolf":
            kills = [d for d in player_dialogues if d.get("action") == "wolf_vote"]
            if kills:
                god_roles = {"seer", "witch", "hunter"}
                correct = sum(1 for k in kills if player_roles.get(k.get("target")) in god_roles)
                kill_efficiency = correct / len(kills)

        # 查验正确率（预言家）
        seer_accuracy = 0.0
        if role == "seer":
            checks = [d for d in player_dialogues if d.get("action") == "seer_check"]
            if checks:
                correct = sum(
                    1 for c in checks
                    if c.get("result") == "wolf" and player_camps.get(c.get("target")) == "evil"
                )
                seer_accuracy = correct / len(checks)

        # 女巫指标
        heal_success = False
        poison_success = False
        if role == "witch":
            heals = [d for d in player_dialogues if d.get("action") == "heal"]
            if heals:
                heal_success = True  # 简化：使用了解药即视为成功
            poisons = [d for d in player_dialogues if d.get("action") == "poison"]
            if poisons:
                poison_success = any(player_camps.get(p.get("target")) == "evil" for p in poisons)

        # 猎人指标
        shot_accuracy = False
        if role == "hunter":
            shots = [d for d in player_dialogues if d.get("action") == "hunter_shot"]
            if shots:
                shot_accuracy = any(player_camps.get(s.get("target")) == "evil" for s in shots)

        return {
            "win": win,
            "survival_days": survival_days,
            "vote_accuracy": vote_accuracy,
            "kill_efficiency": kill_efficiency,
            "seer_accuracy": seer_accuracy,
            "heal_success": heal_success,
            "poison_success": poison_success,
            "shot_accuracy": shot_accuracy,
            "was_voted_out": was_voted_out,
        }

    @staticmethod
    def _extract_highlights(
        player_id: int,
        dialogues: List[Dict],
        death_records: List[Dict],
        survival_days: int,
        was_voted_out: bool,
    ) -> List[str]:
        """提取关键事件摘要"""
        highlights = []
        player_dialogues = [d for d in dialogues if d.get("player_id") == player_id]

        for d in player_dialogues:
            action = d.get("action")
            day = d.get("day", "?")
            if action == "speech":
                content = d.get("content", "")
                if len(content) > 50:
                    highlights.append(f"第{day}天发言：{content[:50]}...")
            elif action == "wolf_vote":
                highlights.append(f"第{day}天击杀玩家{d.get('target')}")
            elif action == "seer_check":
                result = "狼人" if d.get("result") == "wolf" else "好人"
                highlights.append(f"第{day}天查验玩家{d.get('target')}：{result}")
            elif action == "vote":
                highlights.append(f"第{day}天投票给玩家{d.get('target')}")

        if was_voted_out:
            highlights.append(f"在第{survival_days}天被投票出局")

        return highlights
