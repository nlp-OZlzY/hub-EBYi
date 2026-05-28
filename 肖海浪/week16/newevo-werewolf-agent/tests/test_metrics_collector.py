import pytest
from metrics.collector import MetricsCollector


def test_metrics_win_detection():
    """测试胜利检测"""
    # 好人阵营胜利
    metrics = MetricsCollector._compute_role_metrics(
        role="seer",
        camp="good",
        winner="good",
        player_id=2,
        survival_days=3,
        was_voted_out=False,
        dialogues=[],
        death_records=[],
        all_players=[],
    )
    assert metrics["win"] is True

    # 好人阵营失败
    metrics = MetricsCollector._compute_role_metrics(
        role="seer",
        camp="good",
        winner="evil",
        player_id=2,
        survival_days=3,
        was_voted_out=False,
        dialogues=[],
        death_records=[],
        all_players=[],
    )
    assert metrics["win"] is False


def test_metrics_kill_efficiency():
    """测试击杀效率计算"""
    all_players = [
        {"player_id": 0, "role": "werewolf", "camp": "evil"},
        {"player_id": 2, "role": "seer", "camp": "good"},
        {"player_id": 4, "role": "hunter", "camp": "good"},
        {"player_id": 5, "role": "villager", "camp": "good"},
    ]
    dialogues = [
        {"action": "wolf_vote", "player_id": 0, "target": 2, "day": 1},   # 杀神职
        {"action": "wolf_vote", "player_id": 0, "target": 5, "day": 2},   # 杀村民
    ]
    metrics = MetricsCollector._compute_role_metrics(
        role="werewolf",
        camp="evil",
        winner="evil",
        player_id=0,
        survival_days=3,
        was_voted_out=False,
        dialogues=dialogues,
        death_records=[],
        all_players=all_players,
    )
    assert metrics["kill_efficiency"] == 0.5  # 1/2 杀中神职


def test_metrics_vote_accuracy():
    """测试投票正确率计算"""
    all_players = [
        {"player_id": 0, "role": "werewolf", "camp": "evil"},
        {"player_id": 2, "role": "seer", "camp": "good"},
        {"player_id": 3, "role": "villager", "camp": "good"},
    ]
    dialogues = [
        {"action": "vote", "player_id": 2, "target": 0, "day": 1},   # 投中狼人
        {"action": "vote", "player_id": 2, "target": 3, "day": 2},   # 投错
    ]
    metrics = MetricsCollector._compute_role_metrics(
        role="seer",
        camp="good",
        winner="good",
        player_id=2,
        survival_days=3,
        was_voted_out=False,
        dialogues=dialogues,
        death_records=[],
        all_players=all_players,
    )
    assert metrics["vote_accuracy"] == 0.5  # 1/2 投中狼人


def test_metrics_highlights_extraction():
    """测试关键事件提取"""
    dialogues = [
        {"action": "speech", "player_id": 0, "content": "我是预言家，昨晚查验了3号是狼人", "day": 1},
        {"action": "vote", "player_id": 0, "target": 3, "day": 1},
        {"action": "wolf_vote", "player_id": 0, "target": 2, "day": 2},
    ]
    highlights = MetricsCollector._extract_highlights(
        player_id=0,
        dialogues=dialogues,
        death_records=[],
        survival_days=2,
        was_voted_out=False,
    )
    assert len(highlights) > 0


def test_metrics_empty_game():
    """测试空游戏不产生无效指标"""
    all_players = [
        {"player_id": 0, "role": "werewolf", "camp": "evil"},
    ]
    dialogues = []
    metrics = MetricsCollector._compute_role_metrics(
        role="werewolf",
        camp="evil",
        winner=None,
        player_id=0,
        survival_days=0,
        was_voted_out=False,
        dialogues=dialogues,
        death_records=[],
        all_players=all_players,
    )
    assert metrics["win"] is False
    assert metrics["kill_efficiency"] == 0.0
