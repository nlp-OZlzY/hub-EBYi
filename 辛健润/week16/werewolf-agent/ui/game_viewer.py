"""游戏可视化组件"""
import streamlit as st
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class GameViewerConfig:
    """观战配置"""
    show_speeches: bool = True
    show_votes: bool = True
    show_night_actions: bool = True
    max_speech_length: int = 200


def render_player_card(player_id: str, name: str, role: str, alive: bool, camp: str) -> dict:
    """渲染玩家卡片"""
    color = "green" if camp == "good" else "red" if camp == "wolf" else "gray"
    return {
        "player_id": player_id,
        "name": name,
        "role": role,
        "alive": alive,
        "camp": camp,
        "color": color
    }


def render_game_status(status: dict, config: GameViewerConfig = None):
    """渲染游戏状态"""
    if config is None:
        config = GameViewerConfig()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("天数", status.get("day", 1))

    with col2:
        wolf_count = status.get("wolf_count", 0)
        good_count = status.get("good_count", 0)
        st.metric("狼人", wolf_count)

    with col3:
        st.metric("好人", good_count)

    with col4:
        phase = status.get("phase", "night")
        phase_display = {"night": "夜晚", "day_speech": "发言", "day_vote": "投票"}
        st.metric("阶段", phase_display.get(phase, phase))


def render_player_list(players: list[dict]):
    """渲染玩家列表"""
    if not players:
        st.info("暂无玩家")
        return

    # 分成存活和死亡两组
    alive = [p for p in players if p.get("alive", True)]
    dead = [p for p in players if not p.get("alive", True)]

    with st.expander("存活玩家", expanded=True):
        cols = st.columns(3)
        for i, player in enumerate(alive):
            with cols[i % 3]:
                camp = player.get("camp", "")
                color = "🟢" if camp == "good" else "🔴" if camp == "wolf" else "⚪"
                st.write(f"{color} **{player['name']}** ({player.get('role', '未知')})")

    if dead:
        with st.expander(f"死亡玩家 ({len(dead)})"):
            for player in dead:
                st.write(f"❌ ~~{player['name']}~~ ({player.get('role', '未知')})")


def render_speech_history(speeches: list[dict], config: GameViewerConfig = None):
    """渲染发言历史"""
    if config is None:
        config = GameViewerConfig()

    if not speeches:
        st.info("暂无发言")
        return

    st.subheader("发言记录")

    for i, speech in enumerate(speeches[-10:], max(1, len(speeches) - 9)):
        sender = speech.get("sender", "未知")
        content = speech.get("content", "")

        # 截断过长的发言
        if len(content) > config.max_speech_length:
            content = content[:config.max_speech_length] + "..."

        with st.container():
            st.write(f"**{sender}**: {content}")
            st.divider()


def render_vote_history(votes: list[dict]):
    """渲染投票历史"""
    if not votes:
        st.info("暂无投票")
        return

    st.subheader("投票记录")

    for vote in votes[-10:]:
        voter = vote.get("voter", "未知")
        target = vote.get("target", "未知")
        st.write(f"📊 **{voter}** → **{target}**")


def render_night_result(result: dict):
    """渲染夜晚结果"""
    if not result:
        st.info("暂无夜晚结果")
        return

    st.subheader("🌙 夜晚结算")

    col1, col2 = st.columns(2)

    with col1:
        kill = result.get("kill_target")
        if kill:
            st.error(f"狼人刀了: **{kill}**")
        else:
            st.success("狼人空刀")

    with col2:
        verify = result.get("verify_result")
        if verify:
            is_wolf = "狼人" if verify.get("is_wolf") else "好人"
            st.info(f"预言家查验 {verify.get('target_name', '')}: **{is_wolf}**")


def render_game_timeline(events: list[dict]):
    """渲染游戏时间线"""
    if not events:
        return

    st.subheader("📜 游戏进程")

    with st.container():
        for event in events[-5:]:
            event_type = event.get("event", "")
            content = event.get("content", "")
            timestamp = event.get("timestamp", "")

            icon = {
                "game_start": "🎮",
                "night_start": "🌙",
                "day_start": "☀️",
                "speech": "💬",
                "vote": "🗳️",
                "player_death": "💀",
                "game_over": "🏆"
            }.get(event_type, "📌")

            st.write(f"{icon} {content}")


def render_werewolf_ui():
    """渲染狼人杀专用 UI 组件"""
    st.title("🐺 AI 狼人杀 - 观战台")

    # 阵营说明
    with st.expander("游戏规则说明"):
        st.markdown("""
        **角色阵营**:
        - 🔴 **狼人阵营**: 狼人 (每晚可以刀人)
        - 🟢 **好人阵营**: 预言家、女巫、守卫、猎人、村民

        **游戏流程**:
        1. 🌙 夜晚 - 狼人刀人，神职行动
        2. ☀️ 白天 - 发言、投票放逐
        3. 循环直到某阵营获胜
        """)


def render_stats_summary(stats: dict):
    """渲染统计摘要"""
    if not stats:
        return

    st.subheader("📊 统计摘要")

    col1, col2, col3 = st.columns(3)

    with col1:
        total = stats.get("total_games", 0)
        st.metric("总场次", total)

    with col2:
        good_rate = stats.get("good_win_rate", 0)
        st.metric("好人胜率", f"{good_rate:.1%}")

    with col3:
        wolf_rate = stats.get("wolf_win_rate", 0)
        st.metric("狼人胜率", f"{wolf_rate:.1%}")


def render_leaderboard(leaderboard: list[dict]):
    """渲染排行榜"""
    if not leaderboard:
        st.info("暂无排行数据")
        return

    st.subheader("🏆 Agent 排行榜")

    import pandas as pd

    df = pd.DataFrame(leaderboard)
    df = df.rename(columns={
        "agent_id": "Agent ID",
        "role": "角色",
        "generation": "代数",
        "win_rate": "胜率",
        "avg_survival": "平均存活"
    })

    st.dataframe(df, use_container_width=True)