"""AI 狼人杀 - Streamlit 前端"""
import streamlit as st
import time
from datetime import datetime
import json
import sys
import os
import random

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.game_viewer import (
    render_werewolf_ui,
    render_game_status,
    render_player_list,
    render_speech_history,
    render_vote_history,
    render_night_result,
    render_game_timeline,
    render_stats_summary,
    render_leaderboard,
    GameViewerConfig
)


def init_session_state():
    """初始化会话状态"""
    if "game_state" not in st.session_state:
        st.session_state.game_state = None
    if "speeches" not in st.session_state:
        st.session_state.speeches = []
    if "votes" not in st.session_state:
        st.session_state.votes = []
    if "events" not in st.session_state:
        st.session_state.events = []
    if "stats" not in st.session_state:
        st.session_state.stats = {"good_wins": 0, "wolf_wins": 0, "total": 0}
    if "phase_index" not in st.session_state:
        st.session_state.phase_index = 0


def get_next_phase():
    """获取下一个阶段（按游戏顺序），并递增索引"""
    if st.session_state.phase_index == 0:
        phase = "night_start"
    elif st.session_state.phase_index == 1:
        phase = "night_end"
    elif st.session_state.phase_index == 2:
        phase = "day_speech"
    elif st.session_state.phase_index == 3:
        phase = "vote"
    elif st.session_state.phase_index == 4:
        phase = "vote_end"
    elif st.session_state.phase_index == 5:
        phase = "day_death"
    else:
        # 重置到夜晚开始，进入新的一天
        st.session_state.game_state["day"] = st.session_state.game_state.get("day", 1) + 1
        st.session_state.phase_index = 0
        phase = "night_start"

    # 递增索引
    st.session_state.phase_index += 1
    return phase


def simulate_game_step():
    """模拟游戏步骤（按正确顺序推进）"""
    phase = get_next_phase()
    day = st.session_state.game_state.get("day", 1)
    phase_names = {
        "night_start": "🌙 夜晚开始",
        "night_end": "🌙 夜晚结束",
        "day_speech": "☀️ 白天发言",
        "vote": "🗳️ 投票开始",
        "vote_end": "🗳️ 投票结束",
        "day_death": "💀 结算死亡"
    }

    result = {
        "phase": phase,
        "event": phase,
        "content": phase_names.get(phase, phase)
    }

    # 根据阶段生成具体内容
    if phase == "night_start":
        result["content"] = f"🌙 第 {day} 天夜晚开始 - 狼人请睁眼"

    elif phase == "night_end":
        alive = st.session_state.game_state.get("alive", 12)
        if alive > 0:
            if random.random() > 0.3:
                killed = random.choice(["玩家3", "玩家5", "玩家7", "玩家9"])
                result["content"] = f"🌙 狼人刀了 {killed}"
                st.session_state.game_state["alive"] = max(0, alive - 1)
            else:
                result["content"] = "🌙 平安夜，无人伤亡"

    elif phase == "day_speech":
        speakers = ["玩家1", "玩家2", "玩家4", "玩家6", "玩家8", "玩家10", "玩家11", "玩家12"]
        speaker = random.choice(speakers)
        speeches = [
            "我觉得今天应该出掉可疑的玩家。",
            "大家畅所欲言，共同分析。",
            "我观察某人的发言风格，感觉比较可疑。",
            "今天我们需要找出狼人。",
            "昨晚平安夜，狼人怂了。",
            "预言家请报查验。"
        ]
        result = {
            "phase": "day_speech",
            "event": "speech",
            "content": f"**{speaker}**: {random.choice(speeches)}"
        }

    elif phase == "vote":
        voters = ["玩家1", "玩家2", "玩家4", "玩家6", "玩家8", "玩家10", "玩家11", "玩家12"]
        voter = random.choice(voters)
        targets = [p for p in voters if p != voter]
        target = random.choice(targets)
        result = {
            "phase": "vote",
            "event": "vote",
            "content": f"**{voter}** 投票给 **{target}**"
        }

    elif phase == "vote_end":
        alive = st.session_state.game_state.get("alive", 12)
        if alive > 0:
            voted_out = random.choice([None, "玩家3", "玩家5", "玩家7", "玩家9", "玩家11"])
            if voted_out:
                result["content"] = f"🗳️ {voted_out} 被投票出局"
                st.session_state.game_state["alive"] = max(0, alive - 1)
            else:
                result["content"] = "🗳️ 投票平票，无人出局"

    elif phase == "day_death":
        result["content"] = f"☀️ 第 {day} 天结束，进入第 {day + 1} 天"

    return result


def calculate_camp_counts(alive: int) -> tuple:
    """根据存活人数计算狼人和好人数量"""
    if alive is None:
        alive = 12
    # 12人局：3狼 + 9好人
    total_wolves = 3
    total_good = 9

    # 初始存活12人时狼人是3个
    # 存活人数越少，按比例估算（保证狼人不超过存活人数的40%）
    if alive >= 12:
        return (total_wolves, total_good)
    elif alive >= 10:
        return (3, 7)
    elif alive >= 8:
        return (2, 6)
    elif alive >= 6:
        return (2, 4)
    elif alive >= 4:
        return (1, 3)
    elif alive >= 2:
        return (1, 1)
    else:
        return (0, 0)


def main():
    """主函数"""
    st.set_page_config(
        page_title="AI 狼人杀 - 观战台",
        page_icon="🐺",
        layout="wide"
    )

    init_session_state()
    render_werewolf_ui()

    # 侧边栏控制
    with st.sidebar:
        st.header("控制面板")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎮", help="开始新游戏", width='stretch'):
                st.session_state.game_state = {"day": 1, "phase": "night", "alive": 12}
                st.session_state.speeches = []
                st.session_state.votes = []
                st.session_state.events = []
                st.session_state.phase_index = 0
                st.rerun()

        with col2:
            # 检查游戏是否结束
            alive = st.session_state.game_state.get("alive", 12) if st.session_state.game_state else 12
            wolf_c, good_c = calculate_camp_counts(alive)
            game_over = bool(st.session_state.game_state and (wolf_c == 0 or wolf_c >= good_c or alive <= 0))

            if st.button("▶️", help="下一步", width='stretch', disabled=game_over):
                if st.session_state.game_state is None:
                    st.session_state.game_state = {"day": 1, "phase": "night", "alive": 12}
                step = simulate_game_step()
                st.session_state.events.append(step)

                if step["event"] == "speech":
                    sender = step["content"].split(":")[0].replace("**", "")
                    content = step["content"].split(":", 1)[1].replace("**", "").strip()
                    st.session_state.speeches.append({
                        "sender": sender,
                        "content": content
                    })
                elif step["event"] == "vote":
                    parts = step["content"].split("**")
                    if len(parts) >= 4:
                        st.session_state.votes.append({
                            "voter": parts[1],
                            "target": parts[3]
                        })
                st.rerun()

        if st.button("⏩ 自动播放 10 步", width='stretch', disabled=game_over):
            if st.session_state.game_state is None:
                st.session_state.game_state = {"day": 1, "phase": "night", "alive": 12}
            for _ in range(10):
                step = simulate_game_step()
                st.session_state.events.append(step)
                if step["event"] == "speech":
                    sender = step["content"].split(":")[0].replace("**", "")
                    content = step["content"].split(":", 1)[1].replace("**", "").strip()
                    st.session_state.speeches.append({
                        "sender": sender,
                        "content": content
                    })
                elif step["event"] == "vote":
                    parts = step["content"].split("**")
                    if len(parts) >= 4:
                        st.session_state.votes.append({
                            "voter": parts[1],
                            "target": parts[3]
                        })
            st.rerun()

        st.divider()

        # 模拟设置
        st.subheader("设置")
        show_speeches = st.checkbox("显示发言", value=True)
        show_votes = st.checkbox("显示投票", value=True)

        config = GameViewerConfig(
            show_speeches=show_speeches,
            show_votes=show_votes
        )

    # 主内容区
    col_main, col_side = st.columns([3, 1])

    with col_main:
        # 游戏状态
        if st.session_state.game_state:
            alive = st.session_state.game_state.get("alive", 12)
            wolf_count, good_count = calculate_camp_counts(alive)

            # 检查游戏是否结束
            game_over = False
            winner = None

            if alive <= 0:
                game_over = True
                winner = "draw"
            elif wolf_count == 0:
                game_over = True
                winner = "good"
            elif wolf_count >= good_count:
                game_over = True
                winner = "wolf"

            if game_over:
                st.error("🎉 游戏结束！")
                if winner == "good":
                    st.success("🏆 好人阵营获胜！")
                elif winner == "wolf":
                    st.error("🐺 狼人阵营获胜！")
                else:
                    st.warning("🤝 平局！")

                if st.button("🔄 重新开始", width='stretch'):
                    st.session_state.game_state = {"day": 1, "phase": "night", "alive": 12}
                    st.session_state.speeches = []
                    st.session_state.votes = []
                    st.session_state.events = []
                    st.session_state.phase_index = 0
                    st.rerun()
            else:
                status = {
                    "day": st.session_state.game_state.get("day", 1),
                    "phase": st.session_state.game_state.get("phase", "night"),
                    "wolf_count": wolf_count,
                    "good_count": good_count
                }
                render_game_status(status, config)

            st.divider()

            # 发言历史
            if config.show_speeches:
                render_speech_history(st.session_state.speeches, config)

            # 投票历史
            if config.show_votes:
                render_vote_history(st.session_state.votes)

            # 游戏时间线
            render_game_timeline(st.session_state.events)

        else:
            st.info("👈 点击左侧「🎮」按钮开始新游戏")

            # 显示示例数据
            st.subheader("示例：12人标准局")
            sample_players = [
                {"name": "玩家1", "role": "狼人", "alive": True, "camp": "wolf"},
                {"name": "玩家2", "role": "预言家", "alive": True, "camp": "good"},
                {"name": "玩家3", "role": "女巫", "alive": True, "camp": "good"},
                {"name": "玩家4", "role": "守卫", "alive": True, "camp": "good"},
                {"name": "玩家5", "role": "猎人", "alive": True, "camp": "good"},
                {"name": "玩家6", "role": "村民", "alive": True, "camp": "good"},
                {"name": "玩家7", "role": "村民", "alive": True, "camp": "good"},
                {"name": "玩家8", "role": "狼人", "alive": False, "camp": "wolf"},
                {"name": "玩家9", "role": "狼人", "alive": False, "camp": "wolf"},
                {"name": "玩家10", "role": "村民", "alive": True, "camp": "good"},
                {"name": "玩家11", "role": "村民", "alive": True, "camp": "good"},
                {"name": "玩家12", "role": "村民", "alive": True, "camp": "good"},
            ]
            render_player_list(sample_players)

    with col_side:
        # 统计信息
        st.subheader("📊 统计")
        stats = st.session_state.stats
        st.metric("总场次", stats["total"])
        st.metric("好人胜率", f"{stats['good_wins'] / max(stats['total'], 1):.1%}")
        st.metric("狼人胜率", f"{stats['wolf_wins'] / max(stats['total'], 1):.1%}")

        st.divider()

        # 角色说明
        st.subheader("📖 角色说明")
        st.markdown("""
        | 角色 | 阵营 | 能力 |
        |------|------|------|
        | 🐺 狼人 | 狼人 | 刀人、隐藏身份 |
        | 👁️ 预言家 | 好人 | 验人查阵营 |
        | 🧪 女巫 | 好人 | 救人、毒人 |
        | 🛡️ 守卫 | 好人 | 守护玩家 |
        | 🏹 猎人 | 好人 | 死亡可开枪 |
        | 👤 村民 | 好人 | 无特殊能力 |
        """)

        st.divider()

        # Leaderboard
        st.subheader("🏆 排行榜")
        sample_leaderboard = [
            {"agent_id": "agent_1", "role": "狼人", "generation": 3, "win_rate": 0.75, "avg_survival": 4.2},
            {"agent_id": "agent_2", "role": "预言家", "generation": 2, "win_rate": 0.65, "avg_survival": 3.8},
            {"agent_id": "agent_3", "role": "女巫", "generation": 2, "win_rate": 0.60, "avg_survival": 4.5},
        ]
        render_leaderboard(sample_leaderboard)


if __name__ == "__main__":
    main()