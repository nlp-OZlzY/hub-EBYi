"""自演化狼人杀 Agent 主入口

用法：
    python evolve.py --rounds 10                    # 运行10局自演化
    python evolve.py --rounds 10 --config standard_6  # 指定角色配置
    python evolve.py --history werewolf             # 查看狼人 prompt 演化历史
    python evolve.py --rollback werewolf v003       # 回滚狼人 prompt 到指定版本
"""

import argparse
import asyncio
from typing import List

import os

from engine.game_engine import GameEngine, get_role_config, shuffle_roles, ROLE_CONFIGS
from metrics.collector import MetricsCollector
from agent.reflector import SelfReflector
from prompt_store.store import PromptStore


def _get_agent_md_path(role: str) -> str:
    """获取角色 Agent.md 文件路径"""
    return os.path.join("prompts", "agents", f"{role}_agent.md")


def _read_agent_md(role: str) -> str:
    """读取角色 Agent.md，兼容旧格式"""
    # 优先读 Agent.md
    path = _get_agent_md_path(role)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # 兼容旧格式
    return PromptStore().read_prompt(role)


def _write_agent_md(role: str, content: str) -> None:
    """写入角色 Agent.md"""
    path = _get_agent_md_path(role)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def get_involved_roles(engine: GameEngine) -> List[str]:
    """获取本局游戏中实际参与的角色类型列表"""
    roles = set()
    for player in engine.game_state.players:
        roles.add(player.role.role_type.value)
    return list(roles)


def get_role_logs(role: str, engine: GameEngine) -> str:
    """获取指定角色视角的游戏日志（含信息隔离过滤）"""
    target_players = [
        p for p in engine.game_state.players
        if p.role.role_type.value == role
    ]
    if not target_players:
        return "该角色未参与本局游戏。"

    player = target_players[0]
    player_context = engine.game_state.get_player_private_context(player.player_id)
    dialogues = player_context.get("dialogues", [])

    lines = []
    for d in dialogues:
        action = d.get("action", "")
        content = d.get("content", "")
        target = d.get("target")

        if action == "wolf_vote":
            lines.append(f"[夜晚] 建议击杀玩家{target}（{d.get('reasoning', '')}）")
        elif action == "seer_check":
            result = "狼人" if d.get("result") == "wolf" else "好人"
            lines.append(f"[夜晚] 查验玩家{target}，结果：{result}")
        elif action == "heal":
            lines.append(f"[夜晚] 使用解药救活了玩家{target}")
        elif action == "poison":
            lines.append(f"[夜晚] 使用毒药毒杀了玩家{target}")
        elif action == "speech":
            lines.append(f"[白天] 发言：{content[:100]}")
        elif action == "vote":
            lines.append(f"[白天] 投票给玩家{target}")
        elif action == "hunter_shot":
            lines.append(f"[夜晚] 开枪带走了玩家{target}")

    for death in engine.death_records:
        if death["player_id"] == player.player_id:
            cause_map = {
                "night_kill": "被狼人杀害",
                "poison": "被毒杀",
                "vote": "被投票出局",
                "shoot": "被枪杀",
            }
            lines.append(f"[死亡] 在第{death['day']}天{cause_map.get(death['cause'], death['cause'])}")

    return "\n".join(lines) if lines else "无记录"


async def evolve_loop(rounds: int, config_name: str, shuffle: bool = True):
    """自演化主循环

    每轮流程：
    1. 初始化游戏 → 运行至结束 → 收集指标
    2. 对每个参与角色：用 SelfReflector 分析指标 + 日志 → 生成改进 prompt
    3. 若 prompt 有变化：保存旧版本到历史 → 写入新 prompt
    4. 下一轮游戏使用更新后的 prompt，如此迭代

    这就是「读懂自己 → 修改自己 → 运行自己」的自演化循环。
    """
    store = PromptStore()
    reflector = SelfReflector()

    good_wins = 0
    evil_wins = 0
    prompt_changes = {}

    print(f"{'='*50}")
    print("自演化狼人杀 Agent 启动")
    print(f"轮次: {rounds}, 配置: {config_name}")
    print(f"{'='*50}")

    for round_num in range(1, rounds + 1):
        print(f"\n{'='*30} 第 {round_num}/{rounds} 轮 {'='*30}")

        role_assignment = get_role_config(config_name)
        if shuffle:
            role_assignment = shuffle_roles(role_assignment)

        player_names = [f"玩家{i}" for i in range(len(role_assignment))]

        engine = GameEngine(player_names)
        await engine.initialize(role_assignment)

        while not engine.game_state.is_game_over():
            await engine.step()

        winner = engine.game_state.get_winner()
        if winner == "good":
            good_wins += 1
        else:
            evil_wins += 1

        winner_text = "好人" if winner == "good" else "狼人"
        print(f"胜方: {winner_text}")

        metrics = MetricsCollector.collect(engine)

        involved_roles = get_involved_roles(engine)
        changed_roles = []

        for role in involved_roles:
            role_metrics = metrics.get(role)
            if not role_metrics:
                continue

            current_prompt = _read_agent_md(role)
            if not current_prompt:
                continue

            game_logs = get_role_logs(role, engine)

            print(f"  反思 {role}...")
            new_prompt = await reflector.reflect(
                role=role,
                current_prompt=current_prompt,
                metrics=role_metrics["metrics"],
                game_logs=game_logs,
            )

            if new_prompt.strip() != current_prompt.strip():
                store.save_version(role, current_prompt)
                _write_agent_md(role, new_prompt)
                changed_roles.append(role)
                prompt_changes[role] = prompt_changes.get(role, 0) + 1
                print(f"    {role} Agent.md 已更新")
            else:
                print(f"    {role} Agent.md 无变化")

        changed_text = ", ".join(changed_roles) if changed_roles else "无"
        print(f"本轮更新: {changed_text}")

    print(f"\n{'='*50}")
    print("演化完成！")
    print(f"共 {rounds} 局, 好人胜率: {good_wins/rounds*100:.0f}%, 狼人胜率: {evil_wins/rounds*100:.0f}%")
    if prompt_changes:
        changes_text = ", ".join(f"{r}: {c}次" for r, c in prompt_changes.items())
        print(f"Prompt变化: {changes_text}")
    print(f"{'='*50}")


def show_history(role: str):
    """显示角色的 prompt 演化历史"""
    store = PromptStore()
    versions = store.list_versions(role)
    if not versions:
        print(f"{role} 没有演化历史。")
        return
    print(f"{role} 演化历史（共 {len(versions)} 个版本）：")
    for v in versions:
        print(f"  - {v}")


def do_rollback(role: str, version: str):
    """回滚角色的 prompt 到指定版本，并写入 Agent.md"""
    store = PromptStore()
    try:
        store.rollback(role, version)
        # 同步写入 Agent.md
        content = store.read_prompt(role)
        if content:
            _write_agent_md(role, content)
        print(f"{role} 已回滚到 {version}")
    except FileNotFoundError as e:
        print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(description="自演化狼人杀 Agent")
    parser.add_argument("--rounds", type=int, default=5, help="演化轮次（默认5）")
    parser.add_argument(
        "--config",
        type=str,
        default="standard_6",
        choices=list(ROLE_CONFIGS.keys()),
        help="角色配置（默认 standard_6）",
    )
    parser.add_argument("--history", type=str, metavar="ROLE", help="查看指定角色的 prompt 演化历史")
    parser.add_argument(
        "--rollback",
        nargs=2,
        metavar=("ROLE", "VERSION"),
        help="回滚指定角色的 prompt 到指定版本",
    )

    args = parser.parse_args()

    if args.history:
        show_history(args.history)
    elif args.rollback:
        do_rollback(args.rollback[0], args.rollback[1])
    else:
        asyncio.run(evolve_loop(args.rounds, args.config))


if __name__ == "__main__":
    main()
