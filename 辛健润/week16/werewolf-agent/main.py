"""AI 狼人杀入口文件"""
import os
from typing import Optional

from core.game_engine import GameEngine, Phase
from core.role_system import RoleType, STANDARD_12_PLAYER_CONFIG
from core.message_bus import MessageType
from core.event_system import EventType

from llm.base import DeepSeekLLM
from utils.config import config
from utils.logger import logger
from utils.information_processor import InformationProcessor

from agents.role_agents.werewolf import WerewolfAgent
from agents.role_agents.seer import SeerAgent
from agents.role_agents.witch import WitchAgent
from agents.role_agents.guard import GuardAgent
from agents.role_agents.villager import VillagerAgent
from agents.role_agents.hunter import HunterAgent

from agents.evolution.evolution_manager import EvolutionManager, EvolutionConfig


def create_llm() -> DeepSeekLLM:
    """创建 LLM 实例"""
    api_key = config.llm.api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("需要设置 DEEPSEEK_API_KEY 环境变量")

    return DeepSeekLLM(
        api_key=api_key,
        model=config.llm.model
    )


def create_agents(game_engine: GameEngine, llm) -> dict:
    """创建所有 Agent"""
    agents = {}

    for player_id, player in game_engine.state.players.items():
        role = player.role

        agent_map = {
            RoleType.WEREWOLF: WerewolfAgent,
            RoleType.SEER: SeerAgent,
            RoleType.WITCH: WitchAgent,
            RoleType.GUARD: GuardAgent,
            RoleType.VILLAGER: VillagerAgent,
            RoleType.HUNTER: HunterAgent,
        }

        agent_class = agent_map.get(role.role_type)
        if agent_class:
            agent = agent_class(
                agent_id=player_id,
                name=player.name,
                llm=llm,
                role=role
            )
            agents[player_id] = agent

            # 狼人需要知道同伴
            if role.role_type == RoleType.WEREWOLF:
                wolves = game_engine.state.wolf_players
                agent.set_companions([w for w in wolves if w != player_id])

    return agents


def get_agent_context(game_engine: GameEngine, player_id: str, info_processor: InformationProcessor) -> dict:
    """获取 Agent 上下文（包含信息处理器）"""
    context = game_engine.get_player_info(player_id)
    context["info_processor"] = info_processor
    return context


def run_night_phase(game_engine: GameEngine, agents: dict, info_processor: InformationProcessor):
    """执行夜晚阶段"""
    logger.info("night_start", f"第 {game_engine.state.day} 天夜晚开始", phase="night")

    # 清空夜晚行动
    game_engine.state.night_actions.clear()

    # 收集各角色行动
    for player_id, player in game_engine.state.players.items():
        if not player.alive:
            continue

        agent = agents.get(player_id)
        if not agent:
            continue

        # 获取角色信息（含信息处理器）
        context = get_agent_context(game_engine, player_id, info_processor)

        # 狼人行动
        if player.role.role_type == RoleType.WEREWOLF:
            action = agent._act_night(game_engine.state, context)
            if action.get("action") == "kill":
                from core.game_engine import NightAction
                game_engine.record_night_action(NightAction(
                    player_id=player_id,
                    action_type="kill",
                    target=action["target"]
                ))

        # 预言家行动
        elif player.role.role_type == RoleType.SEER:
            action = agent._act_night(game_engine.state, context)
            if action.get("action") == "verify":
                from core.game_engine import NightAction
                game_engine.record_night_action(NightAction(
                    player_id=player_id,
                    action_type="verify",
                    target=action["target"]
                ))
                # 记录查验结果
                seer_agent = agent
                target = action["target"]
                is_wolf = game_engine.state.players[target].role.camp.value == "wolf"
                seer_agent.record_verify_result(target, is_wolf)
                logger.info("seer_verify", f"{player.name} 查验了 {action['target_name']}",
                          player=player_id, phase="night")

        # 女巫行动
        elif player.role.role_type == RoleType.WITCH:
            action = agent._act_night(game_engine.state, context)
            if action.get("action") == "heal":
                from core.game_engine import NightAction
                game_engine.record_night_action(NightAction(
                    player_id=player_id,
                    action_type="heal",
                    target=action["target"]
                ))
                logger.info("witch_heal", f"{player.name} 使用解药",
                           player=player_id, phase="night")
            elif action.get("action") == "poison":
                from core.game_engine import NightAction
                game_engine.record_night_action(NightAction(
                    player_id=player_id,
                    action_type="poison",
                    target=action["target"]
                ))
                logger.info("witch_poison", f"{player.name} 使用毒药",
                           player=player_id, phase="night")

        # 守卫行动
        elif player.role.role_type == RoleType.GUARD:
            action = agent._act_night(game_engine.state, context)
            if action.get("action") == "guard":
                from core.game_engine import NightAction
                game_engine.record_night_action(NightAction(
                    player_id=player_id,
                    action_type="guard",
                    target=action["target"]
                ))
                logger.info("guard_protect", f"{player.name} 守护了 {action['target_name']}",
                           player=player_id, phase="night")

    # 结算夜晚
    night_result = game_engine.resolve_night()

    if night_result["kill_target"]:
        logger.info("kill", f"狼人刀了 {game_engine.state.players[night_result['kill_target']].name}",
                   phase="night")

    if night_result["verify_result"]:
        result = night_result["verify_result"]
        is_wolf_str = "狼人" if result["is_wolf"] else "好人"
        logger.info("seer_result", f"查验结果：{result['target_name']} 是 {is_wolf_str}",
                   player=result["player"], phase="night")

    # 处理死亡
    deaths = game_engine.process_deaths()
    for death_id in deaths:
        logger.info("player_death", f"{game_engine.state.players[death_id].name} 死亡",
                   phase="night", player=death_id)
        info_processor.process_death(death_id, "夜晚死亡")

    return night_result


def run_day_phase(game_engine: GameEngine, agents: dict, info_processor: InformationProcessor):
    """执行白天阶段"""
    logger.info("day_start", f"第 {game_engine.state.day} 天白天开始", phase="day")

    # 发言阶段
    for player_id, player in game_engine.state.players.items():
        if not player.alive:
            continue

        agent = agents.get(player_id)
        if not agent:
            continue

        context = get_agent_context(game_engine, player_id, info_processor)
        action = agent._act_speech(game_engine.state, context)

        if action.get("action") == "speech":
            speech = action["content"]
            game_engine.message_bus.broadcast(
                sender=player_id,
                content=speech,
                msg_type=MessageType.PUBLIC_SPEECH
            )
            logger.info("speech", speech, player=player_id, phase="day")

            # 处理发言信息
            info_processor.process_speech(player_id, speech)

    # 投票阶段
    logger.info("vote_start", "投票开始", phase="vote")

    for player_id, player in game_engine.state.players.items():
        if not player.alive:
            continue

        agent = agents.get(player_id)
        if not agent:
            continue

        context = get_agent_context(game_engine, player_id, info_processor)
        action = agent._act_vote(game_engine.state, context)

        if action.get("action") == "vote" and action.get("target"):
            game_engine.record_vote(player_id, action["target"])
            logger.info("vote", f"{player.name} 投票给 {action['target_name']}",
                       player=player_id, phase="vote")

            # 处理投票信息
            info_processor.process_vote(player_id, action["target"])

    # 结算投票
    voted_out = game_engine.resolve_vote()
    if voted_out:
        game_engine.kill_player(voted_out, "投票出局")
        logger.info("voted_out", f"{game_engine.state.players[voted_out].name} 被投票出局",
                   player=voted_out, phase="vote")
        info_processor.process_death(voted_out, "投票出局")
    else:
        logger.info("vote_tie", "投票平票，无人出局", phase="vote")


def run_game(llm, info_processor: InformationProcessor = None) -> dict:
    """运行一局游戏"""
    if info_processor is None:
        info_processor = InformationProcessor()

    # 初始化游戏
    game_engine = GameEngine()
    game_engine.initialize()
    logger.info("game_start", f"游戏开始，共 {len(game_engine.state.players)} 名玩家")

    # 创建 Agent
    agents = create_agents(game_engine, llm)

    # 游戏主循环
    while True:
        winner = game_engine.check_game_end()
        if winner:
            break

        # 夜晚阶段
        game_engine.set_phase(Phase.NIGHT)
        run_night_phase(game_engine, agents, info_processor)

        # 检查游戏结束
        winner = game_engine.check_game_end()
        if winner:
            break

        # 白天阶段
        game_engine.set_phase(Phase.DAY_SPEECH)
        run_day_phase(game_engine, agents, info_processor)

        # 检查游戏结束
        winner = game_engine.check_game_end()
        if winner:
            break

        # 进入下一天
        game_engine.next_day()

    # 游戏结束
    game_engine.set_phase(Phase.GAME_OVER)
    winner_camp = "好人阵营" if winner == "good" else "狼人阵营"
    logger.info("game_over", f"游戏结束，{winner_camp} 获胜！")

    # 发布游戏结束事件
    from core.event_system import Event
    game_engine.event_system.publish(Event(type=EventType.GAME_OVER, data={"winner": winner}))

    return {
        "winner": winner,
        "day": game_engine.state.day,
        "survivors": [game_engine.state.players[p].name for p in game_engine.state.alive_players],
        "logs": game_engine.get_logs()
    }


def run_multiple_games(num_games: int = 5) -> dict:
    """运行多局游戏并收集统计"""
    print("\n" + "=" * 60)
    print(f"批量运行 {num_games} 局游戏")
    print("=" * 60)

    llm = create_llm()
    stats = {"good_wins": 0, "wolf_wins": 0, "games": []}

    for i in range(num_games):
        print(f"\n--- 第 {i+1}/{num_games} 局 ---")
        info_processor = InformationProcessor()

        try:
            result = run_game(llm, info_processor)
            stats["games"].append(result)

            if result["winner"] == "good":
                stats["good_wins"] += 1
            else:
                stats["wolf_wins"] += 1

            print(f"结果: {'好人' if result['winner'] == 'good' else '狼人'} 获胜")
            print(f"天数: {result['day']}")

        except Exception as e:
            print(f"游戏出错: {e}")

    # 打印统计
    print("\n" + "=" * 60)
    print("批量运行统计")
    print("=" * 60)
    print(f"总场次: {num_games}")
    print(f"好人获胜: {stats['good_wins']} ({stats['good_wins']/num_games:.1%})")
    print(f"狼人获胜: {stats['wolf_wins']} ({stats['wolf_wins']/num_games:.1%})")
    print("=" * 60)

    return stats


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("AI 狼人杀 - 多智能体博弈系统")
    print("=" * 60 + "\n")

    try:
        llm = create_llm()
        print("LLM 初始化成功\n")

        # 运行游戏
        result = run_game(llm)

        # 打印结果
        print("\n" + "=" * 60)
        print("游戏结果")
        print("=" * 60)
        print(f"获胜阵营: {'好人' if result['winner'] == 'good' else '狼人'}")
        print(f"存活天数: {result['day']}")
        print(f"存活玩家: {', '.join(result['survivors'])}")
        print("=" * 60)

        # 打印日志
        print("\n游戏日志:")
        logger.print_summary()

    except ValueError as e:
        print(f"配置错误: {e}")
        print("请设置 DEEPSEEK_API_KEY 环境变量")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()