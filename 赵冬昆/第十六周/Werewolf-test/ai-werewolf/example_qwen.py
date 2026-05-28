"""
使用真实 Qwen 模型进行狼人杀游戏演示
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置stdout编码
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
from engine.game_manager import GameManager
from engine.types import GameConfig
from agent.llm.qwen_client import init_qwen_client, QwenClient
from agent.core.base_agent import BaseAgent
from evaluation.metrics import MetricsCollector

def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def run_game_with_qwen(api_key: str, model_name: str = "qwen3.5-plus-2026-04-20"):
    """使用Qwen模型运行一局游戏"""
    print_separator(f"使用 Qwen 模型: {model_name}")
    
    # 初始化 Qwen 客户端
    print("\n[1] 初始化 Qwen 客户端...")
    try:
        qwen_client = init_qwen_client(api_key, model_name)
        print(f"[OK] Qwen 客户端已初始化")
    except Exception as e:
        print(f"[ERROR] 初始化 Qwen 客户端失败: {e}")
        return
    
    # 创建游戏
    print("\n[2] 创建游戏...")
    gm = GameManager()
    
    config = GameConfig(
        name="Qwen演示局",
        total_players=6,
        werewolf_count=2,
        seer_count=1,
        witch_count=1,
        villager_count=2
    )
    
    game_id = gm.create_game(config)
    state = gm.get_game(game_id)
    
    print(f"[OK] 游戏创建成功，ID: {game_id}")
    print("  玩家配置:")
    for p in state.players:
        print(f"    玩家{p.player_id} ({p.name}): {p.role.value}")
    
    # 创建 Agent
    print("\n[3] 创建 Agent...")
    agents = {}
    for player in state.players:
        agent = BaseAgent(player, use_llm=True, llm_client=qwen_client)
        agents[player.player_id] = agent
    print(f"[OK] 创建了 {len(agents)} 个 Agent")
    
    # 运行游戏
    print("\n[4] 开始游戏...")
    max_steps = 50
    step_count = 0
    
    while not state.is_game_over and step_count < max_steps:
        print(f"\n--- 第 {state.day_number} 天, {state.phase.value} 阶段 ---")
        
        # 让 Agent 行动
        if state.phase.value in ["speech"]:
            print("\n[发言阶段]")
            for player in state.get_alive_players():
                if player.player_id in agents:
                    agent = agents[player.player_id]
                    action = agent.act(state)
                    print(f"\n{player.name} ({player.role.value}):")
                    print(f"  思考: {action.inner_monologue}")
                    print(f"  发言: {action.action.content}")
                    
                    # 记录对话
                    state.dialogues.append({
                        "speaker": player.name,
                        "content": action.action.content,
                        "phase": "speech"
                    })
        
        # 游戏步进
        gm.step(game_id)
        state = gm.get_game(game_id)
        
        # 显示阶段结果
        if state.phase.value in ["day_end", "night_result"]:
            if state.step_data:
                if "eliminated" in state.step_data:
                    eliminated = state.step_data["eliminated"]
                    print(f"\n[结果] {eliminated} 被投票出局")
                elif "killed" in state.step_data:
                    killed = state.step_data["killed"]
                    print(f"\n[结果] {killed} 被狼人杀死")
        
        step_count += 1
    
    # 游戏结束
    print("\n" + "=" * 60)
    print("  游戏结束")
    print("=" * 60)
    print(f"\n获胜方: {state.winner.value if state.winner else '未知'}")
    
    # 收集评测数据
    print("\n[5] 收集评测数据...")
    metrics_collector = MetricsCollector()
    evaluation = metrics_collector.collect_from_game(state)
    
    print("\n玩家得分:")
    for pid, metrics in sorted(evaluation.player_metrics.items()):
        mvp_marker = " [MVP]" if metrics.is_mvp else ""
        print(f"  玩家{pid} ({metrics.role}): {metrics.overall_score}分{mvp_marker}")
    
    return state

def main():
    print("""
============================================================
     AI 狼人杀 - Qwen 模型演示 (DashScope)
============================================================
    """)
    
    # 获取 DashScope API Key
    api_key = "sk-703b34e4e06841b49c72940dade1022d"
    
    if not api_key:
        print("[ERROR] DashScope API Key 未提供")
        return
    
    # 选择模型 - 使用 qwen3.6-35b-a3b（有可用额度）
    model_name = "qwen3.6-35b-a3b"
    print(f"使用模型: {model_name}")
    
    # 运行游戏
    try:
        run_game_with_qwen(api_key, model_name)
    except Exception as e:
        print(f"\n[ERROR] 游戏运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
