"""
评测系统与自演化演示脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.game_manager import GameManager
from engine.types import GameConfig
from evaluation.metrics import MetricsCollector
from evaluation.leaderboard import LeaderboardManager
from evaluation.replay_analyzer import ReplayAnalyzer
from agent.evolution.self_evolution import SelfEvolutionEngine
import time

# 设置stdout编码
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def print_separator(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def demo_evaluation_system():
    """演示评测系统"""
    print_separator("方向 B: 多维评测与复盘 Leaderboard 演示")
    
    # 初始化组件
    gm = GameManager()
    metrics_collector = MetricsCollector()
    leaderboard_manager = LeaderboardManager(storage_path="./demo_leaderboard.json")
    replay_analyzer = ReplayAnalyzer()
    
    # 注册Agent
    print("\n[1] 注册Agent到Leaderboard...")
    for i in range(6):
        leaderboard_manager.register_agent(
            agent_id=f"agent_{i}",
            version="1.0.0",
            model_name="gpt-4" if i % 2 == 0 else "claude-3"
        )
    print("[OK] 6个Agent已注册")
    
    # 运行多局游戏
    print("\n[2] 运行多局游戏并收集评测数据...")
    num_games = 3
    
    for game_num in range(num_games):
        print(f"\n  游戏 {game_num + 1}/{num_games}:")
        
        # 创建游戏
        config = GameConfig(
            name=f"评测局_{game_num + 1}",
            total_players=6,
            werewolf_count=2,
            seer_count=1,
            witch_count=1,
            villager_count=2
        )
        
        game_id = gm.create_game(config)
        state = gm.get_game(game_id)
        
        # 运行游戏
        max_steps = 50
        step_count = 0
        start_time = time.time()
        
        while not state.is_game_over and step_count < max_steps:
            gm.step(game_id)
            step_count += 1
        
        game_duration = time.time() - start_time
        
        # 收集评测数据
        evaluation = metrics_collector.collect_from_game(state)
        evaluation.game_duration_seconds = game_duration
        
        # 记录到Leaderboard
        for player_id, player_metrics in evaluation.player_metrics.items():
            leaderboard_manager.record_game_result(
                agent_id=f"agent_{player_id}",
                game_result={
                    "is_winner": player_metrics.is_winner,
                    "is_mvp": player_metrics.is_mvp,
                    "score": player_metrics.overall_score,
                    "survival_rounds": player_metrics.survival_rounds,
                    "role": player_metrics.role
                }
            )
        
        # 复盘分析
        analysis = replay_analyzer.analyze_game(state)
        
        print(f"    - 游戏ID: {game_id}")
        print(f"    - 获胜方: {state.winner.value if state.winner else '未知'}")
        print(f"    - 总回合: {state.day_number}")
        print(f"    - 关键时刻: {analysis['critical_moments_count']} 个")
        print(f"    - 洞察: {', '.join(analysis['key_insights'][:2])}")
        
        # 显示玩家得分
        print(f"    - 玩家得分:")
        for pid, metrics in sorted(evaluation.player_metrics.items()):
            mvp_marker = " [MVP]" if metrics.is_mvp else ""
            print(f"      玩家{pid} ({metrics.role}): {metrics.overall_score}分{mvp_marker}")
    
    print("\n[OK] 游戏数据收集完成")
    
    # 显示排行榜
    print_separator("[3] Leaderboard 排行榜")
    
    leaderboard = leaderboard_manager.get_leaderboard(min_games=1)
    print(f"\n总排行榜 (共{len(leaderboard)}个Agent):")
    print("-" * 80)
    print(f"{'排名':<6}{'Agent':<12}{'模型':<12}{'场次':<8}{'胜率':<10}{'MVP率':<10}{'均分':<10}{'趋势':<8}")
    print("-" * 80)
    
    for entry in leaderboard[:10]:
        trend_icon = "UP" if entry["trend"] == "up" else "DOWN" if entry["trend"] == "down" else "STABLE"
        print(f"{entry['rank']:<6}{entry['agent_id']:<12}{entry['model_name']:<12}"
              f"{entry['games_played']:<8}{entry['win_rate']:<10.1f}"
              f"{entry['mvp_rate']:<10.1f}{entry['avg_score']:<10.1f}{trend_icon:<8}")
    
    # 按角色排行榜
    print("\n\n按角色排行榜 (Top 3):")
    for role in ["werewolf", "seer", "villager"]:
        role_board = leaderboard_manager.get_leaderboard(role=role, min_games=1)
        if role_board:
            print(f"\n  {role.upper()}:")
            for i, entry in enumerate(role_board[:3], 1):
                print(f"    {i}. {entry['agent_id']} - 均分: {entry['avg_score']:.1f}")
    
    # 显示复盘报告
    print_separator("[4] 游戏复盘报告")
    
    if metrics_collector.evaluations:
        latest_eval = metrics_collector.evaluations[-1]
        report = metrics_collector.generate_report(latest_eval.game_id)
        print(report)
    
    # 导出数据
    print_separator("[5] 导出数据")
    leaderboard_manager.export_report("./demo_leaderboard_report.json")
    metrics_collector.export_to_json("./demo_metrics.json")
    print("[OK] 数据已导出到:")
    print("  - demo_leaderboard_report.json")
    print("  - demo_metrics.json")

def demo_self_evolution():
    """演示自演化系统"""
    print_separator("方向 A: 通用 Agent 自演化系统演示")
    
    # 初始化自演化引擎
    print("\n[1] 初始化自演化引擎...")
    evolution_engine = SelfEvolutionEngine(
        agent_id="agent_0",
        storage_path="./demo_evolution"
    )
    print("[OK] 自演化引擎已初始化")
    print(f"   当前版本: {evolution_engine.current_version}")
    
    # 模拟多场游戏并生成自我批评
    print("\n[2] 模拟多场游戏并生成自我批评...")
    
    gm = GameManager()
    
    for game_num in range(3):
        print(f"\n  游戏 {game_num + 1}:")
        
        # 创建游戏
        config = GameConfig(
            name=f"演化局_{game_num + 1}",
            total_players=6,
            werewolf_count=2,
            seer_count=1,
            witch_count=1,
            villager_count=2
        )
        
        game_id = gm.create_game(config)
        state = gm.get_game(game_id)
        
        # 运行游戏
        max_steps = 50
        step_count = 0
        while not state.is_game_over and step_count < max_steps:
            gm.step(game_id)
            step_count += 1
        
        # 收集评测数据
        metrics_collector = MetricsCollector()
        evaluation = metrics_collector.collect_from_game(state)
        
        # 获取agent_0的指标
        if 0 in evaluation.player_metrics:
            agent_metrics = evaluation.player_metrics[0].to_dict()
            
            # 生成自我批评
            critique = evolution_engine.generate_self_critique(state, agent_metrics)
            
            print(f"    - 得分: {critique.overall_score}")
            print(f"    - 优势: {', '.join(critique.strengths) if critique.strengths else '无'}")
            print(f"    - 劣势: {', '.join(critique.weaknesses) if critique.weaknesses else '无'}")
            
            # 触发演化
            if critique.overall_score < 70 or len(critique.weaknesses) >= 2:
                record = evolution_engine.evolve(critique)
                if record:
                    print(f"    - 演化: {record.from_version} -> {record.to_version}")
                    print(f"    - 类型: {record.evolution_type}")
    
    print("\n[OK] 自我批评和演化完成")
    
    # 显示演化报告
    print_separator("[3] 自演化报告")
    report = evolution_engine.get_evolution_report()
    print(report)
    
    # 显示演化历史
    print("\n【演化历史详情】")
    for i, record in enumerate(evolution_engine.evolution_history, 1):
        print(f"\n  {i}. 版本 {record.from_version} -> {record.to_version}")
        print(f"     时间: {record.timestamp}")
        print(f"     类型: {record.evolution_type}")
        print(f"     原因: {record.trigger_reason}")
        print(f"     变更: {record.changes}")

def main():
    print("""
============================================================
     AI 狼人杀 - 评测与自演化系统演示
                                                              
  方向 B: 多维评测与复盘 Leaderboard                          
  方向 A: 通用 Agent 自演化系统                               
============================================================
    """)
    
    try:
        # 演示评测系统
        demo_evaluation_system()
        
        # 演示自演化系统
        demo_self_evolution()
        
        print_separator("演示完成")
        print("\n[OK] 所有演示已完成！")
        print("\n生成的文件:")
        print("  - demo_leaderboard.json (排行榜数据)")
        print("  - demo_leaderboard_report.json (排行榜报告)")
        print("  - demo_metrics.json (评测数据)")
        print("  - demo_evolution/agent_0_evolution.json (演化数据)")
        
    except Exception as e:
        print(f"\n[ERROR] 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
