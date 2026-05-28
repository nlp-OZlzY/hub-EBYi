#!/usr/bin/env python3
"""
AI Werewolf Game Example
This script demonstrates how to use the game engine programmatically.
"""
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from engine.game_manager import GameManager
from configs.game_configs import get_config

def run_game():
    print("Wolf AI 狼人杀游戏演示")
    print("=" * 50)
    
    manager = GameManager()
    
    print("创建游戏...")
    config = get_config("standard_6")
    game_id = manager.create_game(config)
    print(f"游戏创建成功！ID: {game_id}")
    print()
    
    state = manager.get_game(game_id)
    print("初始玩家列表:")
    for p in state.players:
        print(f"  {p.player_id}: {p.name} - {p.role.value}")
    print()
    
    print("开始游戏循环...")
    print("-" * 50)
    
    while not state.is_game_over:
        state = manager.step(game_id)
        
        print(f"\n【第{state.day_number}天】{state.phase.value}")
        
        if state.dialogues:
            recent_dialogues = state.dialogues[-3:]
            for d in recent_dialogues:
                if 'message' in d:
                    print(f"  [Speech] {d['message']}")
        
        alive_count = len(state.get_alive_players())
        print(f"  [Alive] {alive_count}")
        
        if state.step_data:
            if 'wolf_votes' in state.step_data:
                target = state.step_data.get('final_target')
                print(f"  [Wolf] 狼人选择刀杀玩家{target}")
            if 'witch_action' in state.step_data:
                action = state.step_data['witch_action'].get('message', '')
                print(f"  [Witch] {action}")
    
    print("\n" + "=" * 50)
    winner = "好人阵营" if state.winner.value == "good" else "狼人阵营"
    print(f"[END] 游戏结束！{winner}获胜！")
    print("=" * 50)

if __name__ == "__main__":
    run_game()
