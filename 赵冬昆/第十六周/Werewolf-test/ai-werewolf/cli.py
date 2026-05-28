#!/usr/bin/env python3
import argparse
import sys
from engine.game_manager import GameManager
from configs.game_configs import get_config, list_configs

def main():
    parser = argparse.ArgumentParser(description="AI Werewolf Game CLI")
    subparsers = parser.add_subparsers(dest='command')
    
    create_parser = subparsers.add_parser('create', help='Create a new game')
    create_parser.add_argument('--config', '-c', default='standard_6', 
                              help='Game configuration name')
    create_parser.add_argument('--players', '-p', nargs='+', 
                              help='Custom player names')
    
    step_parser = subparsers.add_parser('step', help='Advance game by one step')
    step_parser.add_argument('game_id', help='Game ID to advance')
    
    show_parser = subparsers.add_parser('show', help='Show game status')
    show_parser.add_argument('game_id', help='Game ID to show')
    
    list_parser = subparsers.add_parser('list', help='List all games')
    
    run_parser = subparsers.add_parser('run', help='Run a complete game')
    run_parser.add_argument('--config', '-c', default='standard_6',
                           help='Game configuration name')
    
    configs_parser = subparsers.add_parser('configs', help='List available configurations')
    
    args = parser.parse_args()
    
    manager = GameManager()
    
    if args.command == 'create':
        config = get_config(args.config)
        player_names = args.players if args.players else None
        game_id = manager.create_game(config, player_names)
        print(f"Created game: {game_id}")
    
    elif args.command == 'step':
        try:
            state = manager.step(args.game_id)
            print(f"Game {args.game_id} advanced to {state.phase.value}")
            print(f"Day: {state.day_number}")
            print(f"Alive: {len(state.get_alive_players())}")
            if state.dialogues:
                for d in state.dialogues[-2:]:
                    print(f"  - {d.get('message', '')}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
    
    elif args.command == 'show':
        state = manager.get_game(args.game_id)
        if state:
            print(f"Game ID: {state.game_id}")
            print(f"Phase: {state.phase.value}")
            print(f"Day: {state.day_number}")
            print(f"Game Over: {state.is_game_over}")
            if state.winner:
                print(f"Winner: {state.winner.value}")
            print("\nPlayers:")
            for p in state.players:
                status = "ALIVE" if p.is_alive else "DEAD"
                print(f"  {p.player_id}: {p.name} ({p.role.value}) - {status}")
            if state.dialogues:
                print("\nRecent Dialogues:")
                for d in state.dialogues[-5:]:
                    print(f"  [{d['phase']}] {d.get('message', '')}")
        else:
            print(f"Game {args.game_id} not found", file=sys.stderr)
    
    elif args.command == 'list':
        games = manager.list_games()
        if games:
            for game in games:
                print(f"{game['game_id']}: {game['phase']} (Day {game['day_number']})")
        else:
            print("No active games")
    
    elif args.command == 'run':
        config = get_config(args.config)
        game_id = manager.create_game(config)
        print(f"Starting game {game_id} with config: {args.config}")
        print("-" * 50)
        
        state = manager.get_game(game_id)
        while not state.is_game_over:
            state = manager.step(game_id)
            print(f"Phase: {state.phase.value} | Day: {state.day_number}")
            
            if state.dialogues:
                for d in state.dialogues[-3:]:
                    if 'message' in d:
                        print(f"  {d['message']}")
            
            print(f"  Alive: {len(state.get_alive_players())}")
            print()
        
        winner = "好人阵营" if state.winner.value == "good" else "狼人阵营"
        print("=" * 50)
        print(f"Game Over! {winner} wins!")
    
    elif args.command == 'configs':
        configs = list_configs()
        for config in configs:
            print(f"{config['name']}: {config['total_players']}人局")
            print(f"  - 狼人: {config['werewolf_count']}")
            print(f"  - 预言家: {config['seer_count']}")
            print(f"  - 女巫: {config['witch_count']}")
            print(f"  - 猎人: {config['hunter_count']}")
            print(f"  - 村民: {config['villager_count']}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
