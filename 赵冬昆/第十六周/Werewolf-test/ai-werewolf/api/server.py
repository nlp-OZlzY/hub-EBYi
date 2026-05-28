import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from engine.game_manager import GameManager
from engine.types import GameConfig, GameState
from configs.game_configs import get_config, list_configs

app = FastAPI(title="AI Werewolf Game API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

game_manager = GameManager()

@app.get("/")
async def root():
    return {"message": "AI Werewolf Game API is running"}

@app.post("/games", response_model=Dict[str, Any])
async def create_game(
    config_name: str = "standard_6",
    player_names: Optional[List[str]] = None,
    shuffle: bool = True
):
    config = get_config(config_name)
    game_id = game_manager.create_game(config, player_names, shuffle)
    state = game_manager.get_game(game_id)
    
    return {
        "game_id": game_id,
        "phase": state.phase.value,
        "day_number": state.day_number,
        "alive_count": len(state.get_alive_players()),
        "is_game_over": state.is_game_over
    }

@app.post("/games/{game_id}/step", response_model=Dict[str, Any])
async def step_game(game_id: str):
    state = game_manager.get_game(game_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    
    new_state = game_manager.step(game_id)
    
    return {
        "game_id": new_state.game_id,
        "phase": new_state.phase.value,
        "day_number": new_state.day_number,
        "alive_count": len(new_state.get_alive_players()),
        "is_game_over": new_state.is_game_over,
        "winner": new_state.winner.value if new_state.winner else None,
        "dialogues": new_state.dialogues[-3:] if new_state.dialogues else []
    }

@app.get("/games/{game_id}", response_model=Dict[str, Any])
async def get_game(game_id: str):
    state = game_manager.get_game(game_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    
    players = []
    for p in state.players:
        players.append({
            "player_id": p.player_id,
            "name": p.name,
            "role": p.role.value,
            "is_alive": p.is_alive,
            "is_sheriff": p.is_sheriff
        })
    
    return {
        "game_id": state.game_id,
        "phase": state.phase.value,
        "day_number": state.day_number,
        "players": players,
        "is_game_over": state.is_game_over,
        "winner": state.winner.value if state.winner else None,
        "dialogues": state.dialogues,
        "deaths": state.deaths
    }

@app.get("/games", response_model=List[Dict[str, Any]])
async def list_games():
    return game_manager.list_games()

@app.delete("/games/{game_id}")
async def delete_game(game_id: str):
    game_manager.delete_game(game_id)
    return {"message": f"Game {game_id} deleted successfully"}

@app.get("/configs", response_model=List[Dict[str, Any]])
async def get_configs():
    return list_configs()
