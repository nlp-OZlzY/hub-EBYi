import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from typing import Optional, Dict, List, Any
from ..prompts.template_loader import load_prompt_template
from ..strategies.base_strategy import BaseStrategy
from ..llm.qwen_client import get_qwen_client, QwenClient
from engine.types import Player, GameState, GameAction, Action, ActionType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class BaseAgent:
    def __init__(self, player: Player, strategy: Optional[BaseStrategy] = None, 
                 use_llm: bool = False, llm_client: Optional[QwenClient] = None):
        self.player = player
        self.strategy = strategy
        self.use_llm = use_llm
        self.llm_client = llm_client
        self.memory = {}
        self.beliefs: Dict[int, float] = {}
        self.prompt_template = load_prompt_template(player.role)
    
    def get_private_context(self, state: GameState) -> Dict[str, Any]:
        context = {
            "player_id": self.player.player_id,
            "player_name": self.player.name,
            "role": self.player.role.value,
            "is_alive": self.player.is_alive,
            "day_number": state.day_number,
            "phase": state.phase.value
        }
        return context
    
    def _build_context_string(self, state: GameState) -> str:
        """构建上下文字符串用于LLM"""
        context = []
        context.append(f"第{state.day_number}天，{state.phase.value}阶段")
        context.append(f"我是{self.player.name}，身份是{self.player.role.value}")
        
        alive = [p.name for p in state.get_alive_players()]
        dead = [p.name for p in state.players if not p.is_alive]
        
        if alive:
            context.append(f"存活玩家: {', '.join(alive)}")
        if dead:
            context.append(f"死亡玩家: {', '.join(dead)}")
        
        # 添加历史对话
        if state.dialogues:
            context.append("\n最近对话:")
            for d in state.dialogues[-3:]:
                speaker = d.get("speaker", "unknown")
                content = d.get("content", "")
                context.append(f"  {speaker}: {content[:30]}...")
        
        return "\n".join(context)
    
    def think(self, state: GameState) -> str:
        if self.use_llm and self.llm_client:
            context_str = self._build_context_string(state)
            thought = self.llm_client.think(self.player.role.value, context_str)
        elif self.strategy:
            context = self.get_private_context(state)
            thought = self.strategy.think(context, self.beliefs)
        else:
            thought = f"当前是第{state.day_number}天，{state.phase.value}阶段。我需要根据局势做出决策。"
        
        logger.debug(f"[AGENT_THINK] player={self.player.player_id}, role={self.player.role.value}, thought={thought[:50]}...")
        return thought
    
    def act(self, state: GameState) -> GameAction:
        inner_monologue = self.think(state)
        
        alive_players = state.get_alive_players()
        targets = [p for p in alive_players if p.player_id != self.player.player_id]
        
        action_type = ActionType.PASS
        target = None
        content = None
        belief_update = {}
        
        if state.phase.value in ["speech"]:
            action_type = ActionType.SPEAK
            content = self.generate_speech(state)
        elif state.phase.value in ["vote"]:
            action_type = ActionType.VOTE
            if targets:
                if self.use_llm and self.llm_client:
                    context_str = self._build_context_string(state)
                    candidate_names = [f"玩家{p.player_id}" for p in targets]
                    vote_result = self.llm_client.vote(self.player.role.value, context_str, candidate_names)
                    try:
                        target = int(vote_result)
                    except ValueError:
                        target = targets[0].player_id
                else:
                    target = targets[0].player_id
                content = f"投票给玩家{target}"
        
        action = Action(
            type=action_type,
            target=target,
            content=content
        )
        
        logger.info(f"[AGENT_ACT] player={self.player.player_id}, role={self.player.role.value}, action_type={action_type.value}, target={target}")
        
        return GameAction(
            inner_monologue=inner_monologue,
            action=action,
            belief_update=belief_update
        )
    
    def generate_speech(self, state: GameState) -> str:
        if self.use_llm and self.llm_client:
            context_str = self._build_context_string(state)
            is_werewolf = self.player.role.value == "werewolf"
            return self.llm_client.speak(self.player.role.value, context_str, is_werewolf)
        elif self.strategy:
            return self.strategy.generate_speech(state, self.player)
        return f"大家好，我是{self.player.name}。目前局势还不太明朗，让我们听听其他人的看法。"
    
    def update_beliefs(self, updates: Dict[int, float]):
        for player_id, delta in updates.items():
            self.beliefs[player_id] = self.beliefs.get(player_id, 0.0) + delta
            self.beliefs[player_id] = max(-1.0, min(1.0, self.beliefs[player_id]))
    
    def get_belief_summary(self) -> Dict[int, str]:
        summary = {}
        for player_id, belief in self.beliefs.items():
            if belief > 0.5:
                summary[player_id] = "信任"
            elif belief < -0.5:
                summary[player_id] = "怀疑"
            else:
                summary[player_id] = "中立"
        return summary
