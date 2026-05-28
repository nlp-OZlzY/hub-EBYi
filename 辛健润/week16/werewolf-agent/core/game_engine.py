"""对局引擎"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from core.role_system import RoleType, Camp, Role, STANDARD_12_PLAYER_CONFIG
from core.message_bus import MessageBus, MessageType, Message
from core.event_system import EventSystem, Event, EventType
import random


class Phase(Enum):
    """游戏阶段"""
    NIGHT = "night"
    DAY_SPEECH = "day_speech"
    DAY_VOTE = "day_vote"
    DAY_DEATH = "day_death"
    GAME_OVER = "game_over"


@dataclass
class Player:
    """玩家"""
    id: str
    name: str
    role: Role
    alive: bool = True
    voted: bool = False
    # 角色特定状态
    seer_verified: list[str] = field(default_factory=list)  # 预言家验过的人
    witch_heal_used: bool = False
    witch_poison_used: bool = False
    guard_last_target: Optional[str] = None  # 守卫上一晚守的人
    hunter_can_shoot: bool = True           # 猎人是否可以开枪


@dataclass
class NightAction:
    """夜晚行动"""
    player_id: str
    action_type: str   # "kill", "verify", "heal", "poison", "guard"
    target: Optional[str] = None


@dataclass
class GameState:
    """游戏状态"""
    day: int = 1
    phase: Phase = Phase.NIGHT
    players: dict[str, Player] = field(default_factory=dict)
    alive_players: list[str] = field(default_factory=list)
    wolf_players: list[str] = field(default_factory=list)
    night_actions: list[NightAction] = field(default_factory=list)
    death_queue: list[str] = field(default_factory=list)
    last_words: dict[str, str] = field(default_factory=dict)  # 遗言


class GameEngine:
    """对局引擎"""

    def __init__(self, player_names: Optional[list[str]] = None):
        self.message_bus = MessageBus()
        self.event_system = EventSystem()
        self.state: Optional[GameState] = None
        self.player_names = player_names or [
            "玩家1", "玩家2", "玩家3", "玩家4", "玩家5", "玩家6",
            "玩家7", "玩家8", "玩家9", "玩家10", "玩家11", "玩家12"
        ]

    def initialize(self, roles: Optional[list[RoleType]] = None):
        """初始化游戏"""
        if roles is None:
            roles = STANDARD_12_PLAYER_CONFIG

        # 随机分配角色
        random.shuffle(roles)
        random.shuffle(self.player_names)

        self.state = GameState()
        self.state.players = {}
        self.state.alive_players = []

        for i, (player_name, role_type) in enumerate(zip(self.player_names, roles)):
            player_id = f"player_{i+1}"
            role = Role.get_role_config(role_type)
            player = Player(
                id=player_id,
                name=player_name,
                role=role
            )
            self.state.players[player_id] = player
            self.state.alive_players.append(player_id)

            if role.camp == Camp.WOLF:
                self.state.wolf_players.append(player_id)

        # 初始化狼人阵营认知
        self._init_wolf_camp_info()

        # 发布游戏开始事件
        self.event_system.publish(Event(
            type=EventType.GAME_START,
            data={"players": len(self.state.players), "day": 1}
        ))

        return self

    def _init_wolf_camp_info(self):
        """初始化狼人阵营认知"""
        wolves = self.state.wolf_players
        for wolf_id in wolves:
            wolf_player = self.state.players[wolf_id]
            # 狼人知道其他狼人
            wolf_player.role.description += f" 狼人同伴: {', '.join([self.state.players[w].name for w in wolves if w != wolf_id])}"

    def get_player_info(self, player_id: str) -> dict:
        """获取玩家信息"""
        player = self.state.players[player_id]

        return {
            "id": player.id,
            "name": player.name,
            "role": player.role.role_type.value,
            "role_name": player.role.name,
            "camp": player.role.camp.value,
            "abilities": player.role.abilities,
            "alive": player.alive,
            "public_info": self.message_bus.get_public_info(player_id, player.role.role_type, player.role.camp),
            "private_info": self.message_bus.get_private_info(player_id, player.role.role_type, player.role.camp),
        }

    def get_game_status(self) -> dict:
        """获取游戏状态"""
        return {
            "day": self.state.day,
            "phase": self.state.phase.value,
            "alive_players": [
                {"id": p.id, "name": p.name, "role": p.role.name}
                for p in self.state.players.values() if p.alive
            ],
            "dead_players": [
                {"id": p.id, "name": p.name, "role": p.role.name, "last_words": self.state.last_words.get(p.id, "")}
                for p in self.state.players.values() if not p.alive
            ],
            "wolf_count": len([p for p in self.state.players.values() if p.role.camp == Camp.WOLF and p.alive]),
            "good_count": len([p for p in self.state.players.values() if p.role.camp == Camp.GOOD and p.alive]),
        }

    def record_night_action(self, action: NightAction):
        """记录夜晚行动"""
        self.state.night_actions.append(action)

    def resolve_night(self) -> dict:
        """结算夜晚"""
        night_result = {
            "kill_target": None,
            "verify_result": None,
            "heal_used": False,
            "poison_target": None,
            "guard_target": None,
        }

        # 收集夜晚行动
        actions = {a.player_id: a for a in self.state.night_actions}

        # 1. 处理狼人刀人
        wolf_vote = {}
        for wolf_id in self.state.wolf_players:
            if not self.state.players[wolf_id].alive:
                continue
            if wolf_id in actions and actions[wolf_id].action_type == "kill":
                target = actions[wolf_id].target
                if target and self.state.players[target].alive:
                    wolf_vote[target] = wolf_vote.get(target, 0) + 1

        if wolf_vote:
            kill_target = max(wolf_vote.items(), key=lambda x: x[1])[0]
            night_result["kill_target"] = kill_target

        # 2. 女巫救人（优先检测）
        heal_target = None
        if "witch" in actions:
            witch_action = actions["witch"]
            if witch_action.action_type == "heal" and night_result["kill_target"]:
                heal_target = night_result["kill_target"]
                night_result["heal_used"] = True

        # 3. 处理死亡判定
        # 如果女巫用药救了，则不死
        if night_result["kill_target"] and night_result["kill_target"] != heal_target:
            self.state.death_queue.append(night_result["kill_target"])

        # 4. 女巫毒人
        if "witch" in actions:
            witch_action = actions["witch"]
            if witch_action.action_type == "poison":
                night_result["poison_target"] = witch_action.target
                if witch_action.target:
                    self.state.death_queue.append(witch_action.target)

        # 5. 守卫守护
        if "guard" in actions:
            guard_action = actions["guard"]
            night_result["guard_target"] = guard_action.target
            if guard_action.target == night_result["kill_target"]:
                # 守卫守护成功，取消死亡
                if night_result["kill_target"] in self.state.death_queue:
                    self.state.death_queue.remove(night_result["kill_target"])
                    night_result["kill_target"] = None

        # 6. 预言家验人
        for player_id, player in self.state.players.items():
            if player.role.role_type == RoleType.SEER and player.alive:
                if player_id in actions and actions[player_id].action_type == "verify":
                    target = actions[player_id].target
                    if target:
                        target_role = self.state.players[target].role
                        night_result["verify_result"] = {
                            "player": player_id,
                            "target": target,
                            "target_name": self.state.players[target].name,
                            "is_wolf": target_role.camp == Camp.WOLF
                        }
                        player.seer_verified.append(target)

        # 发布夜晚结束事件
        self.event_system.publish(Event(
            type=EventType.NIGHT_END,
            data={"result": night_result, "deaths": self.state.death_queue}
        ))

        return night_result

    def kill_player(self, player_id: str, reason: str, last_words: str = ""):
        """击杀玩家"""
        if player_id not in self.state.players:
            return

        player = self.state.players[player_id]
        if not player.alive:
            return

        player.alive = False
        self.state.alive_players.remove(player_id)
        self.state.last_words[player_id] = last_words

        # 广播死亡信息
        self.message_bus.broadcast(
            sender="system",
            content=f"{player.name} 死亡了",
            msg_type=MessageType.DEATH
        )

        # 发布死亡事件
        self.event_system.publish(Event(
            type=EventType.PLAYER_DEATH,
            data={"player": player_id, "reason": reason}
        ))

    def process_deaths(self) -> list[str]:
        """处理死亡队列"""
        deaths = self.state.death_queue.copy()
        self.state.death_queue.clear()

        for player_id in deaths:
            self.kill_player(player_id, "夜晚死亡")

        return deaths

    def check_game_end(self) -> Optional[str]:
        """检查游戏是否结束，返回获胜阵营"""
        wolf_alive = len([p for p in self.state.players.values()
                         if p.role.camp == Camp.WOLF and p.alive])
        good_alive = len([p for p in self.state.players.values()
                         if p.role.camp == Camp.GOOD and p.alive])

        if wolf_alive == 0:
            return "good"
        elif wolf_alive >= good_alive:
            return "wolf"

        return None

    def record_vote(self, voter: str, target: str):
        """记录投票"""
        self.message_bus.broadcast(
            sender=voter,
            content=f"投票给 {self.state.players[target].name}",
            msg_type=MessageType.VOTE,
        )
        self.state.players[voter].voted = True

        # 存储投票信息
        msg = Message(
            type=MessageType.VOTE,
            sender=voter,
            content=f"投票给 {self.state.players[target].name}",
            recipient=None,
            meta={"target": target, "voter": voter}
        )
        self.message_bus.messages.append(msg)

    def resolve_vote(self) -> Optional[str]:
        """结算投票，返回被投票出局者"""
        votes: dict[str, int] = {}

        for player in self.state.players.values():
            if not player.alive:
                continue

        # 统计票数
        for msg in self.message_bus.messages:
            if msg.type == MessageType.VOTE and msg.meta.get("target"):
                target = msg.meta["target"]
                votes[target] = votes.get(target, 0) + 1

        if not votes:
            return None

        # 找出最高票
        max_votes = max(votes.values())
        max_targets = [t for t, v in votes.items() if v == max_votes]

        if len(max_targets) == 1:
            return max_targets[0]
        else:
            # 平票则不死
            return None

    def set_phase(self, phase: Phase):
        """设置游戏阶段"""
        self.state.phase = phase

        if phase == Phase.NIGHT:
            self.event_system.publish(Event(type=EventType.NIGHT_START, data={"day": self.state.day}))
        elif phase == Phase.DAY_SPEECH:
            self.event_system.publish(Event(type=EventType.DAY_START, data={"day": self.state.day}))
        elif phase == Phase.DAY_VOTE:
            self.event_system.publish(Event(type=EventType.VOTE_START))
        elif phase == Phase.GAME_OVER:
            winner = self.check_game_end()
            self.event_system.publish(Event(
                type=EventType.GAME_OVER,
                data={"winner": winner, "day": self.state.day}
            ))

    def next_day(self):
        """进入下一天"""
        self.state.day += 1
        self.state.night_actions.clear()
        # 重置投票状态
        for player in self.state.players.values():
            player.voted = False

    def get_logs(self) -> list[dict]:
        """获取结构化日志"""
        logs = []

        # 游戏事件日志
        for event in self.event_system.get_event_log():
            logs.append({
                "type": "event",
                "event": event.type.value,
                "timestamp": event.timestamp,
                "data": event.data
            })

        # 消息日志
        for msg in self.message_bus.get_all_messages():
            logs.append({
                "type": "message",
                "message_type": msg.type.value,
                "sender": msg.sender,
                "content": msg.content,
                "timestamp": msg.timestamp if hasattr(msg, 'timestamp') else ""
            })

        return logs