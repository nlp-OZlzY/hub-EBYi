"""游戏引擎（裁判）

狼人杀游戏的核心控制器，负责：
1. 初始化游戏：分配角色、创建 AI 代理
2. 推进游戏流程：夜晚（狼人杀人→预言家查验→女巫用药）→ 白天（宣布死亡→发言→投票）
3. 收集玩家决策：调用 PlayerAgent 的 LLM 决策
4. 执行游戏规则：死亡判定、胜负判定、猎人开枪
5. 记录游戏对话：所有行动和发言记录到 GameState
6. 生成玩家总结：游戏结束后调用 SummaryAgent 生成经验
"""

import asyncio
from typing import List, Dict, Any, Optional

from engine.state import GameState
from engine.phase import GamePhase, TurnOrder
from engine.player import Player
from roles.base import RoleType
from roles.werewolf import Werewolf
from roles.seer import Seer
from roles.witch import Witch
from roles.hunter import Hunter
from roles.villager import Villager
from agent.player_agent import PlayerAgent, create_player_agent, create_judge_agent
from agent.summary_agent import SummaryAgent
from memory.experience import save_experience


class GameEngine:
    """游戏引擎（裁判）

    负责：
    1. 初始化游戏（分配角色、创建玩家代理）
    2. 推进游戏流程（白天/夜晚切换）
    3. 收集玩家决策
    4. 执行游戏规则（死亡判定、胜负判定）
    5. 管理游戏状态
    6. 记录游戏对话
    """

    def __init__(self, player_names: List[str], logger=None):
        """初始化游戏引擎

        Args:
            player_names: 玩家名称列表，长度应为6
            logger: 可选的日志记录器，用于记录游戏过程
        """
        self.player_names = player_names
        self.game_state = GameState()
        self.judge_agent = create_judge_agent()
        self.player_agents: Dict[int, PlayerAgent] = {}
        self._is_running = False
        self._controller = None  # 游戏控制器
        self.logger = logger  # 游戏日志记录器
        self.death_records: List[Dict[str, Any]] = []  # 死亡记录
        self._night_death_causes: Dict[int, str] = {}  # 夜晚死亡原因追踪
        self._step_index = 0  # 逐步执行模式下的阶段索引
        self.game_id = None  # 游戏ID（API使用）
        self._summaries_done = False  # 总结是否已完成
        self._summary_task: Optional[asyncio.Task] = None  # 后台总结任务
        self._summary_agent = SummaryAgent()  # 总结代理
        self.summaries: List[Dict] = []  # 本局游戏生成的玩家总结（API展示用）
        self._night_actions_done = set()  # 记录本夜已执行的阶段，防止重复执行

    def _log(self, level: str, msg: str):
        """内部日志方法"""
        if self.logger:
            if level == "log_action":
                self.logger.log_action_event(msg)
            elif level == "log_speech":
                self.logger.log_speech_event(msg)
            elif level == "log_vote":
                self.logger.log_vote_event(msg)
            elif level == "log_death":
                self.logger.log_death_event(msg)
            elif level == "log_night_action":
                self.logger.log_night_action_event(msg)
            elif level == "log_event":
                self.logger.log_event(msg)
            else:
                getattr(self.logger, level.lower())(msg)

    async def initialize(self, role_assignment: Dict[int, str], player_styles: Dict[int, str] = None):
        """初始化游戏

        Args:
            role_assignment: 角色分配字典，key是player_id，value是角色类型
            player_styles: 玩家决策风格字典，可选
        """
        if player_styles is None:
            player_styles = {}
        # 创建玩家和角色
        for player_id, name in enumerate(self.player_names):
            role_type = role_assignment.get(player_id, "villager")
            role = self._create_role(role_type, player_id)
            player = Player(player_id=player_id, role=role, name=name)

            # 创建玩家代理
            style = player_styles.get(player_id, "balanced")
            private_context = player.role.get_private_context()
            agent = create_player_agent(
                player_id=player_id,
                role_name=role.name,
                private_context=private_context,
                camp=role.camp.value,
                decision_style=style,
                role_type=role.role_type.value,
            )
            player.agent = agent
            self.player_agents[player_id] = agent

            self.game_state.players.append(player)

        # 设置发言顺序（警长决定或默认顺序）
        self.game_state.set_speaker_order([p.player_id for p in self.game_state.players])

        print(f"游戏初始化完成，{len(self.game_state.players)} 名玩家已就位。")

    def _create_role(self, role_type: str, player_id: int):
        """创建角色实例

        Args:
            role_type: 角色类型
            player_id: 玩家ID

        Returns:
            角色实例
        """
        role_map = {
            "werewolf": Werewolf,
            "seer": Seer,
            "witch": Witch,
            "hunter": Hunter,
            "villager": Villager,
        }
        role_class = role_map.get(role_type, Villager)
        return role_class(player_id=player_id)

    async def start(self, controller=None):
        """开始游戏

        Args:
            controller: 可选的游戏控制器，用于手动控制游戏流程
        """
        self._is_running = True
        self._controller = controller
        print("=" * 50)
        print("狼人杀游戏开始！")
        print("=" * 50)

        # 游戏循环
        while self._is_running and not self.game_state.is_game_over():
            # 检查控制器的状态
            if self._controller:
                action = await self._controller.wait_if_needed(
                    self.game_state.day_number, "day_start"
                )
                if action == "stop":
                    self._is_running = False
                    break
                elif action == "skip_to_end":
                    # 直接跳到胜负判定
                    break

            # 夜晚阶段
            await self._night_phase()

            # 检查游戏是否结束
            if self.game_state.is_game_over():
                break

            # 白天阶段
            await self._day_phase()

            # 增加天数
            self.game_state.day_number += 1

        # 游戏结束
        await self._end_game()

    async def step(self) -> Dict[str, Any]:
        """逐步执行一个游戏阶段，返回结构化数据

        阶段索引（_step_index）对应的游戏流程：
            0 → 狼人杀人    1 → 预言家查验    2 → 女巫用药
            3 → 夜晚结果宣布  4 → 白天开始      5 → 公开演讲
            6 → 投票环节    7 → 一天结束      8 → 总结阶段
           -1 → 游戏结束

        每次调用推进一个阶段，返回该阶段的结构化结果，供前端/API 逐步展示。
        """
        if not self.game_state.players:
            raise RuntimeError("Game not initialized. Call initialize() first.")

        if self._step_index == -1 or (self.game_state.is_game_over() and self._step_index not in (-1, 7, 8)):
            winner = self.game_state.get_winner()
            # 游戏在 day_end（index 7）之前提前结束，标记下一步进入总结阶段
            if self.game_state.is_game_over() and self._step_index not in (-1, 7, 8) and not self._summaries_done:
                self._step_index = 8
            return {
                "phase": "game_over",
                "day_number": self.game_state.day_number,
                "step_data": {},
                "players": [p.to_dict() for p in self.game_state.players],
                "dialogues": [],
                "deaths": list(self.death_records),
                "is_game_over": True,
                "winner": winner,
            }

        dialog_len_before = len(self.game_state.dialogues)
        death_len_before = len(self.death_records)
        step_data: Dict[str, Any] = {}
        phase_name = ""

        if self._step_index == 0:
            # 狼人杀人（新夜晚开始）
            self.game_state.phase = GamePhase.NIGHT_WOLF
            phase_name = "night_wolf"
            # 每次进入夜晚都清除旧死亡数据（防止重复宣布）
            self.game_state.clear_night_deaths()
            self._night_death_causes.clear()
            if "wolf" not in self._night_actions_done:
                self._night_actions_done.clear()
                self._night_actions_done.add("wolf")
                print(f"\n{'='*30} 第 {self.game_state.day_number} 夜 {'='*30}")
                self._log("info", f"=== 第 {self.game_state.day_number} 夜 ===")
                self._log("debug", f"阶段: {GamePhase.NIGHT_WOLF.value}")
                await self._wolf_kill()
            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            wolf_votes = [
                {"player_id": d["player_id"], "target": d.get("target"), "reasoning": d.get("reasoning")}
                for d in new_dialogues if d.get("action") == "wolf_vote"
            ]
            step_data = {
                "wolf_votes": wolf_votes,
                "final_target": self.game_state.night_deaths[-1] if self.game_state.night_deaths else None,
            }

        elif self._step_index == 1:
            # 预言家查验（每夜只执行一次）
            self.game_state.phase = GamePhase.NIGHT_SEER
            phase_name = "night_seer"
            self._log("debug", f"阶段: {GamePhase.NIGHT_SEER.value}")
            if "seer" not in self._night_actions_done:
                self._night_actions_done.add("seer")
                await self._seer_check()
            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            seer_data = {}
            for d in new_dialogues:
                if d.get("action") == "seer_check":
                    seer_data = {
                        "seer_id": d["player_id"],
                        "target": d.get("target"),
                        "result": d.get("result"),
                        "reasoning": d.get("reasoning"),
                    }
            step_data = seer_data

        elif self._step_index == 2:
            # 女巫用药（每夜只执行一次）
            self.game_state.phase = GamePhase.NIGHT_WITCH
            phase_name = "night_witch"
            self._log("debug", f"阶段: {GamePhase.NIGHT_WITCH.value}")
            if "witch" not in self._night_actions_done:
                self._night_actions_done.add("witch")
                await self._witch_action()
            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            witch_data = {}
            for d in new_dialogues:
                if d.get("action") in ("heal", "poison"):
                    witch_data = {
                        "action": d["action"],
                        "target": d.get("target"),
                    }
            step_data = witch_data

        elif self._step_index == 3:
            # 夜晚结果宣布 + 猎人开枪
            phase_name = "night_result"
            await self._announce_night_deaths()
            await self._handle_hunter_death(list(self.game_state.night_deaths))
            new_deaths = self.death_records[death_len_before:]
            step_data = {
                "night_deaths": list(self.game_state.night_deaths),
                "deaths": new_deaths,
            }

        elif self._step_index == 4:
            # 白天开始
            self.game_state.phase = GamePhase.DAY_START
            phase_name = "day_start"
            print(f"\n{'='*30} 第 {self.game_state.day_number} 天 {'='*30}")
            self._log("info", f"=== 第 {self.game_state.day_number} 天 ===")
            await self._announce_day_start()
            step_data = {}

        elif self._step_index == 5:
            # 公开演讲
            self.game_state.phase = GamePhase.SPEECH
            phase_name = "speech"
            await self._public_speeches()
            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            speeches = [
                {"player_id": d["player_id"], "player_name": d.get("player_name"), "content": d.get("content")}
                for d in new_dialogues if d.get("action") == "speech"
            ]
            step_data = {"speeches": speeches}

        elif self._step_index == 6:
            # 投票环节
            self.game_state.phase = GamePhase.VOTE
            phase_name = "vote"
            await self._vote()
            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            votes = {}
            for d in new_dialogues:
                if d.get("action") == "vote":
                    votes[d["player_id"]] = d.get("target")
            new_deaths = self.death_records[death_len_before:]
            eliminated = None
            for death in new_deaths:
                if death.get("cause") in ("vote", "shoot"):
                    eliminated = death["player_id"]
            step_data = {
                "votes": votes,
                "eliminated": eliminated,
                "is_tie": getattr(self, "_last_vote_tie", False),
                "tied_players": getattr(self, "_last_vote_tied_players", []),
            }

        elif self._step_index == 7:
            # 一天结束：检查胜负 + 天数+1
            phase_name = "day_end"
            game_over = self.game_state.is_game_over()
            winner = self.game_state.get_winner()
            if game_over:
                self._is_running = False
                self._step_index = 8  # 进入总结阶段
            else:
                self.game_state.day_number += 1
                self._step_index = 0  # 回到夜晚循环

            new_dialogues = self.game_state.dialogues[dialog_len_before:]
            new_deaths = self.death_records[death_len_before:]
            step_data = {
                "game_over": game_over,
                "winner": winner,
                "next_day": self.game_state.day_number if not game_over else None,
            }
            return {
                "phase": phase_name,
                "day_number": self.game_state.day_number,
                "step_data": step_data,
                "players": [p.to_dict() for p in self.game_state.players],
                "dialogues": new_dialogues,
                "deaths": new_deaths,
                "is_game_over": game_over,
                "winner": winner,
            }

        elif self._step_index == 8:
            # 总结阶段：后台并行生成，step 可轮询进度（避免单次 HTTP 阻塞数分钟）
            phase_name = "summary"
            winner = self.game_state.get_winner()
            players_total = len([p for p in self.game_state.players if p.agent])

            if not self._summaries_done:
                if self._summary_task is None:
                    self._summary_task = asyncio.create_task(self._run_summaries())
                elif self._summary_task.done():
                    exc = self._summary_task.exception()
                    if exc is not None:
                        print(f"[总结] 后台任务异常: {exc}")
                        self._summary_task = asyncio.create_task(self._run_summaries())

            if self._summaries_done:
                self._step_index = -1
                step_data = {
                    "status": "complete",
                    "summaries_complete": True,
                    "summaries": list(self.summaries),
                    "done": len(self.summaries),
                    "total": players_total,
                }
            else:
                step_data = {
                    "status": "generating",
                    "summaries_complete": False,
                    "done": len(self.summaries),
                    "total": players_total,
                }

            return {
                "phase": phase_name,
                "day_number": self.game_state.day_number,
                "step_data": step_data,
                "players": [p.to_dict() for p in self.game_state.players],
                "dialogues": [],
                "deaths": list(self.death_records),
                "is_game_over": True,
                "winner": winner,
            }

        # 非 day_end / summary 阶段到达这里
        new_dialogues = self.game_state.dialogues[dialog_len_before:]
        new_deaths = self.death_records[death_len_before:]
        self._step_index += 1

        return {
            "phase": phase_name,
            "day_number": self.game_state.day_number,
            "step_data": step_data,
            "players": [p.to_dict() for p in self.game_state.players],
            "dialogues": new_dialogues,
            "deaths": new_deaths,
            "is_game_over": False,
            "winner": None,
        }

    async def _night_phase(self):
        """执行夜晚阶段"""
        day = self.game_state.day_number
        print(f"\n{'='*30} 第 {day} 夜 {'='*30}")
        self._log("info", f"=== 第 {day} 夜 ===")

        # 清除之前的夜晚死亡记录
        self.game_state.clear_night_deaths()
        self._night_death_causes.clear()

        # 获取夜间行动顺序
        night_order = TurnOrder.get_night_order()

        for phase in night_order:
            # 检查是否还有需要执行的夜间阶段
            if self.game_state.is_game_over():
                break

            self.game_state.phase = phase
            print(f"\n[{phase.value}]")
            self._log("debug", f"阶段: {phase.value}")

            if phase == GamePhase.NIGHT_WOLF:
                await self._wolf_kill()
            elif phase == GamePhase.NIGHT_SEER:
                await self._seer_check()
            elif phase == GamePhase.NIGHT_WITCH:
                await self._witch_action()

        # 夜晚结束，宣布死亡
        await self._announce_night_deaths()

        # 处理猎人死亡开枪（在死亡标记之后）
        await self._handle_hunter_death(self.game_state.night_deaths)

    async def _wolf_kill(self):
        """狼人杀人阶段"""
        alive_wolves = [
            p for p in self.game_state.players
            if p.role_type == RoleType.WEREWOLF and p.is_alive
        ]

        if not alive_wolves:
            return

        # 收集狼人决策
        kill_targets = []
        for wolf in alive_wolves:
            if wolf.agent:
                game_state = self.game_state.get_player_private_context(wolf.player_id)
                decision = await wolf.agent.decide_night_action(game_state)
                target = decision.get("target")
                reasoning = decision.get("reasoning", "")
                # 防止狼人自杀
                if target is not None and target != wolf.player_id:
                    kill_targets.append(target)
                elif target is not None:
                    self._log("log_event", f"狼人{wolf.player_id} 试图自杀，被忽略")
                print(f"狼人 {wolf.player_id} 决策：击杀玩家 {target}")
                self._log("log_action", f"狼人{wolf.player_id} 决策: 击杀玩家{target}")
                # 记录对话到 GameState（action 用 wolf_vote 而非 night_kill，区分投票和实际击杀）
                self.game_state.dialogues.append({
                    "day": self.game_state.day_number,
                    "phase": "狼人讨论",
                    "player_id": wolf.player_id,
                    "player_name": wolf.name,
                    "role": wolf.role.name,
                    "action": "wolf_vote",
                    "target": target,
                    "reasoning": reasoning,
                })

        # 多只狼投票选出击杀目标（多数票决）
        if kill_targets:
            # 过滤掉已死亡的无效目标
            alive_ids = {p.player_id for p in self.game_state.get_alive_players()}
            kill_targets = [t for t in kill_targets if t in alive_ids]
            if kill_targets:
                target = self._count_vote(kill_targets)
                print(f"[DEBUG] 狼人投票列表: {kill_targets}, 最终目标: {target}")
                self.game_state.add_night_death(target)
                self._night_death_causes[target] = "night_kill"
                print(f"狼人今晚击杀：玩家 {target}")
                self._log("log_action", f"狼人击杀目标: 玩家{target}")

    async def _seer_check(self):
        """预言家查验阶段"""
        alive_seers = [
            p for p in self.game_state.players
            if p.role_type == RoleType.SEER and p.is_alive
        ]

        for seer in alive_seers:
            if seer.agent:
                game_state = self.game_state.get_player_private_context(seer.player_id)
                decision = await seer.agent.decide_night_action(game_state)
                target = decision.get("target")
                reasoning = decision.get("reasoning", "")
                if target is not None:
                    player = self.game_state.get_player(target)
                    if player:
                        # 返回查验结果（只有预言家知道）
                        result = "wolf" if player.role_type == RoleType.WEREWOLF else "good"
                        print(f"预言家 {seer.player_id} 查验玩家 {target}：{result}")
                        # 将结果存入私有上下文（这里简化处理）
                        seer.role._check_result = {"target": target, "result": result}
                        self._log("log_night_action", f"预言家{seer.player_id} 查验玩家{target}: {result}")
                        # 记录对话
                        self.game_state.dialogues.append({
                            "day": self.game_state.day_number,
                            "phase": "预言家查验",
                            "player_id": seer.player_id,
                            "player_name": seer.name,
                            "role": seer.role.name,
                            "action": "seer_check",
                            "target": target,
                            "result": result,
                            "reasoning": reasoning,
                        })

    async def _witch_action(self):
        """女巫用药阶段"""
        alive_witches = [
            p for p in self.game_state.players
            if p.role_type == RoleType.WITCH and p.is_alive
        ]

        tonight_death = self.game_state.night_deaths[-1] if self.game_state.night_deaths else None

        for witch in alive_witches:
            if witch.agent and (witch.role.has_heal or witch.role.has_poison):
                game_state = self.game_state.get_player_private_context(witch.player_id)
                game_state["tonight_death"] = tonight_death
                decision = await witch.agent.decide_night_action(game_state)

                action = decision.get("action", "")
                target = decision.get("target")

                used_heal = "heal" in action and tonight_death is not None and witch.role.has_heal
                used_poison = "poison" in action and target is not None and witch.role.has_poison and target != witch.player_id

                # 规则：同夜不能同时使用双药（救人药和毒药互斥）
                if used_heal:
                    witch.role.use_heal()
                    # 救人药移除今晚死亡
                    if tonight_death in self.game_state.night_deaths:
                        self.game_state.night_deaths.remove(tonight_death)
                    print(f"女巫 {witch.player_id} 使用救人药，救活了玩家 {tonight_death}")
                    self._log("log_night_action", f"女巫{witch.player_id} 使用救人药")
                elif used_poison:
                    witch.role.use_poison()
                    self.game_state.add_night_death(target)
                    self._night_death_causes[target] = "poison"
                    print(f"女巫 {witch.player_id} 使用毒药，毒死了玩家 {target}")
                    self._log("log_night_action", f"女巫{witch.player_id} 使用毒药，目标: {target}")
                # 记录女巫行动对话
                if used_heal or used_poison:
                    self.game_state.dialogues.append({
                        "day": self.game_state.day_number,
                        "phase": "女巫用药",
                        "player_id": witch.player_id,
                        "player_name": witch.name,
                        "role": witch.role.name,
                        "action": "heal" if used_heal else "poison",
                        "target": tonight_death if used_heal else (target if used_poison else None),
                    })
                elif "poison" in action and target == witch.player_id:
                    self._log("log_event", f"女巫{witch.player_id} 试图毒自己，被忽略")

    async def _handle_hunter_death(self, killed_player_ids: List[int]):
        """处理猎人死亡开枪（在is_alive已设为False后调用）"""
        for player_id in killed_player_ids:
            player = self.game_state.get_player(player_id)
            if player and player.role_type == RoleType.HUNTER and player.role.can_shoot:
                alive_players = self.game_state.get_alive_players()
                if alive_players:
                    # 简化处理：带走第一个存活玩家
                    # TODO: 应由AI代理决策目标
                    target = alive_players[0].player_id
                    player.role.lock_shoot()
                    # 猎人枪杀不计入夜晚死亡展示，直接杀死
                    target_player = self.game_state.get_player(target)
                    if target_player:
                        target_player.role.is_alive = False
                        print(f"猎人 {player_id} 开枪带走了玩家 {target}")
                        self._log("log_death", f"猎人{player_id} 开枪带走玩家{target}")
                        self.death_records.append({
                            "player_id": target,
                            "player_name": target_player.name,
                            "role": target_player.role.name,
                            "cause": "shoot",
                            "day": self.game_state.day_number,
                        })
                        self.game_state.dialogues.append({
                            "day": self.game_state.day_number,
                            "phase": "猎人开枪",
                            "player_id": player_id,
                            "player_name": player.name,
                            "role": player.role.name,
                            "action": "hunter_shot",
                            "target": target,
                            "reasoning": f"猎人{player_id}开枪带走玩家{target}",
                        })

    async def _announce_night_deaths(self):
        """宣布夜晚死亡"""
        if self.game_state.night_deaths:
            for player_id in self.game_state.night_deaths:
                player = self.game_state.get_player(player_id)
                if player:
                    cause = self._night_death_causes.get(player_id, "night_kill")
                    player.kill(cause, self.game_state.to_dict())
                    print(f"玩家 {player_id} ({player.role.name}) 死亡")
                    self._log("log_death", f"玩家{player_id}({player.role.name}) 死亡")
                    # 记录死亡
                    self.death_records.append({
                        "player_id": player_id,
                        "player_name": player.name,
                        "role": player.role.name,
                        "cause": cause,
                        "day": self.game_state.day_number,
                    })
        else:
            print("今晚无人死亡")
            self._log("log_event", "今晚无人死亡")

    async def _day_phase(self):
        """执行白天阶段"""
        print(f"\n{'='*30} 第 {self.game_state.day_number} 天 {'='*30}")
        self._log("info", f"=== 第 {self.game_state.day_number} 天 ===")

        # 阶段1：宣布昨晚死亡
        self.game_state.phase = GamePhase.DAY_START
        await self._announce_day_start()

        # 阶段2：警长选举（简化处理，跳过）
        # self.game_state.phase = GamePhase.ELECTION

        # 阶段3：公开演讲
        self.game_state.phase = GamePhase.SPEECH
        await self._public_speeches()

        # 阶段4：投票
        self.game_state.phase = GamePhase.VOTE
        await self._vote()

    async def _announce_day_start(self):
        """宣布白天开始（不暴露死者角色，符合标准规则）"""
        if self.game_state.night_deaths:
            death_names = [f"玩家{p}" for p in self.game_state.night_deaths]
            print(f"昨晚死亡：{', '.join(death_names)}")
        else:
            print("昨晚是平安夜，无人死亡")

    async def _public_speeches(self):
        """公开演讲阶段"""
        alive_players = self.game_state.get_alive_players()
        print(f"\n公开演讲开始，共 {len(alive_players)} 名存活玩家")
        self._log("log_event", f"公开演讲开始，{len(alive_players)} 名存活玩家")

        for player in alive_players:
            if player.agent:
                game_state = self.game_state.get_player_private_context(player.player_id)
                decision = await player.agent.decide_speech(game_state)
                content = decision.get("content", "")
                print(f"\n{player.name} 发言：")
                print(content)
                self._log("log_speech", f"{player.name}: {content[:100]}...")
                # 记录对话
                self.game_state.dialogues.append({
                    "day": self.game_state.day_number,
                    "phase": "公开演讲",
                    "player_id": player.player_id,
                    "player_name": player.name,
                    "role": player.role.name,
                    "action": "speech",
                    "content": content,
                })

    async def _vote(self):
        """投票阶段"""
        alive_players = self.game_state.get_alive_players()
        votes = {}
        print(f"\n投票环节开始，共 {len(alive_players)} 名存活玩家")
        self._log("log_event", "投票阶段开始")

        for player in alive_players:
            if player.agent:
                game_state = self.game_state.get_player_private_context(player.player_id)
                decision = await player.agent.decide_vote(game_state)
                target = decision.get("target")
                if target is not None and self.game_state.get_player(target) and self.game_state.get_player(target).is_alive:
                    votes[player.player_id] = target
                    self.game_state.add_vote(player.player_id, target)
                    print(f"  {player.name} 投票给 玩家{target}")
                    self._log("log_vote", f"玩家{player.player_id} 投票给 玩家{target}")
                    # 记录对话
                    self.game_state.dialogues.append({
                        "day": self.game_state.day_number,
                        "phase": "投票",
                        "player_id": player.player_id,
                        "player_name": player.name,
                        "role": player.role.name,
                        "action": "vote",
                        "target": target,
                    })
                elif target is not None:
                    self._log("log_event", f"玩家{player.player_id} 投票给已出局玩家{target}，投票无效")

        # 统计票数
        vote_count = {}
        for target in votes.values():
            vote_count[target] = vote_count.get(target, 0) + 1

        self._last_vote_tie = False
        self._last_vote_tied_players = []

        if vote_count:
            max_votes = max(vote_count.values())
            eliminated = [p for p, c in vote_count.items() if c == max_votes]

            if len(eliminated) == 1:
                player = self.game_state.get_player(eliminated[0])
                print(f"\n玩家 {eliminated[0]} ({player.name}) 被投票出局")
                player.kill("vote", self.game_state.to_dict())
                self._log("log_death", f"玩家{eliminated[0]}({player.name}) 被投票出局")
                # 记录死亡
                self.death_records.append({
                    "player_id": eliminated[0],
                    "player_name": player.name,
                    "role": player.role.name,
                    "cause": "vote",
                    "day": self.game_state.day_number,
                })
                # 处理猎人被投票出局后的开枪
                await self._handle_hunter_death([eliminated[0]])
            else:
                self._last_vote_tie = True
                self._last_vote_tied_players = list(eliminated)
                tied_names = [
                    f"玩家{p}({self.game_state.get_player(p).name})"
                    for p in eliminated
                    if self.game_state.get_player(p)
                ]
                print(f"\n平票！得票相同：{', '.join(tied_names)}（各 {max_votes} 票）")
                print("本轮无人出局，进入下一轮")
                self._log("log_event", f"平票：{eliminated}，各{max_votes}票，无人出局")

        self.game_state.reset_vote_record()

    async def _end_game(self):
        """游戏结束"""
        winner = self.game_state.get_winner()
        print(f"\n{'='*50}")
        print("游戏结束！")
        print(f"胜利方：{'善良阵营' if winner == 'good' else '邪恶阵营'}")
        print(f"{'='*50}")

        self._is_running = False

        # 生成总结并保存经验（仅当未通过 step() 执行过）
        if not self._summaries_done:
            await self._run_summaries()
            self._summaries_done = True

    def _build_personal_history(self, player: Player) -> str:
        """将玩家私有视角格式化为总结用文本"""
        player_context = self.game_state.get_player_private_context(player.player_id)
        dialogues = player_context.get("dialogues", [])

        history_lines = []
        for d in dialogues:
            action = d.get("action", "")
            content = d.get("content", "")
            target = d.get("target")

            if action == "wolf_vote":
                history_lines.append(f"[夜晚] 你投票击杀玩家{target}（{d.get('reasoning', '')}）")
            elif action == "seer_check":
                result = "狼人" if d.get("result") == "wolf" else "好人"
                history_lines.append(f"[夜晚] 你查验玩家{target}，结果：{result}")
            elif action == "heal":
                history_lines.append(f"[夜晚] 你使用解药救活了玩家{target}")
            elif action == "poison":
                history_lines.append(f"[夜晚] 你使用毒药毒杀了玩家{target}")
            elif action == "speech":
                history_lines.append(f"[白天] 你发言：{content[:100]}")
            elif action == "vote":
                history_lines.append(f"[白天] 你投票给玩家{target}")
            elif action == "hunter_shot":
                history_lines.append(f"[夜晚] 你开枪带走了玩家{target}")

        for death in self.death_records:
            if death["player_id"] == player.player_id:
                cause_map = {
                    "night_kill": "被狼人杀害",
                    "poison": "被毒杀",
                    "vote": "被投票出局",
                    "shoot": "被枪杀",
                }
                history_lines.append(
                    f"[死亡] 你在第{death['day']}天{cause_map.get(death['cause'], death['cause'])}"
                )

        return "\n".join(history_lines) if history_lines else "无记录"

    async def _run_summaries(self):
        """游戏结束后为每个玩家生成总结并保存经验（并行调用 LLM）"""
        winner = self.game_state.get_winner()
        print(f"\n{'='*30} 玩家总结 {'='*30}")
        self._log("info", "=== 开始生成玩家总结（并行） ===")

        players_with_agent = [p for p in self.game_state.players if p.agent]
        if not players_with_agent:
            return

        async def _generate_one(player: Player):
            personal_history = self._build_personal_history(player)
            summary_data = await self._summary_agent.generate_summary(
                player_name=player.name,
                role_name=player.role.name,
                camp=player.camp.value,
                winner=winner,
                personal_history=personal_history,
            )
            return player, summary_data

        total = len(players_with_agent)
        tasks = [_generate_one(p) for p in players_with_agent]

        for finished in asyncio.as_completed(tasks):
            item = await finished
            if isinstance(item, Exception):
                self._log("error", f"总结生成异常: {item}")
                print(f"  总结生成失败: {item}")
                continue

            player, summary_data = item
            role_type = player.role.role_type.value

            experience = {
                "game_id": self.game_id,
                "player_id": player.player_id,
                "player_name": player.name,
                "camp": player.camp.value,
                "winner": winner,
                "is_winner": player.camp.value == winner,
                **summary_data,
            }
            save_experience(role_type, experience)

            self.summaries.append({
                "player_id": player.player_id,
                "player_name": player.name,
                "role": player.role.name,
                "camp": player.camp.value,
                "role_type": role_type,
                "is_winner": player.camp.value == winner,
                **summary_data,
            })

            print(
                f"  [{len(self.summaries)}/{total}] "
                f"{player.name}({player.role.name}) 总结已保存"
            )
            self._log("info", f"玩家{player.name} 的总结经验已保存")

        self._summaries_done = True
        print(f"{'='*30} 总结完成 {'='*30}")
        self._log("info", "=== 所有玩家总结完成 ===")

        # 保存可读的 Markdown 游戏记录
        self._save_game_log_md(winner)

    def _save_game_log_md(self, winner: str) -> None:
        """将本局游戏总结保存为可读的 Markdown 文件

        文件保存在 game_logs/ 目录，按 game_id 命名。
        """
        import os
        from datetime import datetime

        logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "game_logs")
        os.makedirs(logs_dir, exist_ok=True)

        game_id = self.game_id or "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{game_id}.md"
        filepath = os.path.join(logs_dir, filename)

        winner_text = "善良阵营（好人）" if winner == "good" else "邪恶阵营（狼人）"

        lines = [
            f"# 游戏记录 - {game_id}",
            "",
            f"- **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- **胜方**: {winner_text}",
            f"- **天数**: {self.game_state.day_number}",
            f"- **玩家数**: {len(self.game_state.players)}",
            "",
            "---",
            "",
            "## 玩家总结",
            "",
        ]

        for s in self.summaries:
            result = "胜利" if s.get("is_winner") else "失败"
            camp_text = "好人" if s.get("camp") == "good" else "狼人"
            lines.append(f"### {s.get('player_name')}（{s.get('role')} / {camp_text} / {result}）")
            lines.append("")
            if s.get("summary"):
                lines.append(f"**总结**: {s['summary']}")
                lines.append("")
            if s.get("strategies"):
                lines.append(f"**策略**: {s['strategies']}")
                lines.append("")
            if s.get("mistakes"):
                lines.append(f"**失误**: {s['mistakes']}")
                lines.append("")
            if s.get("lessons"):
                lines.append(f"**建议**: {s['lessons']}")
                lines.append("")
            lines.append("---")
            lines.append("")

        # 附录：死亡记录
        if self.death_records:
            lines.append("## 死亡记录")
            lines.append("")
            lines.append("| 玩家 | 角色 | 死因 | 天数 |")
            lines.append("|------|------|------|------|")
            cause_map = {"night_kill": "被狼人杀害", "poison": "被毒杀", "vote": "被投票出局", "shoot": "被猎人枪杀"}
            for death in self.death_records:
                cause = cause_map.get(death.get("cause", ""), death.get("cause", ""))
                lines.append(f"| {death.get('player_name', '?')} | {death.get('role', '?')} | {cause} | 第{death.get('day', '?')}天 |")
            lines.append("")

        content = "\n".join(lines)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  游戏记录已保存: {filepath}")
        self._log("info", f"游戏记录已保存: {filepath}")

    def _count_vote(self, targets: List[int]) -> int:
        """统计票数，返回最高票的目标"""
        from collections import Counter
        count = Counter(targets)
        return count.most_common(1)[0][0]

    def stop(self):
        """停止游戏"""
        self._is_running = False


async def create_game(player_names: List[str], role_assignment: Dict[int, str]) -> GameEngine:
    """工厂函数：创建并初始化游戏

    Args:
        player_names: 玩家名称列表
        role_assignment: 角色分配

    Returns:
        GameEngine实例
    """
    engine = GameEngine(player_names)
    await engine.initialize(role_assignment)
    return engine


# 标准6人局角色配置
STANDARD_6P_ROLES = {
    0: "werewolf",
    1: "werewolf",
    2: "seer",
    3: "witch",
    4: "hunter",
    5: "villager",
}

# 预定义角色配置
ROLE_CONFIGS = {
    "standard_6": {
        "name": "标准6人局",
        "description": "2狼、1预言家、1女巫、1猎人、1村民",
        "roles": {
            0: "werewolf",
            1: "werewolf",
            2: "seer",
            3: "witch",
            4: "hunter",
            5: "villager",
        }
    },
    "simple_4": {
        "name": "简易4人局",
        "description": "1狼、1预言家、1女巫、1村民",
        "roles": {
            0: "werewolf",
            1: "seer",
            2: "witch",
            3: "villager",
        }
    },
    "big_9": {
        "name": "标准9人局",
        "description": "3狼、1预言家、1女巫、1猎人、2村民、1白痴",
        "roles": {
            0: "werewolf",
            1: "werewolf",
            2: "werewolf",
            3: "seer",
            4: "witch",
            5: "hunter",
            6: "villager",
            7: "villager",
            8: "idiot",  # 白痴（暂未实现）
        }
    },
}


def get_role_config(config_name: str) -> Dict[int, str]:
    """获取角色配置

    Args:
        config_name: 配置名称，如 "standard_6", "simple_4"

    Returns:
        角色分配字典
    """
    if config_name in ROLE_CONFIGS:
        return ROLE_CONFIGS[config_name]["roles"].copy()
    else:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(ROLE_CONFIGS.keys())}")


def shuffle_roles(role_assignment: Dict[int, str]) -> Dict[int, str]:
    """随机打乱角色分配

    Args:
        role_assignment: 原始角色分配

    Returns:
        打乱后的角色分配
    """
    import random
    roles = list(role_assignment.values())
    random.shuffle(roles)
    return {i: r for i, r in enumerate(roles)}


def create_random_roles(num_players: int, wolf_ratio: float = 0.3) -> Dict[int, str]:
    """根据玩家数量和狼人比例随机生成角色

    Args:
        num_players: 玩家数量
        wolf_ratio: 狼人比例，默认0.3

    Returns:
        随机角色分配
    """
    import random

    num_wolves = max(1, int(num_players * wolf_ratio))
    num_gods = max(1, num_players // 6)
    num_villagers = num_players - num_wolves - num_gods

    roles = (
        ["werewolf"] * num_wolves +
        ["seer", "witch", "hunter"][:num_gods] +
        ["villager"] * num_villagers
    )

    # 确保有狼人和好人
    if "werewolf" not in roles:
        roles[0] = "werewolf"
    if not any(r in roles for r in ["seer", "witch", "hunter"]):
        roles[1] = "seer"

    random.shuffle(roles)
    return {i: r for i, r in enumerate(roles)}


async def run_game(player_names: List[str] = None, config_name: str = "standard_6", shuffle: bool = True):
    """运行游戏

    Args:
        player_names: 玩家名称列表，默认使用默认名称
        config_name: 角色配置名称，如 "standard_6", "simple_4", "big_9"
        shuffle: 是否打乱角色分配，默认True
    """
    if player_names is None:
        config = ROLE_CONFIGS.get(config_name, ROLE_CONFIGS["standard_6"])
        player_names = [f"玩家{i}" for i in range(len(config["roles"]))]

    # 获取角色配置
    role_assignment = get_role_config(config_name)

    # 打乱角色（可选）
    if shuffle:
        role_assignment = shuffle_roles(role_assignment)

    print(f"使用配置：{ROLE_CONFIGS[config_name]['name']}")
    print(f"角色分配：{role_assignment}")

    game = await create_game(player_names, role_assignment)
    await game.start()


if __name__ == "__main__":
    asyncio.run(run_game())