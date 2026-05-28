"""后台自演化任务执行模块

提供异步自演化任务的创建和执行能力，供 FastAPI 后台任务调用。
每轮游戏结束后自动触发 SelfReflector 反思并更新 prompt。
支持 SSE 实时推送日志到前端。
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from engine.game_engine import GameEngine, get_role_config, shuffle_roles
from metrics.collector import MetricsCollector
from agent.reflector import SelfReflector
from prompt_store.store import PromptStore
from evolve import get_involved_roles, get_role_logs

ROLE_LABELS = {
    "werewolf": "狼人",
    "seer": "预言家",
    "witch": "女巫",
    "hunter": "猎人",
    "villager": "村民",
}


def _role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role)


# SSE 订阅者：每个 job_id 对应一组 asyncio.Queue
_sse_subscribers: Dict[str, Set[asyncio.Queue]] = {}


def subscribe_sse(job_id: str) -> asyncio.Queue:
    """订阅指定任务的 SSE 日志流"""
    queue: asyncio.Queue = asyncio.Queue()
    if job_id not in _sse_subscribers:
        _sse_subscribers[job_id] = set()
    _sse_subscribers[job_id].add(queue)
    return queue


def unsubscribe_sse(job_id: str, queue: asyncio.Queue) -> None:
    """取消订阅 SSE 日志流"""
    if job_id in _sse_subscribers:
        _sse_subscribers[job_id].discard(queue)
        if not _sse_subscribers[job_id]:
            del _sse_subscribers[job_id]


async def _broadcast(job_id: str, data: dict) -> None:
    """向所有订阅者广播消息"""
    if job_id in _sse_subscribers:
        for queue in _sse_subscribers[job_id]:
            await queue.put(data)


async def _broadcast_status(job: "EvolveJob") -> None:
    """向 SSE 订阅者推送任务进度（轮次、胜负等）"""
    await _broadcast(
        job.job_id,
        {
            "type": "status",
            "current_round": job.current_round,
            "rounds": job.rounds,
            "status": job.status,
            "good_wins": job.good_wins,
            "evil_wins": job.evil_wins,
            "prompt_changes": dict(job.prompt_changes),
        },
    )


@dataclass
class EvolveJob:
    """自演化任务数据类

    记录任务状态、进度、胜负统计和日志，供 API 层查询。
    """
    job_id: str
    rounds: int
    config_name: str
    shuffle: bool = True
    status: str = "pending"
    current_round: int = 0
    logs: List[Dict[str, str]] = field(default_factory=list)
    prompt_changes: Dict[str, int] = field(default_factory=dict)
    good_wins: int = 0
    evil_wins: int = 0
    error: Optional[str] = None

    def log(self, text: str) -> None:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "text": text,
        }
        self.logs.append(entry)
        # 异步广播给 SSE 订阅者（不阻塞）
        asyncio.create_task(_broadcast(self.job_id, entry))

    def notify_status(self) -> None:
        """推送当前任务进度到 SSE（轮次、胜负统计等）"""
        asyncio.create_task(_broadcast_status(self))


evolve_jobs: Dict[str, EvolveJob] = {}


async def run_evolve_job(job: EvolveJob) -> None:
    """执行自演化循环（供 API 后台任务调用）"""
    job.status = "running"
    job.log("自演化任务启动")
    job.notify_status()

    store = PromptStore()
    reflector = SelfReflector()

    try:
        for round_num in range(1, job.rounds + 1):
            job.current_round = round_num
            job.notify_status()
            job.log(f"—— 第 {round_num}/{job.rounds} 轮 ——")

            role_assignment = get_role_config(job.config_name)
            if job.shuffle:
                role_assignment = shuffle_roles(role_assignment)

            player_names = [f"玩家{i}" for i in range(len(role_assignment))]
            engine = GameEngine(player_names)
            await engine.initialize(role_assignment)

            while not engine.game_state.is_game_over():
                await engine.step()

            winner = engine.game_state.get_winner()
            if winner == "good":
                job.good_wins += 1
            else:
                job.evil_wins += 1

            winner_text = "好人" if winner == "good" else "狼人"
            job.log(f"胜方: {winner_text}")
            job.notify_status()

            metrics = MetricsCollector.collect(engine)
            involved_roles = get_involved_roles(engine)
            changed = []

            for role in involved_roles:
                role_metrics = metrics.get(role)
                if not role_metrics:
                    continue

                current_prompt = store.read_prompt(role)
                if not current_prompt:
                    continue

                game_logs = get_role_logs(role, engine)
                job.log(f"反思 {_role_label(role)} Agent...")
                new_prompt = await reflector.reflect(
                    role=role,
                    current_prompt=current_prompt,
                    metrics=role_metrics["metrics"],
                    game_logs=game_logs,
                )

                if new_prompt.strip() != current_prompt.strip():
                    store.save_version(role, current_prompt)
                    store.write_prompt(role, new_prompt)
                    changed.append(role)
                    job.prompt_changes[role] = job.prompt_changes.get(role, 0) + 1
                    job.log(f"  {_role_label(role)} Agent 已更新")
                else:
                    job.log(f"  {_role_label(role)} Agent 无变化")

            changed_labels = [_role_label(r) for r in changed]
            job.log(f"本轮 Agent 更新: {', '.join(changed_labels) if changed_labels else '无'}")
            job.notify_status()

        job.log("演化完成")
        job.status = "completed"
        job.notify_status()
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.log(f"错误: {e}")
        job.notify_status()


def create_evolve_job(rounds: int, config_name: str, shuffle: bool) -> EvolveJob:
    job_id = str(uuid.uuid4())[:8]
    job = EvolveJob(
        job_id=job_id,
        rounds=rounds,
        config_name=config_name,
        shuffle=shuffle,
    )
    evolve_jobs[job_id] = job
    return job
