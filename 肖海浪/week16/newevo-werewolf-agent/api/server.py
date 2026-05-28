"""FastAPI 服务端

提供三大功能：
1. 游戏观战 API：创建游戏、逐步推进、查看状态
2. 自演化 API：启动后台演化任务、查看进度和日志
3. Prompt 浏览 API：查看各角色当前策略和演化历史
4. 前端静态资源托管（Vue 3 构建产物）
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from api.models import (
    CreateGameRequest,
    StepResponse,
    GameStatusResponse,
    GameSummaryResponse,
    ConfigInfo,
    StartEvolveRequest,
    EvolveJobResponse,
    EvolveLogEntry,
    PromptInfo,
)
from api.evolve_service import evolve_jobs, create_evolve_job, run_evolve_job, subscribe_sse, unsubscribe_sse
from engine.game_engine import (
    GameEngine,
    get_role_config,
    shuffle_roles,
    ROLE_CONFIGS,
)
from prompt_store.store import PromptStore

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="Self-Evolving Werewolf API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

games: Dict[str, GameEngine] = {}

# 暂未实现白痴角色，前端不展示 big_9
PLAYABLE_CONFIGS = {k: v for k, v in ROLE_CONFIGS.items() if k != "big_9"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


# ── 游戏观战 API ──────────────────────────────────────────────


@app.post("/games")
async def create_game(req: CreateGameRequest) -> GameSummaryResponse:
    config_name = req.config_name
    if config_name not in PLAYABLE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown config: {config_name}")

    role_assignment = get_role_config(config_name)
    player_count = len(role_assignment)

    player_names = req.player_names
    if player_names is None:
        player_names = [f"玩家{i}" for i in range(player_count)]
    if len(player_names) != player_count:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {player_count} players, got {len(player_names)}",
        )

    if req.shuffle:
        role_assignment = shuffle_roles(role_assignment)

    engine = GameEngine(player_names)
    await engine.initialize(role_assignment, req.player_styles)

    game_id = str(uuid.uuid4())[:8]
    engine.game_id = game_id
    games[game_id] = engine

    return GameSummaryResponse(
        game_id=game_id,
        phase=engine.game_state.phase.value,
        day_number=engine.game_state.day_number,
        alive_count=len(engine.game_state.get_alive_players()),
        is_game_over=False,
    )


@app.post("/games/{game_id}/step")
async def step_game(game_id: str) -> StepResponse:
    engine = games.get(game_id)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    result = await engine.step()
    summaries = result.get("step_data", {}).get("summaries", []) or []

    return StepResponse(
        phase=result["phase"],
        day_number=result["day_number"],
        step_data=result["step_data"],
        players=result["players"],
        dialogues=result["dialogues"],
        deaths=result["deaths"],
        is_game_over=result["is_game_over"],
        winner=result.get("winner"),
        summaries=summaries,
    )


@app.get("/games/{game_id}")
async def get_game_status(game_id: str) -> GameStatusResponse:
    engine = games.get(game_id)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    players_with_agent = len([p for p in engine.game_state.players if p.agent])
    if engine._summaries_done:
        summary_status = "complete"
    elif engine._summary_task is not None and not engine._summary_task.done():
        summary_status = "generating"
    elif engine._summary_task is not None and engine._summary_task.done() and not engine._summaries_done:
        summary_status = "generating"
    else:
        summary_status = "idle"

    return GameStatusResponse(
        game_id=game_id,
        phase=engine.game_state.phase.value,
        day_number=engine.game_state.day_number,
        players=[p.to_dict() for p in engine.game_state.players],
        dialogues=list(engine.game_state.dialogues),
        death_records=list(engine.death_records),
        winner=engine.game_state.get_winner(),
        is_game_over=engine.game_state.is_game_over(),
        summaries=list(engine.summaries),
        summary_status=summary_status,
        summary_done=len(engine.summaries),
        summary_total=players_with_agent,
    )


@app.get("/games/{game_id}/summaries")
async def get_game_summaries(game_id: str) -> Dict[str, Any]:
    engine = games.get(game_id)
    if not engine:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    players_with_agent = len([p for p in engine.game_state.players if p.agent])
    return {
        "game_id": game_id,
        "summaries": list(engine.summaries),
        "summary_status": "complete" if engine._summaries_done else "generating",
        "summary_done": len(engine.summaries),
        "summary_total": players_with_agent,
    }


@app.get("/games")
async def list_games() -> Dict[str, GameSummaryResponse]:
    return {
        gid: GameSummaryResponse(
            game_id=gid,
            phase=engine.game_state.phase.value,
            day_number=engine.game_state.day_number,
            alive_count=len(engine.game_state.get_alive_players()),
            is_game_over=engine.game_state.is_game_over(),
        )
        for gid, engine in games.items()
    }


@app.delete("/games/{game_id}")
async def delete_game(game_id: str) -> Dict[str, str]:
    if game_id not in games:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")
    del games[game_id]
    return {"message": f"Game {game_id} deleted"}


@app.get("/configs")
async def list_configs() -> Dict[str, ConfigInfo]:
    return {
        name: ConfigInfo(
            name=cfg["name"],
            description=cfg["description"],
            player_count=len(cfg["roles"]),
        )
        for name, cfg in PLAYABLE_CONFIGS.items()
    }


# ── 自演化 API ────────────────────────────────────────────────


@app.post("/evolve/jobs")
async def start_evolve(req: StartEvolveRequest, background_tasks: BackgroundTasks) -> EvolveJobResponse:
    if req.config_name not in PLAYABLE_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown config: {req.config_name}")

    job = create_evolve_job(req.rounds, req.config_name, req.shuffle)
    background_tasks.add_task(run_evolve_job, job)
    return _job_to_response(job)


@app.get("/evolve/jobs/{job_id}")
async def get_evolve_job(job_id: str) -> EvolveJobResponse:
    job = evolve_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return _job_to_response(job)


@app.get("/evolve/jobs")
async def list_evolve_jobs() -> Dict[str, EvolveJobResponse]:
    return {jid: _job_to_response(job) for jid, job in evolve_jobs.items()}


def _job_to_response(job) -> EvolveJobResponse:
    return EvolveJobResponse(
        job_id=job.job_id,
        status=job.status,
        rounds=job.rounds,
        config_name=job.config_name,
        current_round=job.current_round,
        logs=[EvolveLogEntry(**e) for e in job.logs],
        prompt_changes=dict(job.prompt_changes),
        good_wins=job.good_wins,
        evil_wins=job.evil_wins,
        error=job.error,
    )


# ── Prompt 浏览 API ───────────────────────────────────────────


@app.get("/prompts/roles")
async def list_role_prompts() -> Dict[str, PromptInfo]:
    store = PromptStore()
    roles = ["werewolf", "seer", "witch", "hunter", "villager"]
    result = {}
    for role in roles:
        # 优先读 Agent.md
        agent_path = os.path.join("prompts", "agents", f"{role}_agent.md")
        if os.path.exists(agent_path):
            with open(agent_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = store.read_prompt(role)
        versions = store.list_versions(role)
        result[role] = PromptInfo(role=role, content=content, versions=versions)
    return result


@app.get("/prompts/roles/{role}")
async def get_role_prompt(role: str) -> PromptInfo:
    store = PromptStore()
    agent_path = os.path.join("prompts", "agents", f"{role}_agent.md")
    if os.path.exists(agent_path):
        with open(agent_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = store.read_prompt(role)
    if not content:
        raise HTTPException(status_code=404, detail=f"No prompt for role: {role}")
    return PromptInfo(role=role, content=content, versions=store.list_versions(role))


@app.get("/prompts/roles/{role}/versions/{version}")
async def get_role_prompt_version(role: str, version: str) -> Dict[str, str]:
    """获取指定角色的某个历史版本 prompt"""
    store = PromptStore()
    import glob as glob_mod
    role_dir = os.path.join(store.versions_dir, role)
    filepath = os.path.join(role_dir, f"{version}.md")
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Version {version} not found for role {role}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return {"role": role, "version": version, "content": content}


# ── SSE 实时日志推送 ───────────────────────────────────────────


@app.get("/evolve/jobs/{job_id}/stream")
async def stream_evolve_logs(job_id: str):
    """SSE 端点：实时推送演化任务日志"""

    async def event_generator():
        queue = subscribe_sse(job_id)
        try:
            # 只推送新日志（已有日志通过 startEvolve API 返回，不需要重复发送）
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {__import__('json').dumps(data, ensure_ascii=False)}\n\n"
                    # 如果任务结束，发送完成信号并关闭
                    job = evolve_jobs.get(job_id)
                    if job and job.status in ("completed", "failed"):
                        yield f"data: {__import__('json').dumps({'time': '', 'text': '__DONE__'}, ensure_ascii=False)}\n\n"
                        break
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        finally:
            unsubscribe_sse(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 前端静态资源 ──────────────────────────────────────────────


if FRONTEND_DIST.is_dir():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    async def serve_frontend():
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return FileResponse(str(index))
        return {"message": "Frontend not built. Run: cd frontend && npm install && npm run build"}

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        if full_path.startswith(("games", "configs", "evolve", "prompts", "api")):
            raise HTTPException(status_code=404)
        index = FRONTEND_DIST / "index.html"
        if index.is_file():
            return FileResponse(str(index))
        raise HTTPException(status_code=404)
