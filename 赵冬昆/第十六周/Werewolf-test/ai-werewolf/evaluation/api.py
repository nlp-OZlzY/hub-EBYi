"""
评测系统 API
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from .metrics import MetricsCollector, GameEvaluation
from .leaderboard import LeaderboardManager
from .replay_analyzer import ReplayAnalyzer

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

metrics_collector = MetricsCollector()
leaderboard_manager = LeaderboardManager()
replay_analyzer = ReplayAnalyzer()

class GameResultRequest(BaseModel):
    game_id: str
    game_state: Dict[str, Any]
    metrics: Dict[str, Any]

class AgentRegistrationRequest(BaseModel):
    agent_id: str
    version: str
    model_name: str
    prompt_version: str = "1.0"
    strategy_version: str = "1.0"

class LeaderboardQuery(BaseModel):
    role: Optional[str] = None
    min_games: int = 5
    model_filter: Optional[str] = None

@router.post("/record_game")
async def record_game(result: GameResultRequest):
    """记录游戏结果"""
    try:
        evaluation = metrics_collector.collect_from_game(result.game_state)
        
        # 记录到Leaderboard
        for player_id, player_metrics in evaluation.player_metrics.items():
            leaderboard_manager.record_game_result(
                agent_id=f"player_{player_id}",
                game_result={
                    "is_winner": player_metrics.is_winner,
                    "is_mvp": player_metrics.is_mvp,
                    "score": player_metrics.overall_score,
                    "survival_rounds": player_metrics.survival_rounds,
                    "role": player_metrics.role
                }
            )
        
        return {
            "success": True,
            "evaluation_id": evaluation.game_id,
            "player_scores": {
                str(pid): m.overall_score 
                for pid, m in evaluation.player_metrics.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/leaderboard")
async def get_leaderboard(
    role: Optional[str] = None,
    min_games: int = 5,
    model_filter: Optional[str] = None
):
    """获取排行榜"""
    try:
        leaderboard = leaderboard_manager.get_leaderboard(
            role=role,
            min_games=min_games,
            model_filter=model_filter
        )
        return {
            "success": True,
            "total_agents": len(leaderboard),
            "leaderboard": leaderboard
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/register_agent")
async def register_agent(request: AgentRegistrationRequest):
    """注册Agent"""
    try:
        leaderboard_manager.register_agent(
            agent_id=request.agent_id,
            version=request.version,
            model_name=request.model_name,
            prompt_version=request.prompt_version,
            strategy_version=request.strategy_version
        )
        return {"success": True, "message": "Agent registered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agent_stats/{agent_id}")
async def get_agent_stats(agent_id: str):
    """获取Agent统计"""
    try:
        stats = leaderboard_manager.get_agent_stats(agent_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Agent not found")
        return {"success": True, "stats": stats}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze_replay")
async def analyze_replay(game_state: Dict[str, Any]):
    """分析游戏复盘"""
    try:
        analysis = replay_analyzer.analyze_game(game_state)
        narrative = replay_analyzer.generate_narrative(game_state)
        
        return {
            "success": True,
            "analysis": analysis,
            "narrative": narrative
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/game_report/{game_id}")
async def get_game_report(game_id: str):
    """获取游戏复盘报告"""
    try:
        report = metrics_collector.generate_report(game_id)
        return {
            "success": True,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/overall_report")
async def get_overall_report():
    """获取总体报告"""
    try:
        report = metrics_collector.generate_report()
        return {
            "success": True,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/top_performers")
async def get_top_performers(category: str = "avg_score", limit: int = 10):
    """获取顶尖表现者"""
    try:
        top = leaderboard_manager.get_top_performers(
            category=category,
            limit=limit
        )
        return {
            "success": True,
            "category": category,
            "top_performers": top
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare_agents")
async def compare_agents(agent_ids: List[str]):
    """对比多个Agent"""
    try:
        comparison = leaderboard_manager.compare_agents(agent_ids)
        return {
            "success": True,
            "comparison": comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
