"""API 请求/响应 Pydantic 模型

定义 FastAPI 接口的数据结构，包括游戏创建、步进、状态查询、
自演化任务和 Prompt 浏览等接口的请求和响应格式。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CreateGameRequest(BaseModel):
    config_name: str = "standard_6"
    player_names: Optional[List[str]] = None
    shuffle: bool = True
    player_styles: Optional[Dict[int, str]] = None


class StepResponse(BaseModel):
    phase: str
    day_number: int
    step_data: Dict[str, Any] = Field(default_factory=dict)
    players: List[Dict[str, Any]] = Field(default_factory=list)
    dialogues: List[Dict[str, Any]] = Field(default_factory=list)
    deaths: List[Dict[str, Any]] = Field(default_factory=list)
    is_game_over: bool = False
    winner: Optional[str] = None
    summaries: List[Dict[str, Any]] = Field(default_factory=list)


class GameStatusResponse(BaseModel):
    game_id: str
    phase: str
    day_number: int
    players: List[Dict[str, Any]] = Field(default_factory=list)
    dialogues: List[Dict[str, Any]] = Field(default_factory=list)
    death_records: List[Dict[str, Any]] = Field(default_factory=list)
    winner: Optional[str] = None
    is_game_over: bool = False
    config_name: str = ""
    summaries: List[Dict[str, Any]] = Field(default_factory=list)
    summary_status: str = "idle"  # idle | generating | complete
    summary_done: int = 0
    summary_total: int = 0


class GameSummaryResponse(BaseModel):
    game_id: str
    phase: str
    day_number: int
    alive_count: int
    is_game_over: bool
    summaries: List[Dict[str, Any]] = Field(default_factory=list)


class ConfigInfo(BaseModel):
    name: str
    description: str
    player_count: int


class StartEvolveRequest(BaseModel):
    rounds: int = Field(default=3, ge=1, le=50)
    config_name: str = "simple_4"
    shuffle: bool = True


class EvolveLogEntry(BaseModel):
    time: str
    text: str


class EvolveJobResponse(BaseModel):
    job_id: str
    status: str
    rounds: int
    config_name: str
    current_round: int = 0
    logs: List[EvolveLogEntry] = Field(default_factory=list)
    prompt_changes: Dict[str, int] = Field(default_factory=dict)
    good_wins: int = 0
    evil_wins: int = 0
    error: Optional[str] = None


class PromptInfo(BaseModel):
    role: str
    content: str
    versions: List[str] = Field(default_factory=list)
