from agent.base import BaseAgent

__all__ = [
    "BaseAgent",
    "PlayerAgent",
    "JudgeAgent",
    "SummaryAgent",
    "SelfReflector",
    "create_player_agent",
    "create_judge_agent",
]


def __getattr__(name):
    if name in {"PlayerAgent", "JudgeAgent", "create_player_agent", "create_judge_agent"}:
        from agent.player_agent import JudgeAgent, PlayerAgent, create_judge_agent, create_player_agent

        exports = {
            "PlayerAgent": PlayerAgent,
            "JudgeAgent": JudgeAgent,
            "create_player_agent": create_player_agent,
            "create_judge_agent": create_judge_agent,
        }
        return exports[name]

    if name == "SummaryAgent":
        from agent.summary_agent import SummaryAgent

        return SummaryAgent

    if name == "SelfReflector":
        from agent.reflector import SelfReflector

        return SelfReflector

    raise AttributeError(f"module 'agent' has no attribute {name!r}")
