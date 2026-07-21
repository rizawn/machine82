"""
Agent subsystem — chat, planning, autonomous loops, and workspace analysis.

Usage:
    from core.agents import AgentLoop, ChatEngine, TaskPlanner
"""

__all__ = [
    "ChatEngine",
    "TaskPlanner",
    "Plan",
    "AgentLoop",
    "AgentResult",
    "WorkspaceAgent",
]

def __getattr__(name):
    if name == "ChatEngine":
        from core.agents.chat_engine import ChatEngine as _CE
        return _CE
    if name == "TaskPlanner":
        from core.agents.task_planner import TaskPlanner as _TP
        return _TP
    if name == "Plan":
        from core.agents.task_planner import Plan as _P
        return _P
    if name in ("AgentLoop", "AgentResult"):
        from core.agents.agent_loop import AgentLoop, AgentResult
        return {"AgentLoop": AgentLoop, "AgentResult": AgentResult}[name]
    if name == "WorkspaceAgent":
        from core.agents.workspace_agent import WorkspaceAgent as _WA
        return _WA
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
