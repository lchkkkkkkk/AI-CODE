"""Tool: spawn_agent — centralised multi-agent orchestration tool.

The main agent calls this tool to delegate work to sub-agents.
Control stays with the main agent — sub-agents are executed as
blocking tool calls with constrained permissions.
"""

from __future__ import annotations

from typing import Any

from minicode.agent_orchestrator import (
    AgentOrchestrator,
    AgentRole,
    AgentTeamConfig,
    CollaborationMode,
    SubAgentSpec,
)
from minicode.tooling import ToolContext, ToolDefinition, ToolResult

# Singleton orchestrator reference (set by main.py at init time)
_orchestrator: AgentOrchestrator | None = None


def set_orchestrator(orchestrator: AgentOrchestrator) -> None:
    """Wire the orchestrator singleton for the tool to use."""
    global _orchestrator
    _orchestrator = orchestrator


_ROLE_MAP: dict[str, AgentRole] = {
    "explore": AgentRole.EXPLORE,
    "plan": AgentRole.PLAN,
    "execute": AgentRole.EXECUTE,
    "review": AgentRole.REVIEW,
}


def _validate(input_data: dict) -> dict:
    task = input_data.get("task", "")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task is required")
    role = input_data.get("role", "explore")
    if role not in _ROLE_MAP:
        raise ValueError(f"role must be one of {list(_ROLE_MAP.keys())}")
    return {
        "task": task.strip(),
        "role": role,
        "allowed_tools": input_data.get("allowed_tools", []),
        "path_boundary": input_data.get("path_boundary", ""),
        "max_turns": max(1, min(20, input_data.get("max_turns", 10))),
        "mode": input_data.get("mode", "single"),
        "team_tasks": input_data.get("team_tasks", []),
        "worktree_base": input_data.get("worktree_base", "main"),
    }


def create_spawn_agent_tool(cwd: str) -> ToolDefinition:
    def _run(input_data: dict, ctx: ToolContext) -> ToolResult:
        if _orchestrator is None:
            return ToolResult(
                ok=False,
                output="Agent orchestrator not initialised. "
                       "Multi-agent features require the orchestrator to be wired at startup.",
            )

        task = input_data["task"]
        role = _ROLE_MAP[input_data["role"]]
        mode = input_data["mode"]
        allowed_tools = input_data["allowed_tools"]

        # ── Mode: Agent Team (parallel spawn) ─────────────
        if mode == "team" and input_data["team_tasks"]:
            config = AgentTeamConfig(
                agents=[
                    SubAgentSpec(
                        task=t["task"] if isinstance(t, dict) else str(t),
                        role=_ROLE_MAP.get(
                            t.get("role", "explore") if isinstance(t, dict) else "explore",
                            AgentRole.EXPLORE,
                        ),
                        allowed_tools=allowed_tools,
                        path_boundary=input_data["path_boundary"],
                        max_turns=input_data["max_turns"],
                    )
                    for t in input_data["team_tasks"]
                ],
                mode=CollaborationMode.AGENT_TEAM,
                max_concurrency=3,
            )
            agent_ids = _orchestrator.spawn_team(config)
            results = _orchestrator.run_all()
            parts = [f"Agent Team: {len(agent_ids)} agents spawned."]
            for aid in agent_ids:
                r = results.get(aid)
                if r:
                    parts.append(_orchestrator.format_result_for_main_context(r))
            return ToolResult(ok=True, output="\n\n".join(parts))

        # ── Mode: Worktree (git isolation) ────────────────
        if mode == "worktree":
            spec = SubAgentSpec(
                task=task,
                role=role,
                allowed_tools=allowed_tools,
                path_boundary=input_data["path_boundary"],
                max_turns=input_data["max_turns"],
            )
            agent_id = _orchestrator.spawn_worktree(spec, input_data["worktree_base"])
            if agent_id is None:
                return ToolResult(
                    ok=False,
                    output="Worktree creation failed. Ensure the project is a git repository.",
                )
            result = _orchestrator.collect(agent_id)
            if result is None:
                return ToolResult(ok=False, output=f"Sub-agent {agent_id} not found.")
            review = _orchestrator.review_result(result)
            formatted = _orchestrator.format_result_for_main_context(result)
            return ToolResult(
                ok=review != "reject",
                output=f"[Worktree: {agent_id}]\n{formatted}\n\nReview: {review}",
            )

        # ── Mode: Fork (clone environment, clean context) ──
        if mode == "fork":
            agent_id = _orchestrator.fork_session(
                task, role, input_data["path_boundary"])
            if agent_id is None:
                return ToolResult(
                    ok=False,
                    output="Fork failed — could not clone project environment.",
                )
            result = _orchestrator.collect(agent_id)
            if result is None:
                return ToolResult(ok=False, output=f"Fork session {agent_id} failed.")
            return ToolResult(
                ok=result.status == "completed",
                output=_orchestrator.format_result_for_main_context(result),
            )

        # ── Mode: Single (default) ─────────────────────────
        spec = SubAgentSpec(
            task=task,
            role=role,
            allowed_tools=allowed_tools,
            path_boundary=input_data["path_boundary"],
            max_turns=input_data["max_turns"],
        )
        agent_id = _orchestrator.spawn(spec)
        result = _orchestrator.collect(agent_id)
        if result is None:
            return ToolResult(ok=False, output=f"Sub-agent {agent_id} not found.")
        return ToolResult(
            ok=result.status == "completed",
            output=_orchestrator.format_result_for_main_context(result),
        )

    return ToolDefinition(
        name="spawn_agent",
        description=(
            "Spawn one or more sub-agents to execute tasks in parallel with "
            "constrained tool permissions and path boundaries. Modes: single "
            "(default), fork (clone session), worktree (git isolation), "
            "team (parallel multi-agent). Sub-agents work independently and "
            "return structured results — control stays with the main agent."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string",
                         "description": "Task description for the sub-agent"},
                "role": {"type": "string", "enum": list(_ROLE_MAP.keys()),
                         "description": "explore/plan/execute/review"},
                "allowed_tools": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Tools the sub-agent may use (empty = all)"},
                "path_boundary": {"type": "string",
                                  "description": "Sub-agent cannot escape this path"},
                "max_turns": {"type": "integer",
                              "description": "Max turns (1-20, default 10)"},
                "mode": {"type": "string",
                         "enum": ["single", "fork", "worktree", "team"],
                         "description": "Collaboration mode"},
                "team_tasks": {"type": "array",
                               "description": "Tasks for team mode"},
                "worktree_base": {"type": "string",
                                  "description": "Base branch for worktree (default: main)"},
            },
            "required": ["task"],
        },
        validator=_validate,
        run=_run,
    )
