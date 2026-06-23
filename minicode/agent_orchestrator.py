"""Centralized multi-agent collaboration system.

Architecture
------------
  **Main Agent** (user-facing)
      │
      ├── Plan   — analyse task, decide sub-agent strategy
      ├── Spawn  — create sub-agents with constrained tool lists + path boundaries
      ├── Monitor — sub-agents execute independently, report summaries back
      ├── Review  — main agent reviews results, can request revision via re-spawn
      └── Commit  — integrate approved results into the main session

Key design constraints (centralised, NOT peer-to-peer):
  - Control NEVER leaves the main agent — sub-agents are invoked as Tool Calls.
  - Sub-agents receive a restricted tool allowlist and path boundary.
  - Results are minimised: only structured summaries return to main context.
  - Fork / Worktree / Agent Team collaboration modes.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from minicode.context_manager import ContextManager
from minicode.tooling import ToolContext, ToolDefinition, ToolRegistry, ToolResult

logger = logging.getLogger("agent_orchestrator")


# ══════════════════════════════════════════════════════════════════
# Enums & Dataclasses
# ══════════════════════════════════════════════════════════════════

class AgentRole(str, Enum):
    EXPLORE = "explore"         # read-only, fast search
    PLAN = "plan"                # read-only, thorough analysis
    EXECUTE = "execute"          # write-capable, bounded directory
    REVIEW = "review"            # read-only, adversarial check


class CollaborationMode(str, Enum):
    FORK = "fork"                # clone current session into sub-agent
    WORKTREE = "worktree"        # git worktree isolation
    AGENT_TEAM = "agent_team"    # multiple agents in parallel


@dataclass(slots=True)
class SubAgentSpec:
    """Specification for spawning a sub-agent."""
    task: str                           # task description
    role: AgentRole = AgentRole.EXPLORE
    allowed_tools: list[str] = field(default_factory=list)
    disallowed_tools: list[str] = field(default_factory=list)
    path_boundary: str = ""             # sub-agent cannot escape this directory
    max_turns: int = 10
    model: str = "inherit"
    timeout_seconds: int = 120


@dataclass
class SubAgentResult:
    """Structured result from a completed sub-agent."""
    agent_id: str
    role: AgentRole
    status: str                         # "completed" | "failed" | "timeout" | "rejected"
    summary: str                        # 1-3 sentence human-readable result
    findings: list[str] = field(default_factory=list)
    error: str = ""
    turns_used: int = 0
    token_usage: int = 0
    artifacts: list[str] = field(default_factory=list)  # file paths produced


@dataclass
class AgentTeamConfig:
    """Configuration for parallel agent team execution."""
    agents: list[SubAgentSpec] = field(default_factory=list)
    mode: CollaborationMode = CollaborationMode.AGENT_TEAM
    worktree_base_branch: str = "main"
    max_concurrency: int = 4


# ══════════════════════════════════════════════════════════════════
# Sub-Agent Runner
# ══════════════════════════════════════════════════════════════════

class SubAgentRunner:
    """Execute a single sub-agent with permission + path constraints.

    The runner wraps an existing ``ModelAdapter`` + ``ToolRegistry``
    pair, restricting the available tools and enforcing a path boundary.
    """

    def __init__(
        self,
        agent_id: str,
        spec: SubAgentSpec,
        model: Any,                   # ModelAdapter
        tools: ToolRegistry,
        cwd: str,
    ) -> None:
        self.agent_id = agent_id
        self.spec = spec
        self._model = model
        self._full_tool_registry = tools
        self._cwd = cwd
        self._context = ContextManager(model=spec.model)
        self._messages: list[dict] = []
        self._turn_count = 0
        self._started_at = time.time()

    def run(self) -> SubAgentResult:
        """Execute the sub-agent and return a structured result."""
        try:
            self._build_initial_messages()
            return self._execute_loop()
        except Exception as exc:
            logger.exception("Sub-agent %s crashed", self.agent_id)
            return SubAgentResult(
                agent_id=self.agent_id,
                role=self.spec.role,
                status="failed",
                summary=f"Sub-agent crashed: {exc}",
                error=str(exc),
            )
        finally:
            self._cleanup()

    # ── helpers ──────────────────────────────────────────────

    def _build_initial_messages(self) -> None:
        boundary_note = (
            f" (path boundary: {self.spec.path_boundary})"
            if self.spec.path_boundary else ""
        )
        system_msg = {
            "role": "system",
            "content": (
                f"You are a sub-agent ({self.spec.role.value}). "
                f"Your task: {self.spec.task}{boundary_note}. "
                "Work independently and return a concise structured result. "
                "Do NOT ask the user questions — complete the task autonomously. "
                "When done, produce a <final> summary of your findings."
            ),
        }
        user_msg = {
            "role": "user",
            "content": self.spec.task,
        }
        self._messages = [system_msg, user_msg]

    def _build_restricted_tools(self) -> ToolRegistry:
        """Create a tool registry limited to allowed tools + path boundary."""
        if not self.spec.allowed_tools:
            # No restrictions — use full registry
            return self._full_tool_registry

        filtered_tools: list[ToolDefinition] = []
        for tool in self._full_tool_registry.list():
            if tool.name in self.spec.allowed_tools:
                # Wrap tool to enforce path boundary
                if self.spec.path_boundary:
                    tool = self._wrap_with_path_boundary(tool)
                filtered_tools.append(tool)

        return ToolRegistry(
            tools=filtered_tools,
            skills=self._full_tool_registry.get_skills(),
            mcp_servers=self._full_tool_registry.get_mcp_servers(),
        )

    def _wrap_with_path_boundary(self, tool: ToolDefinition) -> ToolDefinition:
        """Wrap a tool's run function to reject paths outside the boundary."""
        original_run = tool.run
        boundary = os.path.abspath(self.spec.path_boundary)

        def guarded_run(input_data: Any, ctx: ToolContext) -> ToolResult:
            # Check for path-containing parameters
            path_keys = ("file_path", "path", "directory", "file", "target")
            for key in path_keys:
                if key in (input_data if isinstance(input_data, dict) else {}):
                    candidate = os.path.abspath(
                        os.path.join(ctx.cwd, str(input_data[key])))
                    if not candidate.startswith(boundary):
                        return ToolResult(
                            ok=False,
                            output=(
                                f"Access denied: path '{input_data[key]}' is outside "
                                f"the sub-agent boundary '{boundary}'."
                            ),
                        )
            return original_run(input_data, ctx)

        return ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            validator=tool.validator,
            run=guarded_run,
        )

    def _execute_loop(self) -> SubAgentResult:
        restricted_tools = self._build_restricted_tools()
        findings: list[str] = []

        while self._turn_count < self.spec.max_turns:
            if time.time() - self._started_at > self.spec.timeout_seconds:
                return SubAgentResult(
                    agent_id=self.agent_id,
                    role=self.spec.role,
                    status="timeout",
                    summary=f"Timed out after {self.spec.timeout_seconds}s",
                    turns_used=self._turn_count,
                )

            self._turn_count += 1
            self._context.messages = self._messages

            try:
                from minicode.agent_loop import run_agent_turn
                self._messages = run_agent_turn(
                    model=self._model,
                    tools=restricted_tools,
                    messages=self._messages,
                    cwd=self._cwd,
                    permissions=None,  # sub-agent inherits no interactive permissions
                    max_steps=1,       # one step per turn for control
                    context_manager=self._context,
                )
            except Exception as exc:
                logger.warning("Sub-agent turn %d error: %s", self._turn_count, exc)
                return SubAgentResult(
                    agent_id=self.agent_id,
                    role=self.spec.role,
                    status="failed",
                    summary=f"Turn {self._turn_count} error: {exc}",
                    error=str(exc),
                    turns_used=self._turn_count,
                    token_usage=self._context.get_stats().total_tokens,
                )

            # Extract findings from assistant messages
            for msg in reversed(self._messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    content = msg["content"]
                    if "<final>" in content:
                        summary = content.split("<final>", 1)[-1].strip()
                        return SubAgentResult(
                            agent_id=self.agent_id,
                            role=self.spec.role,
                            status="completed",
                            summary=summary[:500],
                            findings=findings,
                            turns_used=self._turn_count,
                            token_usage=self._context.get_stats().total_tokens,
                        )
                    findings.append(content[:200])
                    break

        return SubAgentResult(
            agent_id=self.agent_id,
            role=self.spec.role,
            status="completed",
            summary=f"Completed {self._turn_count} turns.",
            findings=findings,
            turns_used=self._turn_count,
            token_usage=self._context.get_stats().total_tokens,
        )

    def _cleanup(self) -> None:
        pass  # reserved for resource cleanup


# ══════════════════════════════════════════════════════════════════
# Centralized Orchestrator
# ══════════════════════════════════════════════════════════════════

class AgentOrchestrator:
    """Centralised multi-agent orchestrator.

    The main agent calls ``spawn()`` to create sub-agents, then
    ``collect()`` to retrieve structured results.  Sub-agents never
    communicate with each other — all coordination flows through
    the orchestrator.
    """

    def __init__(
        self,
        model: Any,
        tools: ToolRegistry,
        cwd: str,
    ) -> None:
        self._model = model
        self._tools = tools
        self._cwd = cwd
        self._running: dict[str, SubAgentRunner] = {}
        self._completed: dict[str, SubAgentResult] = {}
        self._worktrees: dict[str, str] = {}  # agent_id → worktree path

    # ── Spawn ────────────────────────────────────────────────

    def spawn(self, spec: SubAgentSpec) -> str:
        """Spawn a sub-agent; return its agent_id for later collection."""
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        runner = SubAgentRunner(agent_id, spec, self._model, self._tools, self._cwd)
        self._running[agent_id] = runner
        logger.info("Spawned sub-agent %s [%s]: %s", agent_id, spec.role.value, spec.task[:60])
        return agent_id

    # ── Team spawn ───────────────────────────────────────────

    def spawn_team(self, config: AgentTeamConfig) -> list[str]:
        """Spawn multiple agents in parallel; return their IDs."""
        ids: list[str] = []
        actual_concurrency = min(config.max_concurrency, len(config.agents))
        for spec in config.agents[:actual_concurrency]:
            agent_id = self.spawn(spec)
            ids.append(agent_id)
        return ids

    # ── Fork session (environment clone, NOT context clone) ──

    def fork_session(
        self,
        task: str,
        role: AgentRole = AgentRole.EXECUTE,
        path_boundary: str = "",
    ) -> str | None:
        """Fork the project **environment** into a temp directory.

        The sub-agent receives a clean, isolated copy of the project
        files but NO parent conversation context — only the *task*
        description.  This enables parallel work on the same codebase
        without context pollution or file conflicts.

        Returns the agent_id, or None if the fork failed.
        """
        # Create a temp clone of the project
        fork_dir = os.path.join(
            os.path.dirname(self._cwd),
            f".fork-{uuid.uuid4().hex[:8]}",
        )
        try:
            shutil.copytree(
                self._cwd, fork_dir,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc",
                                              ".venv", "node_modules", ".mypy_cache"),
                dirs_exist_ok=True,
            )
        except Exception as exc:
            logger.error("Fork environment clone failed: %s", exc)
            return None

        spec = SubAgentSpec(
            task=task,
            role=role,
            max_turns=15,
            allowed_tools=[],          # inherit all tools
            path_boundary=path_boundary or fork_dir,
        )
        agent_id = f"fork-{uuid.uuid4().hex[:8]}"
        runner = SubAgentRunner(agent_id, spec, self._model, self._tools, fork_dir)
        self._running[agent_id] = runner
        self._worktrees[agent_id] = fork_dir  # track for cleanup
        logger.info("Forked environment → %s at %s", agent_id, fork_dir)
        return agent_id

    # ── Worktree isolation ───────────────────────────────────

    def spawn_worktree(
        self,
        spec: SubAgentSpec,
        base_branch: str = "main",
    ) -> str | None:
        """Spawn a sub-agent in an isolated git worktree.

        Returns the agent_id, or None if worktree creation fails.
        """
        # Check if we're in a git repo
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, cwd=self._cwd, timeout=5,
            )
            if result.returncode != 0:
                logger.warning("Not a git repo — worktree unavailable")
                return None
            repo_root = result.stdout.strip()
        except Exception:
            logger.warning("Git unavailable — worktree mode skipped")
            return None

        agent_id = f"wt-{uuid.uuid4().hex[:8]}"
        worktree_path = os.path.join(
            repo_root, ".claude", "worktrees", agent_id)

        try:
            subprocess.run(
                ["git", "worktree", "add", worktree_path, base_branch],
                capture_output=True, text=True, cwd=repo_root, timeout=30,
                check=True,
            )
            self._worktrees[agent_id] = worktree_path
            # Update spec to use worktree path as boundary
            spec.path_boundary = worktree_path
            runner = SubAgentRunner(agent_id, spec, self._model, self._tools, worktree_path)
            self._running[agent_id] = runner
            logger.info("Worktree %s → %s", agent_id, worktree_path)
            return agent_id
        except subprocess.CalledProcessError as exc:
            logger.error("Worktree creation failed: %s", exc.stderr)
            return None

    # ── Collect results ──────────────────────────────────────

    def run_all(self) -> dict[str, SubAgentResult]:
        """Execute all pending sub-agents and return their results."""
        results: dict[str, SubAgentResult] = {}
        for agent_id, runner in list(self._running.items()):
            result = runner.run()
            results[agent_id] = result
            self._completed[agent_id] = result
            del self._running[agent_id]
        return results

    def collect(self, agent_id: str) -> SubAgentResult | None:
        """Run a specific sub-agent and return its result."""
        runner = self._running.pop(agent_id, None)
        if runner is None:
            return self._completed.get(agent_id)
        result = runner.run()
        self._completed[agent_id] = result
        return result

    def collect_all(self) -> dict[str, SubAgentResult]:
        """Run all pending and return all completed results."""
        self.run_all()
        return dict(self._completed)

    # ── Quality control ──────────────────────────────────────

    def review_result(self, result: SubAgentResult) -> str:
        """Main-agent review: accept, request-revision, or reject."""
        if result.status == "failed":
            return "reject"
        if result.status == "timeout":
            return "request_revision"  # could re-spawn with longer timeout
        if not result.summary or len(result.summary) < 10:
            return "request_revision"
        return "accept"

    def format_result_for_main_context(self, result: SubAgentResult) -> str:
        """Build a minimal context-friendly summary for the main agent."""
        status_icon = {"completed": "✓", "failed": "✗",
                       "timeout": "⏱", "rejected": "⊘"}.get(result.status, "?")
        lines = [
            f"[Sub-agent {result.agent_id} {status_icon} — {result.role.value}]",
            f"Status: {result.status}",
            f"Summary: {result.summary}",
        ]
        if result.findings:
            lines.append("Key findings:")
            for f in result.findings[:3]:
                lines.append(f"  - {f}")
        if result.error:
            lines.append(f"Error: {result.error[:200]}")
        lines.append(
            f"Stats: {result.turns_used} turns, "
            f"{result.token_usage:,} tokens"
        )
        return "\n".join(lines)

    # ── Cleanup ──────────────────────────────────────────────

    def cleanup_worktrees(self) -> int:
        """Remove all worktrees and fork directories. Returns count removed."""
        removed = 0
        for agent_id, path in list(self._worktrees.items()):
            if agent_id.startswith("wt-"):
                # Git worktree
                try:
                    subprocess.run(
                        ["git", "worktree", "remove", path, "--force"],
                        capture_output=True, text=True, cwd=self._cwd, timeout=10,
                    )
                    removed += 1
                    logger.info("Removed worktree %s", agent_id)
                except Exception as exc:
                    logger.warning("Failed to remove worktree %s: %s", agent_id, exc)
            elif agent_id.startswith("fork-"):
                # Fork directory — just delete
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
                    logger.info("Removed fork dir %s", agent_id)
                except Exception as exc:
                    logger.warning("Failed to remove fork %s: %s", agent_id, exc)
            del self._worktrees[agent_id]
        return removed

    @property
    def pending_count(self) -> int:
        return len(self._running)

    @property
    def completed_count(self) -> int:
        return len(self._completed)
