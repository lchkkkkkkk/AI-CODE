from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol
from abc import abstractmethod


# ---------------------------------------------------------------------------
# Tool metadata (inspired by Claude Code's Tool type)
# ---------------------------------------------------------------------------

class ToolCapability(str, Enum):
    """Tool capability flags."""
    READ_ONLY = "read_only"
    DESTRUCTIVE = "destructive"
    CONCURRENCY_SAFE = "concurrency_safe"
    REQUIRES_PERMISSION = "requires_permission"


class RiskLevel(str, Enum):
    """Tool self-declared risk level for the security chain."""
    SAFE = "safe"             # read-only, no side effects (e.g. read_file, list_files)
    LOW = "low"               # minor side effects (e.g. web_fetch)
    MEDIUM = "medium"         # file modifications (e.g. edit_file, write_file)
    HIGH = "high"             # shell execution (e.g. run_command)
    CRITICAL = "critical"     # system-level danger (e.g. run_command with sudo)


@dataclass
class ToolMetadata:
    """Tool metadata with self-inspection (Layer 2 of security chain).

    Each tool declares its own risk profile — the security chain reads
    these declarations and decides whether to escalate to higher layers.
    """
    name: str
    description: str
    capabilities: set[ToolCapability] = field(default_factory=set)
    input_schema: dict[str, Any] = field(default_factory=dict)
    is_enabled: bool = True
    max_result_size_chars: int = 10_000
    tags: list[str] = field(default_factory=list)
    # Tool self-inspection fields (Layer 2)
    risk_level: RiskLevel = RiskLevel.SAFE
    requires_review: bool = False          # always ask human before executing
    safety_hints: list[str] = field(default_factory=list)  # human-readable warnings
    auto_approve_in_auto_mode: bool = False  # skip prompt in auto mode
    
    @property
    def is_read_only(self) -> bool:
        """Check if tool is read-only."""
        return ToolCapability.READ_ONLY in self.capabilities
    
    @property
    def is_destructive(self) -> bool:
        """Check if tool can modify/delete data."""
        return ToolCapability.DESTRUCTIVE in self.capabilities
    
    @property
    def is_concurrency_safe(self) -> bool:
        """Check if tool is safe for concurrent execution."""
        return ToolCapability.CONCURRENCY_SAFE in self.capabilities


# ---------------------------------------------------------------------------
# Tool Protocol (inspired by Claude Code's Tool interface)
# ---------------------------------------------------------------------------

class Tool(Protocol):
    """Tool protocol defining a complete tool lifecycle.
    
    Inspired by Claude Code's Tool type which includes:
    - call: Execution logic
    - description: Dynamic description generation
    - validate_input: Input validation
    - check_permissions: Permission checking
    - Metadata: is_read_only, is_destructive, etc.
    """
    
    @property
    def name(self) -> str: ...
    
    @property
    def description_template(self) -> str: ...
    
    def get_description(self, args: dict[str, Any], options: dict[str, Any] | None = None) -> str: ...
    def validate_input(self, args: dict[str, Any]) -> tuple[bool, str]: ...
    def check_permissions(self, args: dict[str, Any], context: ToolContext) -> tuple[bool, str]: ...
    def call(
        self,
        args: dict[str, Any],
        context: ToolContext,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> ToolResult: ...
    def is_enabled(self) -> bool: ...
    def is_read_only(self, args: dict[str, Any]) -> bool: ...
    def is_destructive(self, args: dict[str, Any]) -> bool: ...


@dataclass(slots=True)
class BackgroundTaskResult:
    taskId: str
    type: str
    command: str
    pid: int
    status: str
    startedAt: int


@dataclass(slots=True)
class ToolResult:
    ok: bool
    output: str
    backgroundTask: BackgroundTaskResult | None = None
    awaitUser: bool = False


@dataclass(slots=True)
class ToolContext:
    cwd: str
    permissions: Any | None = None


Validator = Callable[[Any], Any]
Runner = Callable[[Any, ToolContext], ToolResult]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    validator: Validator
    run: Runner
    risk_level: RiskLevel = RiskLevel.SAFE
    requires_review: bool = False
    safety_hints: list[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(
        self,
        tools: list[ToolDefinition],
        skills: list[dict[str, Any]] | None = None,
        mcp_servers: list[dict[str, Any]] | None = None,
        disposer: Callable[[], Any] | None = None,
    ) -> None:
        self._tools = tools
        self._skills = skills or []
        self._mcp_servers = mcp_servers or []
        self._disposer = disposer
        # Routing support
        self._skill_objects: list[Any] = []  # list[LoadedSkill]
        self._routed_skills: list[dict[str, Any]] | None = None

    def list(self) -> list[ToolDefinition]:
        return list(self._tools)

    def get_skills(self, routed_only: bool = False) -> list[dict[str, Any]]:
        if routed_only and self._routed_skills is not None:
            return list(self._routed_skills)
        return list(self._skills)

    def get_all_skills(self) -> list[dict[str, Any]]:
        """Always return all discovered skills (for /skills command)."""
        return list(self._skills)

    def get_skill_objects(self) -> list[Any]:
        """Return raw LoadedSkill objects (for the router)."""
        return list(self._skill_objects)

    def set_skill_objects(self, skill_objects: list[Any]) -> None:
        """Store LoadedSkill objects for routing."""
        self._skill_objects = list(skill_objects)

    def set_routed_skills(self, skills: list[dict[str, Any]] | None) -> None:
        """Set skills selected by the router for this turn."""
        self._routed_skills = list(skills) if skills is not None else None

    def get_mcp_servers(self) -> list[dict[str, Any]]:
        return list(self._mcp_servers)

    def find(self, name: str) -> ToolDefinition | None:
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    def execute(self, tool_name: str, input_data: Any, context: ToolContext) -> ToolResult:
        tool = self.find(tool_name)
        if tool is None:
            return ToolResult(ok=False, output=f"Unknown tool: {tool_name}")

        try:
            parsed = tool.validator(input_data)
            return tool.run(parsed, context)
        except (KeyboardInterrupt, SystemExit):
            # 这些异常应该向上传播，不应该被捕获
            raise
        except Exception as error:  # noqa: BLE001
            return ToolResult(ok=False, output=f"{type(tool).__name__} error: {error}")

    def dispose(self) -> None:
        if self._disposer is not None:
            self._disposer()
