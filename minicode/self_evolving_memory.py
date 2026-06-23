"""Self-evolving memory pipeline.

Implements the closed-loop memory lifecycle:
  **Execution → Reflection → Refinement → Classification → Index → Reuse**

Traces captured during agent tool execution are analysed by a rule engine
(zero LLM cost) that distills them into structured ``MemoryEntry`` objects.
Memories accrue confidence with repeated observation, decay when unused,
and are automatically injected into future system prompts via
``MemoryManager.search_relevant()``.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from minicode.memory import MEMORY_CATEGORIES, MemoryManager, MemoryEntry, MemoryScope

logger = logging.getLogger("self_evolving_memory")

# ── Chinese + English correction / negation keywords ───────────
_CORRECTION_KEYWORDS: list[str] = [
    "不要", "别", "改成", "纠正", "错了", "不对", "应该是",
    "don't", "don't use", "use ... not", "instead of", "should be",
    "prefer", "preferably", "avoid", "never use", "不用",
]


# ══════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ExecutionTrace:
    """Full trace of a single tool invocation."""
    tool_name: str
    tool_input: dict
    tool_output: str
    is_error: bool
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    user_input_context: str = ""


@dataclass(slots=True)
class ReflectionResult:
    """One distilled memory from the reflection phase."""
    category: str
    content: str
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    source_trace_indices: list[int] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# Engine
# ══════════════════════════════════════════════════════════════════

class MemoryEvolutionEngine:
    """Orchestrates the self-evolving memory pipeline.

    Usage::

        engine = MemoryEvolutionEngine(memory_manager)
        engine.capture_trace(trace)     # called during agent loop
        ...
        for result in engine.reflect_and_evolve(user_input):
            memory_manager.add_entry(...)
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self._memory = memory_manager
        self._traces: list[ExecutionTrace] = []
        # Persistent counters for pattern tracking across turns
        self._tool_success_counts: dict[str, list[dict]] = {}  # tool_name → [args]

    # ── Trace capture ──────────────────────────────────────────

    def capture_trace(self, trace: ExecutionTrace) -> None:
        """Record a tool execution trace."""
        self._traces.append(trace)
        # Update tool success counters for Rule 4
        if not trace.is_error:
            if trace.tool_name not in self._tool_success_counts:
                self._tool_success_counts[trace.tool_name] = []
            self._tool_success_counts[trace.tool_name].append(trace.tool_input)

    # ── Reflection pipeline ────────────────────────────────────

    def reflect_and_evolve(self, user_input: str) -> list[ReflectionResult]:
        """Run the full reflection pipeline and return distilled memories."""
        if not self._traces:
            return []

        all_results: list[ReflectionResult] = []

        # Rule 1: error patterns
        all_results.extend(self._extract_error_patterns())

        # Rule 2: code conventions
        all_results.extend(self._extract_code_conventions())

        # Rule 3: user preferences (needs user_input)
        all_results.extend(self._extract_user_preferences(user_input))

        # Rule 4: tool usage patterns
        all_results.extend(self._extract_tool_patterns())

        # Rule 5: project insights
        all_results.extend(self._extract_project_insights())

        # Persist results into MemoryManager
        stored = 0
        for result in all_results:
            if result.confidence >= 0.5:
                self._store_reflection(result)
                stored += 1

        logger.info(
            "Reflection complete: %d results, %d stored (%.0f%% hit rate)",
            len(all_results), stored,
            (stored / max(len(all_results), 1)) * 100,
        )

        # Clear this turn's traces (counters persist across turns)
        trace_count = len(self._traces)
        self._traces.clear()
        logger.debug("Cleared %d traces for next turn", trace_count)

        return all_results

    # ── Rule engines ───────────────────────────────────────────

    def _extract_error_patterns(self) -> list[ReflectionResult]:
        """Rule 1: Failed tool calls → error_pattern memories."""
        results: list[ReflectionResult] = []
        error_traces = [t for t in self._traces if t.is_error]

        for i, trace in enumerate(error_traces):
            summary = _summarise_error(trace.tool_output)
            if not summary:
                continue

            # Check if this error pattern already exists → boost confidence
            confidence = 0.6
            existing = self._memory.search_relevant(
                f"{trace.tool_name} {summary}", limit=1,
            )
            if existing and existing[0].category == "error_pattern":
                confidence = min(1.0, existing[0].confidence + 0.15)

            content = (
                f"Tool '{trace.tool_name}' failed: {summary}. "
                f"Check input parameters and retry with adjusted approach."
            )
            results.append(ReflectionResult(
                category="error_pattern",
                content=content,
                tags=[trace.tool_name, "error", "auto"],
                confidence=confidence,
                source_trace_indices=[i],
            ))
        return results

    def _extract_code_conventions(self) -> list[ReflectionResult]:
        """Rule 2: Successful edit/write → code_convention memories."""
        results: list[ReflectionResult] = []
        edit_tools = {"edit_file", "write_file", "patch_file", "modify_file", "multi_edit"}
        edit_traces = [t for t in self._traces if t.tool_name in edit_tools and not t.is_error]

        for i, trace in enumerate(edit_traces):
            file_path = _extract_file_path(trace.tool_input)
            if not file_path:
                continue
            ext = file_path.rsplit(".", 1)[-1] if "." in file_path else "unknown"

            content = (
                f"File '{file_path}' was modified by '{trace.tool_name}'. "
                f"Extension: .{ext}. Review the edit pattern for reusability."
            )
            results.append(ReflectionResult(
                category="code_convention",
                content=content,
                tags=[ext, trace.tool_name, "auto"],
                confidence=0.5,
                source_trace_indices=[i],
            ))
        return results

    def _extract_user_preferences(self, user_input: str) -> list[ReflectionResult]:
        """Rule 3: Correction/negation keywords → user_preference memories."""
        results: list[ReflectionResult] = []
        if not user_input:
            return results

        matched = any(
            kw.lower() in user_input.lower() for kw in _CORRECTION_KEYWORDS
        )
        if not matched:
            return results

        # Only extract when the user has actually corrected something
        # (i.e. the turn contains error traces or edit traces)
        has_errors = any(t.is_error for t in self._traces)
        has_edits = any(
            t.tool_name in {"edit_file", "write_file", "patch_file", "modify_file"}
            for t in self._traces
        )
        if not has_errors and not has_edits:
            return results

        content = (
            f"User expressed a preference: '{user_input}'. "
            "Adjust future responses and code style to match this preference."
        )
        results.append(ReflectionResult(
            category="user_preference",
            content=content,
            tags=["user", "preference", "auto"],
            confidence=0.7,
            source_trace_indices=list(range(len(self._traces))),
        ))
        return results

    def _extract_tool_patterns(self) -> list[ReflectionResult]:
        """Rule 4: Repeated successful tool usage → tool_usage memories."""
        results: list[ReflectionResult] = []
        for tool_name, inputs in self._tool_success_counts.items():
            n = len(inputs)
            if n < 3:
                continue
            # Extract common parameter keys
            param_keys: set[str] = set()
            for inp in inputs[-10:]:  # last 10
                if isinstance(inp, dict):
                    param_keys.update(inp.keys())
            confidence = min(1.0, 0.55 + 0.05 * min(n, 9))
            content = (
                f"Tool '{tool_name}' succeeded {n} times. "
                f"Common parameters: {', '.join(sorted(param_keys)[:5])}. "
                f"This tool is reliable for this project."
            )
            results.append(ReflectionResult(
                category="tool_usage",
                content=content,
                tags=[tool_name, "reliable", "auto"],
                confidence=confidence,
                source_trace_indices=[],
            ))
        return results

    def _extract_project_insights(self) -> list[ReflectionResult]:
        """Rule 5: list_files / grep_files → project_insight memories."""
        results: list[ReflectionResult] = []
        discovery_tools = {"list_files", "grep_files", "file_tree"}
        discovery_traces = [
            t for t in self._traces
            if t.tool_name in discovery_tools and not t.is_error
        ]
        for i, trace in enumerate(discovery_traces):
            path = trace.tool_input.get("path") or trace.tool_input.get("directory") or ""
            pattern = trace.tool_input.get("pattern", "")
            if not path and not pattern:
                continue
            insight = f"path={path}" if path else f"pattern='{pattern}'"
            content = (
                f"Project exploration via '{trace.tool_name}' found: {insight}. "
                "This path/pattern is relevant for future navigation."
            )
            results.append(ReflectionResult(
                category="project_insight",
                content=content,
                tags=["exploration", trace.tool_name, "auto"],
                confidence=0.5,
                source_trace_indices=[i],
            ))
        return results

    # ── Storage helper ─────────────────────────────────────────

    def _store_reflection(self, result: ReflectionResult) -> MemoryEntry | None:
        """Persist a single reflection result into the MemoryManager."""
        try:
            entry_id = f"evolve-{int(time.time())}-{result.category}"
            entry = MemoryEntry(
                id=entry_id,
                scope=MemoryScope.PROJECT,
                category=result.category,
                content=result.content,
                tags=result.tags,
                source_type="auto_reflection",
                confidence=result.confidence,
                decay_rate=0.01,  # gentle decay: ~3% per day when unused
            )
            return self._memory.memories[MemoryScope.PROJECT].update_or_add(entry)
        except Exception as exc:
            logger.warning("Failed to store reflection: %s", exc)
            return None


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════

def _summarise_error(output: str, max_len: int = 120) -> str:
    """Extract a concise error summary from tool output."""
    if not output:
        return ""
    text = output.strip()
    # Try to grab the last line (often the most informative error line)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return text[:max_len]
    # Prefer lines with "Error", "error", "failed", etc.
    for keyword in ("Error:", "error:", "ERROR:", "failed", "Failed", "Traceback",
                    "Exception", "cannot", "Could not", "No such file", "not found"):
        for line in lines:
            if keyword in line:
                return line[:max_len]
    return lines[-1][:max_len]


def _extract_file_path(tool_input: dict) -> str:
    """Pull a file-path from common tool-input keys."""
    for key in ("file_path", "path", "file", "target"):
        if key in tool_input:
            val = tool_input[key]
            if isinstance(val, str) and val:
                return val
    return ""
