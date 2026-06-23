"""Layered context compression system.

Implements a four-layer context governance loop:

  **Summary Preview → Placeholder Replacement → On-demand Retrieval → Overflow Fallback**

Layer 1 — Large Tool Result Offloading
  Intercepts tool outputs > threshold, persists full content to disk, and
  replaces in-context content with a structured preview + retrieval path.

Layer 2 — Cache-Friendly Placeholder Compression
  Marks static system-prompt blocks with ``<cache_control>`` breakpoints
  so the Anthropic API can cache them across turns, reducing input token cost.

Layer 3 — Structured Note Summarisation
  When compaction triggers, replaces raw tool outputs with structured
  key-value summaries rather than silently dropping them.

Layer 4 — Overflow Governance
  Auto-compact at 95% utilisation with a proper summary note instead of a
  misleading "summarised" label when no summary was actually generated.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from minicode.tooling import ToolResult

logger = logging.getLogger("layered_context")

# ── Constants ──────────────────────────────────────────────────
TOOL_RESULT_SIZE_THRESHOLD = 10_000   # chars — over this → offload to disk
TOOL_RESULT_PREVIEW_LENGTH = 800      # chars kept in-context
TOOL_RESULT_DIR_NAME = "tool_results" # sub-dir under ~/.mini-code/


# ══════════════════════════════════════════════════════════════════
# Layer 1 — Tool Result Externalization
# ══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class OffloadedResult:
    """Metadata for a tool result that has been externalized to disk."""
    tool_name: str
    tool_use_id: str
    file_path: str          # absolute path on disk
    size_chars: int
    preview: str            # truncated preview kept in-context
    created_at: float = field(default_factory=time.time)


class ToolResultStorage:
    """Persist large tool results to disk and serve previews."""

    def __init__(self, session_id: str = "") -> None:
        from minicode.config import MINI_CODE_DIR
        self._session_id = session_id or f"turn-{int(time.time())}"
        self._base = MINI_CODE_DIR / TOOL_RESULT_DIR_NAME / self._session_id

    # ── offload ─────────────────────────────────────────────

    def should_offload(self, output: str) -> bool:
        return len(output) > TOOL_RESULT_SIZE_THRESHOLD

    def offload(self, tool_name: str, tool_use_id: str,
                output: str) -> OffloadedResult:
        """Write full output to disk; return an OffloadedResult with preview."""
        self._ensure_dir()
        file_name = f"{tool_name}_{tool_use_id}.txt"[:120]
        file_path = str(self._base / file_name)
        file_path.encode("utf-8")[:240]
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(output)

        size = len(output)
        preview = output[:TOOL_RESULT_PREVIEW_LENGTH]
        return OffloadedResult(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            file_path=file_path,
            size_chars=size,
            preview=preview,
        )

    def build_preview_message(self, offloaded: OffloadedResult) -> str:
        """Return the message that replaces the raw tool result in context."""
        return (
            f"[TOOL RESULT OFFLOADED — {offloaded.tool_name}]\n"
            f"Preview (first {TOOL_RESULT_PREVIEW_LENGTH} chars of "
            f"{offloaded.size_chars:,}):\n"
            f"{'─' * 50}\n"
            f"{offloaded.preview}\n"
            f"{'─' * 50}\n"
            f"Full result stored at: {offloaded.file_path}\n"
            f"To retrieve the full output, use the read_file tool "
            f"with path={offloaded.file_path}"
        )

    def retrieve(self, tool_use_id: str) -> str | None:
        """Read the full tool result from disk by tool_use_id."""
        if not self._base.exists():
            return None
        for f in self._base.iterdir():
            if tool_use_id in f.name:
                return f.read_text(encoding="utf-8")
        return None

    def cleanup(self) -> int:
        """Remove all files for this session.  Returns number deleted."""
        deleted = 0
        if self._base.exists():
            for f in self._base.iterdir():
                try:
                    f.unlink()
                    deleted += 1
                except OSError:
                    pass
            try:
                self._base.rmdir()
            except OSError:
                pass
        return deleted

    def _ensure_dir(self) -> None:
        self._base.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# Layer 2 — Cache-Friendly Prompt Structuring
# ══════════════════════════════════════════════════════════════════

class PromptCacheOptimizer:
    """Inserts Anthropic ``cache_control`` breakpoints into the system prompt.

    The strategy:
     1. Wrap STATIC blocks (identity, governance rules) in cache markers.
     2. Keep DYNAMIC blocks (permissions, routed skills, memories) outside
        the cacheable region so they don't invalidate the cache every turn.
     3. Return a list of content blocks (instead of a single string) so the
        adapter can send them with per-block cache_control.
    """

    # Static blocks — safe to cache across turns
    STATIC_MARKER_START = "<!-- cache_control: static-start -->"
    STATIC_MARKER_END = "<!-- cache_control: static-end -->"

    @staticmethod
    def wrap_static(parts: list[str]) -> tuple[list[str], int]:
        """Identify static vs dynamic prompt parts and return the
        reordered list with cache markers on the static prefix.

        Returns (marked_parts, static_part_count).
        """
        # The first few parts of the system prompt are always static:
        #   - identity / role definition
        #   - governance rules
        # We detect the boundary: the first part that contains "Permission"
        # or "Relevant skills" or "Available skills".
        result: list[str] = []
        in_static = True
        static_count = 0

        for part in parts:
            is_dynamic = any(kw in part for kw in (
                "Permission context",
                "Relevant skills",
                "Available skills",
                "Relevant past experience",
                "Configured MCP",
                "Global instructions",
                "Project instructions",
                "SEQUENTIAL THINKING",
            ))
            if in_static and is_dynamic:
                # Close the static block
                if static_count > 0:
                    result[-1] = result[-1] + "\n" + PromptCacheOptimizer.STATIC_MARKER_END
                in_static = False

            if in_static and not is_dynamic:
                if static_count == 0:
                    part = PromptCacheOptimizer.STATIC_MARKER_START + "\n" + part
                static_count += 1

            result.append(part)

        return result, static_count

    @staticmethod
    def generate_ephemeral_marker() -> dict:
        """Return the ``cache_control`` dict for the Anthropic API."""
        return {"type": "ephemeral"}


# ══════════════════════════════════════════════════════════════════
# Layer 3 — Structured Note Summarisation
# ══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class StructuredNote:
    """A structured summary extracted from raw tool output."""
    tool_name: str
    tool_use_id: str
    summary: str                 # 1-2 sentence human-readable summary
    key_values: dict[str, str]   # extracted key-value pairs
    original_size_chars: int


class NoteSummarizer:
    """Convert raw tool outputs into structured summaries.

    Uses heuristics (not LLM) — zero additional cost.
    """

    SUMMARIZE_MAX_CHARS = 200

    def summarize(self, tool_name: str, tool_use_id: str,
                  output: str) -> StructuredNote:
        """Produce a structured note for *output*."""
        size = len(output)
        kv = self._extract_key_values(output)
        summary = self._build_summary(tool_name, kv, size)
        return StructuredNote(
            tool_name=tool_name,
            tool_use_id=tool_use_id,
            summary=summary,
            key_values=kv,
            original_size_chars=size,
        )

    def format_note(self, note: StructuredNote) -> str:
        """Render a structured note as context-friendly text."""
        lines = [
            f"[Structured Summary — {note.tool_name}]",
            f"{note.summary}",
        ]
        if note.key_values:
            lines.append("Key findings:")
            for k, v in list(note.key_values.items())[:8]:
                lines.append(f"  {k}: {v[:120]}")
        lines.append(
            f"[Original output: {note.original_size_chars:,} chars — "
            f"compacted to {len(lines[-1]):,} chars]"
        )
        return "\n".join(lines)

    # ── heuristics ──────────────────────────────────────────

    def _build_summary(self, tool_name: str, kv: dict[str, str],
                       size: int) -> str:
        """Build a one-line summary from available signals."""
        if "error" in kv:
            return f"{tool_name} failed: {kv['error'][:100]}"
        if "count" in kv:
            return f"{tool_name} returned {kv['count']} items ({size:,} chars total)."
        if "lines" in kv:
            return f"{tool_name} output {kv['lines']} lines ({size:,} chars)."
        return f"{tool_name} completed ({size:,} chars output)."

    def _extract_key_values(self, output: str) -> dict[str, str]:
        """Extract simple key-value signals from tool output."""
        kv: dict[str, str] = {}
        if not output:
            return kv

        lines = output.strip().split("\n")

        # Error detection
        for line in lines[:5]:
            stripped = line.strip()
            for kw in ("error:", "Error:", "ERROR:", "failed", "Traceback",
                       "Exception:"):
                if kw in stripped:
                    kv["error"] = stripped[:200]
                    break
            if "error" in kv:
                break

        # Line / item counting
        if len(lines) > 1:
            kv["lines"] = str(len(lines))

        # First significant line
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and len(stripped) > 3:
                kv["first_line"] = stripped[:200]
                break

        return kv


# ══════════════════════════════════════════════════════════════════
# Layer 4 — Overflow Governance
# ══════════════════════════════════════════════════════════════════

@dataclass
class CompactDecision:
    """Result of overflow governance check."""
    should_compact: bool
    usage_pct: float
    reason: str = ""
    suggested_action: str = "none"  # "none" | "offload" | "compact" | "block"


class OverflowGovernor:
    """Four-level overflow decision engine.

    ┌──────────┬────────────────────────────────┐
    │ Level    │ Action                          │
    ├──────────┼────────────────────────────────┤
    │ normal   │ < 70% — no action               │
    │ warning  │ 70-85% — offload large results  │
    │ critical │ 85-95% — compact + summarize    │
    │ blocked  │ > 95% — hard compact            │
    └──────────┴────────────────────────────────┘
    """

    LEVEL_NORMAL = 0.70
    LEVEL_WARNING = 0.85
    LEVEL_CRITICAL = 0.95

    def check(self, usage_pct: float,
              large_result_count: int = 0) -> CompactDecision:
        """Decide what to do based on context utilisation."""
        if usage_pct < self.LEVEL_NORMAL:
            return CompactDecision(
                should_compact=False, usage_pct=usage_pct,
                reason="normal", suggested_action="none",
            )
        if usage_pct < self.LEVEL_WARNING:
            if large_result_count > 0:
                return CompactDecision(
                    should_compact=True, usage_pct=usage_pct,
                    reason="warning — offloading large results",
                    suggested_action="offload",
                )
            return CompactDecision(
                should_compact=False, usage_pct=usage_pct,
                reason="warning — no large results to offload",
                suggested_action="none",
            )
        if usage_pct < self.LEVEL_CRITICAL:
            return CompactDecision(
                should_compact=True, usage_pct=usage_pct,
                reason="critical — compacting with summaries",
                suggested_action="compact",
            )
        return CompactDecision(
            should_compact=True, usage_pct=usage_pct,
            reason="blocked — hard compact required",
            suggested_action="block",
        )
