"""Multi-layer security review chain.

Layers
------
  Layer 1 — Rule-Based Filtering
    Hard blocklists (file extensions, path patterns, IP ranges).  If a
    rule fires, the action is rejected immediately — no further layers run.

  Layer 2 — Tool Self-Inspection
    Each ``ToolDefinition`` declares its own ``risk_level`` and
    ``safety_hints``.  The chain reads these declarations and decides
    whether to continue to Layer 3.

  Layer 3 — AI Risk Classification (prompt injection + output safety)
    Pattern-based prompt injection detection on user input plus a
    lightweight output classifier for tool results.

  Layer 4 — Human Confirmation
    The existing ``PermissionManager.prompt`` gateway.  Reached only
    when the tool's risk level exceeds the configured auto-approve
    threshold *and* the AI classifier did not block it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from minicode.tooling import RiskLevel, ToolDefinition

logger = logging.getLogger("security_chain")


# ══════════════════════════════════════════════════════════════════
# Enums & Data Structures
# ══════════════════════════════════════════════════════════════════

class SecurityVerdict(str, Enum):
    ALLOW = "allow"               # proceed
    WARN = "warn"                 # proceed with warnings injected into context
    ESCALATE = "escalate"         # need human confirmation (Layer 4)
    BLOCK = "block"               # rejected — do not execute


@dataclass
class SecurityReport:
    """Full result from the security chain for a single action."""
    action_type: str                     # "tool_call" | "user_input" | "tool_result"
    tool_name: str = ""
    tool_risk_level: RiskLevel = RiskLevel.SAFE
    verdict: SecurityVerdict = SecurityVerdict.ALLOW
    layer_reached: int = 1               # which layer made the final decision
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    risk_score: float = 0.0              # 0.0 (safe) – 1.0 (critical)
    injection_detected: bool = False
    auto_approved: bool = False


# ══════════════════════════════════════════════════════════════════
# Layer 1 — Rule-Based Filtering
# ══════════════════════════════════════════════════════════════════

class RuleFilter:
    """Hard blocklists — no AI, no human, just rules."""

    # File extensions that are never touched
    BLOCKED_EXTENSIONS = frozenset({
        ".pem", ".key", ".crt", ".cer", ".der",  # crypto keys
        ".p12", ".pfx", ".jks", ".keystore",      # keystores
        ".env.production", ".env.prod",            # production env
        ".gpg", ".asc",                             # encrypted
    })

    # File paths (suffix) that are never touched
    BLOCKED_PATH_PATTERNS = [
        "/etc/passwd", "/etc/shadow", "/etc/sudoers",
        "~/.ssh/", "~/.gnupg/",
        ".git/config", ".git/hooks/",
        "/proc/", "/sys/", "/dev/",
        "credentials", "secrets", ".env.prod", ".env.production",
        "id_rsa", "id_ed25519", "id_ecdsa",
    ]

    # Commands that are always blocked regardless of context
    BLOCKED_COMMANDS = frozenset({
        "sudo", "su", "passwd",
        "shutdown", "reboot", "halt", "poweroff",
        "mkfs", "fdisk", "dd",
        "iptables", "ufw", "firewall-cmd",
        "wget", "curl",  # blocked by default — use web_fetch / web_search tools instead
    })

    # IP ranges / hosts that should not be contacted
    BLOCKED_HOST_PATTERNS = [
        "127.0.0.1", "::1", "localhost",
        "169.254.", "0.0.0.0",
    ]

    def check_tool_call(self, tool_name: str,
                        tool_input: dict) -> SecurityReport | None:
        """Return a SecurityReport if the tool call should be BLOCKED."""
        report = SecurityReport(
            action_type="tool_call",
            tool_name=tool_name,
            layer_reached=1,
        )

        # Extension check
        for key in ("file_path", "path", "file", "target"):
            if key in tool_input:
                val = str(tool_input[key])
                if any(val.endswith(ext) for ext in self.BLOCKED_EXTENSIONS):
                    report.verdict = SecurityVerdict.BLOCK
                    report.reason = f"Blocked extension in '{key}={val}'"
                    report.risk_score = 1.0
                    return report

        # Path pattern check
        for key in ("file_path", "path", "file", "target"):
            if key in tool_input:
                val = str(tool_input[key])
                for pat in self.BLOCKED_PATH_PATTERNS:
                    if pat in val:
                        report.verdict = SecurityVerdict.BLOCK
                        report.reason = f"Blocked path pattern '{pat}' in '{val}'"
                        report.risk_score = 1.0
                        return report

        # Command check
        command = str(tool_input.get("command", "")).lower()
        if command in self.BLOCKED_COMMANDS:
            report.verdict = SecurityVerdict.BLOCK
            report.reason = f"Blocked command: {command}"
            report.risk_score = 1.0
            return report

        # Host check
        for key in ("url", "host", "endpoint", "base_url"):
            if key in tool_input:
                val = str(tool_input[key])
                for pat in self.BLOCKED_HOST_PATTERNS:
                    if pat.lower() in val.lower():
                        report.verdict = SecurityVerdict.BLOCK
                        report.reason = f"Blocked host pattern '{pat}' in '{val}'"
                        report.risk_score = 0.9
                        return report

        return None  # Layer 1 passed — continue chain


# ══════════════════════════════════════════════════════════════════
# Layer 2 — Tool Self-Inspection
# ══════════════════════════════════════════════════════════════════

class ToolSelfInspector:
    """Read tool-declared risk metadata and decide escalation level."""

    def inspect(self, tool: ToolDefinition,
                report: SecurityReport) -> SecurityReport:
        """Update *report* based on the tool's self-declared risk profile."""
        report.layer_reached = 2
        report.tool_risk_level = tool.risk_level

        # If the tool itself says "always ask a human"
        if tool.requires_review:
            report.verdict = SecurityVerdict.ESCALATE
            report.reason = f"Tool '{tool.name}' requires human review"
            report.warnings.extend(tool.safety_hints)
            return report

        # Map risk level to verdict
        level_map: dict[RiskLevel, tuple[SecurityVerdict, float]] = {
            RiskLevel.SAFE:     (SecurityVerdict.ALLOW, 0.0),
            RiskLevel.LOW:      (SecurityVerdict.ALLOW, 0.15),
            RiskLevel.MEDIUM:   (SecurityVerdict.WARN,  0.4),
            RiskLevel.HIGH:     (SecurityVerdict.ESCALATE, 0.7),
            RiskLevel.CRITICAL: (SecurityVerdict.ESCALATE, 0.95),
        }
        verdict, risk_score = level_map.get(
            tool.risk_level, (SecurityVerdict.ESCALATE, 0.7))
        report.verdict = verdict
        report.risk_score = max(report.risk_score, risk_score)
        report.reason = f"Tool self-declared risk: {tool.risk_level.value}"
        report.warnings = list(tool.safety_hints)

        return report


# ══════════════════════════════════════════════════════════════════
# Layer 3 — AI Risk Classification (Prompt Injection + Output Safety)
# ══════════════════════════════════════════════════════════════════

class AIRiskClassifier:
    """Pattern-based prompt injection detection + output safety classification.

    This is a heuristic classifier — zero LLM cost.  It catches common
    injection patterns, jailbreak attempts, and unsafe output patterns.
    """

    # ── Prompt injection patterns ──────────────────────────────

    INJECTION_PATTERNS = [
        # Override system prompt
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)", re.I),
        re.compile(r"forget\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)", re.I),
        re.compile(r"you\s+are\s+now\s+(a\s+)?(different|new)\s+(AI|assistant|model)", re.I),
        re.compile(r"system\s*prompt\s*[:=]", re.I),
        re.compile(r"new\s+system\s+(prompt|instructions?)", re.I),
        # Role switching / jailbreak
        re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I),
        re.compile(r"act\s+as\s+(if\s+)?(you\s+are|a\s+different)", re.I),
        re.compile(r"DAN\s*(mode|jailbreak)", re.I),
        re.compile(r"角色扮演|越狱|绕过(安全|权限)", re.I),
        # Instruction leakage
        re.compile(r"(print|show|reveal|display|echo)\s+(your|the)\s+(system\s+)?(prompt|instructions?)", re.I),
        re.compile(r"(what|tell\s+me)\s+(is\s+)?(your|the)\s+(system\s+)?prompt", re.I),
        # Tool abuse
        re.compile(r"(run|execute)\s+(sudo|rm\s+-rf|mkfs|dd\s+if)", re.I),
        re.compile(r"(download|curl|wget)\s+.*\|\s*(bash|sh|python)", re.I),
    ]

    # ── Output safety patterns ─────────────────────────────────

    UNSAFE_OUTPUT_PATTERNS = [
        re.compile(r"(password|secret|token|api[_-]?key|private[_-]?key)\s*[:=]\s*\S+", re.I),
        re.compile(r"-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----", re.I),
        re.compile(r"ghp_[a-zA-Z0-9]{36}", re.I),    # GitHub PAT
        re.compile(r"sk-[a-zA-Z0-9]{32,}", re.I),     # OpenAI/Anthropic API key
    ]

    def classify_input(self, user_input: str,
                       report: SecurityReport) -> SecurityReport:
        """Scan *user_input* for prompt injection patterns."""
        report.layer_reached = 3
        if not user_input:
            return report

        for pat in self.INJECTION_PATTERNS:
            if pat.search(user_input):
                report.injection_detected = True
                report.risk_score = max(report.risk_score, 0.85)
                report.warnings.append(
                    f"Prompt injection detected: pattern '{pat.pattern[:60]}' "
                    f"matched in user input"
                )
                if report.risk_score >= 0.9:
                    report.verdict = SecurityVerdict.BLOCK
                    report.reason = "Prompt injection blocked by AI classifier"
                else:
                    report.verdict = SecurityVerdict.ESCALATE
                    report.reason = "Prompt injection suspected — requires human review"
                return report

        return report

    def classify_output(self, tool_output: str,
                        report: SecurityReport) -> SecurityReport:
        """Scan *tool_output* for sensitive data leakage."""
        if not tool_output:
            return report

        for pat in self.UNSAFE_OUTPUT_PATTERNS:
            match = pat.search(tool_output)
            if match:
                report.warnings.append(
                    f"Sensitive data detected in tool output: "
                    f"pattern '{pat.pattern[:50]}'"
                )
                report.risk_score = max(report.risk_score, 0.75)
                # Don't block — just warn so the agent context knows

        return report


# ══════════════════════════════════════════════════════════════════
# Layer 4 — Human Confirmation Gateway (delegates to PermissionManager)
# ══════════════════════════════════════════════════════════════════

class HumanConfirmationGateway:
    """Thin wrapper around the existing PermissionManager prompt mechanism."""

    def __init__(self, permission_manager: Any) -> None:
        self._pm = permission_manager

    def confirm(self, report: SecurityReport,
                tool_input: dict, cwd: str) -> SecurityReport:
        """Route to appropriate PermissionManager method based on action type.

        The PermissionManager itself (permissions.py) handles the actual
        user interaction.  This gateway just translates the security
        report into the right ensure_* call.
        """
        report.layer_reached = 4

        if report.action_type == "tool_call":
            tool_name = report.tool_name
            if tool_name == "run_command":
                command = str(tool_input.get("command", ""))
                args = tool_input.get("args", [])
                try:
                    self._pm.ensure_command(
                        command, args if isinstance(args, list) else [],
                        cwd,
                        force_prompt_reason=report.reason,
                    )
                    report.verdict = SecurityVerdict.ALLOW
                except RuntimeError as exc:
                    report.verdict = SecurityVerdict.BLOCK
                    report.reason = str(exc)

            elif tool_name in ("edit_file", "write_file", "patch_file",
                               "modify_file", "multi_edit"):
                file_path = (
                    tool_input.get("file_path") or
                    tool_input.get("path") or
                    tool_input.get("file") or ""
                )
                if file_path:
                    try:
                        self._pm.ensure_edit(
                            str(file_path),
                            f"Risk: {report.tool_risk_level.value}\n"
                            + "\n".join(report.warnings),
                        )
                        report.verdict = SecurityVerdict.ALLOW
                    except RuntimeError as exc:
                        report.verdict = SecurityVerdict.BLOCK
                        report.reason = str(exc)
        return report


# ══════════════════════════════════════════════════════════════════
# Security Chain Orchestrator
# ══════════════════════════════════════════════════════════════════

class SecurityChain:
    """Orchestrates all four security layers in sequence.

    Usage::

        chain = SecurityChain(permission_manager)
        report = chain.review_tool_call(tool, tool_input, user_input, cwd)
        if report.verdict == SecurityVerdict.BLOCK:
            return ToolResult(ok=False, output=report.reason)
        # otherwise proceed with tool execution
    """

    AUTO_APPROVE_THRESHOLD = RiskLevel.LOW  # auto-approve SAFE and LOW in auto mode

    def __init__(
        self,
        permission_manager: Any | None = None,
        *,
        auto_mode: bool = False,
    ) -> None:
        self._rule_filter = RuleFilter()
        self._tool_inspector = ToolSelfInspector()
        self._ai_classifier = AIRiskClassifier()
        self._human_gateway = (
            HumanConfirmationGateway(permission_manager)
            if permission_manager else None
        )
        self._auto_mode = auto_mode

    def review_tool_call(
        self,
        tool: ToolDefinition,
        tool_input: dict,
        user_input: str,
        cwd: str,
    ) -> SecurityReport:
        """Run the full 4-layer chain for a tool call."""

        report = SecurityReport(
            action_type="tool_call",
            tool_name=tool.name,
        )

        # ── Layer 1: Rule filter ───────────────────────────
        blocked = self._rule_filter.check_tool_call(tool.name, tool_input)
        if blocked is not None:
            logger.warning("Security L1 BLOCK: %s", blocked.reason)
            return blocked

        # ── Layer 2: Tool self-inspection ──────────────────
        report = self._tool_inspector.inspect(tool, report)

        if report.verdict == SecurityVerdict.ALLOW:
            if self._auto_mode:
                report.auto_approved = True
                return report
            # In non-auto mode, SAFE and LOW are auto-approved
            if tool.risk_level in (RiskLevel.SAFE, RiskLevel.LOW):
                report.auto_approved = True
                return report

        # ── Layer 3: AI risk classification ────────────────
        report = self._ai_classifier.classify_input(user_input, report)
        if report.verdict == SecurityVerdict.BLOCK:
            logger.warning("Security L3 BLOCK: %s", report.reason)
            return report

        # ── Layer 4: Human confirmation ────────────────────
        if report.verdict == SecurityVerdict.ESCALATE:
            if self._human_gateway is None:
                report.verdict = SecurityVerdict.BLOCK
                report.reason = "Human confirmation required but no prompt handler available"
                return report
            report = self._human_gateway.confirm(report, tool_input, cwd)

        logger.info(
            "Security chain: %s [L%d] risk=%.2f verdict=%s",
            tool.name, report.layer_reached, report.risk_score,
            report.verdict.value,
        )
        return report

    def review_output(
        self,
        tool_output: str,
        report: SecurityReport,
    ) -> SecurityReport:
        """Post-execution output safety scan (Layer 3 only)."""
        return self._ai_classifier.classify_output(tool_output, report)
