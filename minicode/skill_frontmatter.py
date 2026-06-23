"""Minimal YAML frontmatter parser for SKILL.md files.

Extracts metadata from YAML frontmatter blocks delimited by ``---`` at the
top of SKILL.md files.  No external dependencies — the parser handles only
the constrained YAML subset used in skill frontmatter:

- scalar key: value pairs
- nested dicts (indented blocks, depth <= 2)
- inline lists  ``[a, b]``
- block lists   ``- item``
- quoted / unquoted strings
- ``#`` comments (whole-line only)

Returns ``(SkillFrontmatter | None, body_markdown)`` so legacy SKILL.md
files without frontmatter continue to work transparently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger("skill_frontmatter")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SkillBoundary:
    """When a skill can / cannot be used."""
    can_use: list[str] = field(default_factory=list)
    cannot_use: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SkillFrontmatter:
    """Parsed YAML frontmatter from a SKILL.md file."""
    name: str = ""
    description: str = ""
    domain: str = "general"
    layer: str = "unknown"
    tags: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)
    boundary: SkillBoundary = field(default_factory=SkillBoundary)
    input_examples: list[str] = field(default_factory=list)
    version: str = "0.0.0"
    priority: int = 1


# ---------------------------------------------------------------------------
# Tiny YAML subset parser
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    """Remove a whole-line ``#`` comment (only when hash is at the start
    of a value position, not inside a quoted string)."""
    return line


def _parse_scalar(value: str) -> str | int | float | bool:
    """Coerce a raw YAML scalar string to the most appropriate Python type."""
    v = value.strip()
    # quoted strings
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if v == "true" or v == "True":
        return True
    if v == "false" or v == "False":
        return False
    if v == "null" or v == "~":
        return ""
    if v == "":
        return ""
    # integer
    try:
        return int(v)
    except ValueError:
        pass
    # float
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_inline_list(value: str) -> list:
    """Parse ``[a, b, c]`` inline list values."""
    v = value.strip()
    if not (v.startswith("[") and v.endswith("]")):
        return [v]  # not a list, wrap as single-element
    inner = v[1:-1].strip()
    if not inner:
        return []
    items = []
    for item in inner.split(","):
        parsed = _parse_scalar(item.strip().strip('"').strip("'"))
        items.append(parsed if isinstance(parsed, str) else str(parsed))
    return items


def _is_list_item(line: str) -> bool:
    """Check if a line starts with ``- `` (block list item)."""
    return line.startswith("- ") or line.startswith("-  ")


def _indent_level(line: str) -> int:
    """Return the number of leading spaces."""
    return len(line) - len(line.lstrip(" "))


def parse_frontmatter(markdown: str) -> tuple[SkillFrontmatter | None, str]:
    """Extract YAML frontmatter from SKILL.md content.

    Returns
    -------
    (SkillFrontmatter | None, body_md)
        ``None`` when no valid frontmatter block is found (legacy skills).
        The second element is the remaining markdown body.
    """
    if not markdown or not markdown.startswith("---"):
        return None, markdown

    lines = markdown.split("\n")
    if len(lines) < 3:
        return None, markdown

    # Find closing ``---``
    closing_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            closing_idx = i
            break

    if closing_idx is None:
        return None, markdown

    # Extract frontmatter lines and body
    fm_lines = lines[1:closing_idx]
    body = "\n".join(lines[closing_idx + 1:]).lstrip("\n")

    try:
        fm = _parse_frontmatter_lines(fm_lines)
        return fm, body
    except Exception as exc:
        logger.warning("Failed to parse frontmatter: %s", exc)
        return None, markdown


def _parse_frontmatter_lines(lines: list[str]) -> SkillFrontmatter:
    """Parse a list of frontmatter lines into a SkillFrontmatter object.

    Handles scalar pairs, nested ``boundary:`` block, and both inline /
    block list values.
    """
    fm = SkillFrontmatter()
    boundary_can: list[str] = []
    boundary_cannot: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # skip blank lines and comments
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        # --- scalar key: value ---
        if ":" in stripped and not _is_list_item(stripped):
            key_part, _, val_part = stripped.partition(":")
            key = key_part.strip()
            val = val_part.strip()

            if key == "boundary":
                # nested block — consume indented lines
                i += 1
                while i < len(lines):
                    sub = lines[i]
                    sub_stripped = sub.strip()
                    if not sub_stripped or sub_stripped.startswith("#"):
                        i += 1
                        continue
                    if not sub.startswith("  ") and not sub.startswith("\t"):
                        break  # dedented — end of boundary block
                    if ":" in sub_stripped and not _is_list_item(sub_stripped):
                        sub_key, _, sub_val = sub_stripped.partition(":")
                        sub_key = sub_key.strip()
                        sub_val = sub_val.strip()
                        if sub_key in ("can_use", "can-use"):
                            boundary_can.extend(_parse_block_list(lines, i, sub_val))
                        elif sub_key in ("cannot_use", "cannot-use"):
                            boundary_cannot.extend(_parse_block_list(lines, i, sub_val))
                    i += 1
                fm.boundary = SkillBoundary(can_use=boundary_can, cannot_use=boundary_cannot)
                continue

            elif key == "tags":
                fm.tags = _parse_list_value(val, lines, i)

            elif key == "intents":
                fm.intents = _parse_list_value(val, lines, i)

            elif key == "input_examples" or key == "input-examples":
                fm.input_examples = _parse_list_value(val, lines, i)

            elif key == "name":
                fm.name = str(_parse_scalar(val))

            elif key == "description":
                fm.description = str(_parse_scalar(val))

            elif key == "domain":
                fm.domain = str(_parse_scalar(val))

            elif key == "layer":
                fm.layer = str(_parse_scalar(val))

            elif key == "version":
                fm.version = str(_parse_scalar(val))

            elif key == "priority":
                parsed = _parse_scalar(val)
                fm.priority = int(parsed) if isinstance(parsed, (int, float)) else 1

        i += 1

    return fm


def _parse_list_value(val: str, lines: list[str], idx: int) -> list[str]:
    """Parse a value that may be an inline list ``[a, b]`` or block list
    spanning subsequent indented ``- item`` lines."""
    v = val.strip()
    if v.startswith("["):
        items = _parse_inline_list(v)
        return [it for it in items if it]

    # empty — check for block list on following lines
    if not v or v == "|":
        return _parse_block_list(lines, idx, "")

    # single scalar
    return [str(_parse_scalar(v))]


def _parse_block_list(lines: list[str], start_idx: int, _initial_val: str) -> list[str]:
    """Consume indented ``- item`` lines starting from *after* start_idx.

    Stops when it hits: a dedented line, a key-value pair (``key: value``)
    at the same indent level, or end of input.
    """
    items: list[str] = []
    base_indent = _indent_level(lines[start_idx]) if start_idx < len(lines) else 2

    for j in range(start_idx + 1, len(lines)):
        sub = lines[j]
        stripped = sub.strip()

        if not stripped or stripped.startswith("#"):
            continue

        sub_indent = _indent_level(sub)

        # stop at a new key-value pair at same or lower indent
        if ":" in stripped and not _is_list_item(stripped):
            if sub_indent <= base_indent:
                break

        if _is_list_item(stripped):
            item_text = stripped[2:].strip().strip('"').strip("'")
            if item_text:
                items.append(item_text)
        elif sub_indent < base_indent:
            break  # dedented — end of list

    return items
