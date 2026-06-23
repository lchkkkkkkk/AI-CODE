from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from minicode.skill_frontmatter import SkillFrontmatter, parse_frontmatter

logger = logging.getLogger("skills")


@dataclass(slots=True)
class SkillSummary:
    name: str
    description: str
    path: str
    source: str
    layer: str  # no default here so LoadedSkill can add non-default fields


@dataclass(slots=True)
class LoadedSkill(SkillSummary):
    content: str = ""
    frontmatter: SkillFrontmatter | None = None


def extract_description(markdown: str, frontmatter: SkillFrontmatter | None = None) -> str:
    """Extract the first non-heading paragraph as description.

    If *frontmatter* carries a description, it takes precedence.
    """
    if frontmatter and frontmatter.description:
        return frontmatter.description
    normalized = markdown.replace("\r\n", "\n")
    paragraphs = [block.strip() for block in normalized.split("\n\n") if block.strip()]
    for block in paragraphs:
        if block.startswith("#"):
            continue
        for line in [part.strip() for part in block.split("\n")]:
            if line and not line.startswith("#"):
                return line.replace("`", "")
    return "No description provided."


def _home_dir() -> Path:
    return Path.home()


def _skill_roots(cwd: str | Path) -> list[tuple[Path, str]]:
    base = Path(cwd)
    home = _home_dir()
    return [
        (base / ".mini-code" / "skills", "project"),
        (home / ".mini-code" / "skills", "user"),
        (base / ".claude" / "skills", "compat_project"),
        (home / ".claude" / "skills", "compat_user"),
    ]


def _list_skill_dirs(root: Path, source: str) -> list[LoadedSkill]:
    if not root.exists():
        return []
    results: list[LoadedSkill] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        skill_path = entry / "SKILL.md"
        if not skill_path.exists():
            continue
        content = skill_path.read_text(encoding="utf-8")
        results.append(
            LoadedSkill(
                name=entry.name,
                description=extract_description(content),
                path=str(skill_path),
                source=source,
                content=content,
            )
        )
    return results


_LAYER_DIRS = ("atomic", "workflow", "domain")


def _scan_hierarchical(root: Path, source: str) -> list[LoadedSkill]:
    """Scan *root* for both flat and layered skill directories.

    Layered structure (new)::

        root/atomic/<name>/SKILL.md     → layer="atomic"
        root/workflow/<name>/SKILL.md   → layer="workflow"
        root/domain/<name>/SKILL.md     → layer="domain"

    Flat structure (legacy)::

        root/<name>/SKILL.md            → layer="unknown"
    """
    if not root.exists():
        return []
    results: list[LoadedSkill] = []

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue

        # ── layered: entry is a layer directory ──
        if entry.name in _LAYER_DIRS:
            layer = entry.name
            for skill_dir in sorted(entry.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_path = skill_dir / "SKILL.md"
                if not skill_path.exists():
                    continue
                loaded = _load_skill_file(skill_path, skill_dir.name, source, layer)
                results.append(loaded)
            continue

        # ── flat legacy: entry IS a skill directory ──
        skill_path = entry / "SKILL.md"
        if not skill_path.exists():
            continue
        loaded = _load_skill_file(skill_path, entry.name, source, "unknown")
        results.append(loaded)

    return results


def _load_skill_file(skill_path: Path, name: str, source: str, layer: str) -> LoadedSkill:
    """Read a SKILL.md file, parse frontmatter, and return a LoadedSkill."""
    content = skill_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(content)
    if fm:
        if fm.name:
            name = fm.name
        if fm.layer and fm.layer != "unknown":
            layer = fm.layer
    desc = extract_description(body if fm else content, fm)
    return LoadedSkill(
        name=name,
        description=desc,
        path=str(skill_path),
        source=source,
        layer=layer,
        content=content,
        frontmatter=fm,
    )


def discover_skills_enriched(cwd: str | Path) -> list[LoadedSkill]:
    """Discover all skills with full metadata (frontmatter, layer, content).

    Returns ``list[LoadedSkill]`` carrying parsed frontmatter, layer
    inferred from directory structure, and the raw markdown body.
    Uses hierarchical scanning for new-style layouts while remaining
    compatible with legacy flat directories.
    """
    by_name: dict[str, LoadedSkill] = {}
    for root, source in _skill_roots(cwd):
        for skill in _scan_hierarchical(root, source):
            by_name.setdefault(skill.name, skill)
    return list(by_name.values())


def discover_skills(cwd: str | Path) -> list[SkillSummary]:
    """Discover skills and return lightweight summaries (backward compat).

    Delegates to :func:`discover_skills_enriched` internally, stripping
    the ``content`` and ``frontmatter`` fields to keep the existing
    API contract.
    """
    enriched = discover_skills_enriched(cwd)
    return [
        SkillSummary(
            name=s.name,
            description=s.description,
            path=s.path,
            source=s.source,
            layer=s.layer,
        )
        for s in enriched
    ]


def load_skill(cwd: str | Path, name: str) -> LoadedSkill | None:
    normalized_name = name.strip()
    if not normalized_name:
        return None
    for root, source in _skill_roots(cwd):
        # Try layered paths first
        for layer_dir in ("",) + _LAYER_DIRS:
            base = root / layer_dir if layer_dir else root
            skill_path = base / normalized_name / "SKILL.md"
            if skill_path.exists():
                content = skill_path.read_text(encoding="utf-8")
                fm, body = parse_frontmatter(content)
                layer = layer_dir if layer_dir else (
                    fm.layer if fm and fm.layer != "unknown" else "unknown"
                )

                desc = extract_description(body if fm else content, fm)
                effective_name = fm.name if fm and fm.name else normalized_name
                return LoadedSkill(
                    name=effective_name,
                    description=desc,
                    path=str(skill_path),
                    source=source,
                    layer=layer,
                    content=content,
                    frontmatter=fm,
                )
    return None


def _managed_skill_root(scope: str, cwd: str | Path) -> Path:
    return (Path(cwd) / ".mini-code" / "skills") if scope == "project" else (_home_dir() / ".mini-code" / "skills")


def install_skill(cwd: str | Path, source_path: str, name: str | None = None, scope: str = "user") -> dict[str, str]:
    source = Path(source_path)
    if not source.is_absolute():
        source = Path(cwd) / source
    if source.is_dir():
        skill_file = source / "SKILL.md"
        inferred_name = source.name
    else:
        skill_file = source if source.name == "SKILL.md" else source / "SKILL.md"
        inferred_name = skill_file.parent.name
    if not skill_file.exists():
        raise RuntimeError(f"No SKILL.md found in {source}")

    skill_name = (name or inferred_name).strip()
    if not skill_name:
        raise RuntimeError("Skill name cannot be empty.")

    target_dir = _managed_skill_root(scope, cwd) / skill_name
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(skill_file, target_dir / "SKILL.md")
    return {"name": skill_name, "targetPath": str(target_dir / "SKILL.md")}


def remove_managed_skill(cwd: str | Path, name: str, scope: str = "user") -> dict[str, object]:
    target_path = _managed_skill_root(scope, cwd) / name
    if not target_path.exists():
        return {"removed": False, "targetPath": str(target_path)}
    shutil.rmtree(target_path)
    return {"removed": True, "targetPath": str(target_path)}

