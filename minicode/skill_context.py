"""Project context inspector for skill boundary checking.

Gathers lightweight project metadata at the start of a conversation turn
so the Skill Router can evaluate ``boundary.can_use`` / ``boundary.cannot_use``
conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ProjectContext:
    """Snapshot of relevant project metadata for boundary evaluation."""
    cwd: str = ""
    is_git_repo: bool = False
    is_python_project: bool = False
    languages: list[str] = field(default_factory=list)


# Markers that signal a project is Python-based
_PYTHON_MARKERS = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "Pipfile",
    "poetry.lock",
)

# Common language indicator files
_LANGUAGE_MARKERS: dict[str, tuple[str, ...]] = {
    "python": _PYTHON_MARKERS,
    "javascript": ("package.json", "tsconfig.json", "jsconfig.json"),
    "typescript": ("tsconfig.json",),
    "rust": ("Cargo.toml",),
    "go": ("go.mod", "go.sum"),
    "java": ("pom.xml", "build.gradle", "build.gradle.kts"),
    "ruby": ("Gemfile",),
    "php": ("composer.json",),
}


def inspect_project_context(cwd: str | Path) -> ProjectContext:
    """Inspect *cwd* for known project markers.

    Returns a ``ProjectContext`` that the router can pass to boundary
    checks without touching the filesystem repeatedly.
    """
    root = Path(cwd).resolve()
    ctx = ProjectContext(cwd=str(root))

    # .git detection
    ctx.is_git_repo = (root / ".git").exists()

    # Language detection
    detected: list[str] = []
    for lang, markers in _LANGUAGE_MARKERS.items():
        if any((root / m).exists() for m in markers):
            detected.append(lang)

    ctx.languages = detected
    ctx.is_python_project = "python" in detected

    return ctx
