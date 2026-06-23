"""Two-stage skill routing engine.

Stage 1 (Recall): intent-keyword + tag matching → top-K candidates.
Stage 2 (Ranking):  boundary checks + example similarity → final order.

Includes language-aware tokenization that produces character bigrams for
CJK text and word unigrams+bigrams for English, allowing keyword-based
matching without any NLP dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from minicode.skills import LoadedSkill

logger = logging.getLogger("skill_router")

# ── English stopwords filtered during tokenization ──────────────
_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "not", "no", "yes", "it", "its", "this",
    "that", "can", "you", "i", "me", "my", "we", "our", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "has", "have", "had", "am", "very", "just", "how", "what",
    "when", "where", "which", "who", "why", "if", "then", "else",
}

# ── CJK Unicode blocks ──────────────────────────────────────────
_CJK_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x20000, 0x2A6DF), # CJK Extension B
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F), # CJK Compatibility Supplement
]


def _is_cjk(char: str) -> bool:
    cp = ord(char)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


@dataclass(slots=True)
class _SkillIndex:
    """Pre-computed token sets for fast scoring."""
    name: str
    skill: LoadedSkill
    intent_tokens: set[str] = field(default_factory=set)
    tag_tokens: set[str] = field(default_factory=set)
    example_token_sets: list[set[str]] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
# Tokenization
# ══════════════════════════════════════════════════════════════════

def tokenize(text: str) -> set[str]:
    """Language-aware tokenizer.

    - CJK characters → character unigrams + adjacent bigrams.
    - ASCII/English → lowercased word unigrams + character bigrams.
    - Stopwords removed.
    """
    tokens: set[str] = set()
    if not text:
        return tokens

    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if _is_cjk(ch):
            tokens.add(ch)  # unigram
            if i + 1 < n and _is_cjk(text[i + 1]):
                tokens.add(ch + text[i + 1])  # bigram
            i += 1
            continue

        if ch.isalpha():
            start = i
            while i < n and text[i].isalpha() and not _is_cjk(text[i]):
                i += 1
            word = text[start:i].lower()
            if len(word) >= 2 and word not in _STOPWORDS:
                tokens.add(word)
                # character bigrams
                for j in range(len(word) - 1):
                    tokens.add(word[j:j + 2])
            elif len(word) == 1 and word not in _STOPWORDS and word != "a" and word != "i":
                tokens.add(word)
            continue

        i += 1

    return tokens - _STOPWORDS


# ══════════════════════════════════════════════════════════════════
# Router
# ══════════════════════════════════════════════════════════════════

@dataclass
class _RankedSkill:
    skill: LoadedSkill
    score: float = 0.0


class SkillRouter:
    """Two-stage skill routing engine.

    Parameters
    ----------
    skills: list[LoadedSkill]
        All discovered skills (enriched, with frontmatter when available).
    """

    def __init__(self, skills: list[LoadedSkill]) -> None:
        self._skills = skills
        self._index: list[_SkillIndex] = []
        self._build_index()

    # ── index ────────────────────────────────────────────────────

    def _build_index(self) -> None:
        for skill in self._skills:
            intent_tokens: set[str] = set()
            tag_tokens: set[str] = set()
            example_sets: list[set[str]] = []

            if skill.frontmatter:
                for intent in skill.frontmatter.intents:
                    intent_tokens |= tokenize(intent)
                for tag in skill.frontmatter.tags:
                    tag_tokens |= tokenize(tag)
                for ex in skill.frontmatter.input_examples:
                    example_sets.append(tokenize(ex))

            self._index.append(_SkillIndex(
                name=skill.name,
                skill=skill,
                intent_tokens=intent_tokens,
                tag_tokens=tag_tokens,
                example_token_sets=example_sets,
            ))

    # ── public API ───────────────────────────────────────────────

    def route(
        self,
        user_input: str,
        project_context: dict | None = None,
        *,
        top_k: int = 5,
        confidence_threshold: float = 0.3,
    ) -> tuple[list[LoadedSkill], float]:
        """Route *user_input* to the most relevant skills.

        Returns
        -------
        (ranked_skills, confidence)
            *ranked_skills* is ordered by relevance (best first).
            *confidence* is a ratio where > 1.5 = strong match,
            < 0.3 triggers degradation.
        """
        if not self._index:
            return [], 0.0

        user_tokens = tokenize(user_input)
        if not user_tokens:
            return [idx.skill for idx in self._index], 0.0

        # ── Stage 1: Recall ──────────────────────────────────
        candidates = self._recall(user_tokens, top_k)

        if not candidates:
            return [idx.skill for idx in self._index], 0.0

        # ── Stage 2: Ranking ─────────────────────────────────
        ranked = self._rank(candidates, user_input, user_tokens, project_context or {})

        # ── Confidence ───────────────────────────────────────
        scores = [s for _, s in ranked]
        max_score = scores[0] if scores else 0.0
        rest = scores[1:] if len(scores) > 1 else [0.0]
        avg_rest = sum(rest) / max(len(rest), 1)
        confidence = max_score / (avg_rest + 0.01)

        ranked_skills = [idx.skill for idx, _ in ranked]

        if confidence >= confidence_threshold:
            return ranked_skills, confidence

        # Degradation: append remaining skills after ranked ones
        ranked_names = {s.name for s in ranked_skills}
        remaining = [idx.skill for idx in self._index if idx.name not in ranked_names]
        return ranked_skills + remaining, confidence

    # ── Stage 1 ──────────────────────────────────────────────────

    def _recall(
        self, user_tokens: set[str], top_k: int,
    ) -> list[tuple[_SkillIndex, float]]:
        scored: list[tuple[_SkillIndex, float]] = []

        for idx in self._index:
            intent_score = 0.0
            if idx.intent_tokens:
                overlap = len(user_tokens & idx.intent_tokens)
                intent_score = overlap / max(len(idx.intent_tokens), 1)

            tag_score = 0.0
            if idx.tag_tokens:
                overlap = len(user_tokens & idx.tag_tokens)
                tag_score = overlap / max(len(idx.tag_tokens), 1)

            recall_score = 0.6 * intent_score + 0.4 * tag_score

            # Priority bonus
            priority = 1
            if idx.skill.frontmatter:
                priority = idx.skill.frontmatter.priority
            recall_score *= (1.0 + 0.1 * max(priority - 1, 0))

            if recall_score > 0.0001:
                scored.append((idx, recall_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ── Stage 2 ──────────────────────────────────────────────────

    def _rank(
        self,
        candidates: list[tuple[_SkillIndex, float]],
        user_input: str,
        user_tokens: set[str],
        context: dict,
    ) -> list[tuple[_SkillIndex, float]]:
        scored: list[tuple[_SkillIndex, float]] = []

        for idx, recall_score in candidates:
            final = recall_score

            # boundary check
            if idx.skill.frontmatter and idx.skill.frontmatter.boundary:
                b = idx.skill.frontmatter.boundary

                # disqualify if any cannot_use condition holds
                disqualified = False
                for cond in b.cannot_use:
                    if _eval_boundary_condition(cond, context):
                        disqualified = True
                        break

                if disqualified:
                    continue  # excluded

                if b.can_use:
                    matched = sum(
                        1 for c in b.can_use
                        if _eval_boundary_condition(c, context)
                    )
                    if matched == len(b.can_use):
                        final += 0.2  # full match bonus
                    elif matched > 0:
                        final *= 0.5  # partial match penalty
                    # else: no match, no penalty

            # example similarity
            if idx.example_token_sets:
                best_example = 0.0
                for ex_tokens in idx.example_token_sets:
                    if not ex_tokens:
                        continue
                    overlap = len(user_tokens & ex_tokens)
                    ratio = overlap / max(len(ex_tokens), 1)
                    if ratio > best_example:
                        best_example = ratio
                final += 0.3 * best_example

            scored.append((idx, final))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ══════════════════════════════════════════════════════════════════
# Boundary condition evaluator
# ══════════════════════════════════════════════════════════════════

def _eval_boundary_condition(condition: str, context: dict) -> bool:
    """Evaluate a simple boundary condition against *context*."""
    if not condition:
        return False

    c = condition.strip()

    if c == "is_python_project":
        return context.get("is_python_project", False)

    if c == "is_git_repo":
        return context.get("is_git_repo", False)

    if c == "file_exists_in_workspace":
        return True  # always positive — used as a pre-condition, not a filter

    if c == "is_binary_file":
        return context.get("is_binary_file", False)

    # Unknown condition → assume satisfied (don't block)
    logger.debug("Unknown boundary condition: %s", c)
    return False
