from __future__ import annotations

import re
from dataclasses import dataclass

from .models import VisualProductAnalysis

_MAX_QUERY_WORDS = 7
_MAX_QUERY_CHARS = 100
_VAGUE_COLORS = {"multicolor", "multicolored", "multi-colored", "various", "mixed", "unknown"}


class QueryBuildError(ValueError):
    pass


@dataclass(frozen=True)
class ProductQuery:
    text: str

    @property
    def cache_key(self) -> str:
        return normalize_query(self.text)


def normalize_query(text: str) -> str:
    cleaned = re.sub(r"[^\w\s&'-]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


class ProductQueryBuilder:
    """Deterministically turns a VisualProductAnalysis into one shopping query.

    The query is "<color> <primary style> <primary material> <product noun>",
    skipping any part that is missing or already present, so it stays specific
    enough to be similar but short enough to keep search recall. Only the first
    style and material are used; stacking every attribute destroys recall.
    Gemini's own searchKeywords are the fallback when no product noun exists.
    Later milestones can extend ``build`` with explicit user constraints.
    """

    def build(self, analysis: VisualProductAnalysis) -> ProductQuery:
        if not analysis.objectDetected:
            raise QueryBuildError("The analysis did not detect a product to search for.")

        noun = _clean(analysis.subcategory) or _clean(analysis.category)
        if not noun:
            for keyword in analysis.searchKeywords:
                keyword = _clean(keyword)
                if keyword:
                    return ProductQuery(_truncate(keyword))
            raise QueryBuildError("The analysis has no product type or search keywords.")

        words: list[str] = []
        color = _clean(analysis.color)
        if color and color not in _VAGUE_COLORS:
            _append_phrase(words, color, noun)
        if analysis.style:
            _append_phrase(words, _clean(analysis.style[0]), noun)
        if analysis.materials:
            _append_phrase(words, _clean(analysis.materials[0]), noun)

        noun_words = noun.split()
        budget = max(0, _MAX_QUERY_WORDS - len(noun_words))
        return ProductQuery(_truncate(" ".join(words[:budget] + noun_words)))


def _clean(value: str | None) -> str:
    return normalize_query(value) if value else ""


def _append_phrase(words: list[str], phrase: str, noun: str) -> None:
    noun_words = set(noun.split())
    for word in phrase.split():
        if word not in noun_words and word not in words:
            words.append(word)


def _truncate(text: str) -> str:
    if len(text) <= _MAX_QUERY_CHARS:
        return text
    return text[:_MAX_QUERY_CHARS].rsplit(" ", 1)[0]
