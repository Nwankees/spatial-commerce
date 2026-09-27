from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import VisualProductAnalysis
from .product_models import ProductCandidate

_STOPWORDS = {
    "a", "an", "and", "the", "with", "for", "of", "in", "on", "to", "by", "or", "set", "new", "pack",
    "inch", "in", "cm", "mm", "x", "w", "d", "h", "modern", "style", "design", "product", "item",
}


def tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    words = re.findall(r"[a-z0-9]+", text.lower())
    result = set()
    for word in words:
        if word in _STOPWORDS or (len(word) < 3 and not word.isdigit()):
            continue
        result.add(word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word)
    return result


@dataclass(frozen=True)
class RerankWeights:
    provider_rank: float = 1.0      # best provider position across queries (1.0 at #1, fading out)
    multi_query: float = 0.5        # per additional query that also returned the product (capped)
    primary_query: float = 0.2      # returned by the most specific (first) query
    brand: float = 2.0              # × brand confidence, when the brand appears in title/retailer
    product_type: float = 1.5       # × share of category/subcategory terms in the title
    identity_text: float = 1.5      # × share of model-family/visible-text terms in the title
    features: float = 1.0           # × share of color/material/feature terms in the title
    lens_exact: float = 2.4         # exact visual match, rank-decayed
    lens_product: float = 1.6       # purchasable Lens result, rank-decayed
    lens_visual: float = 1.2        # appearance-based Lens result, rank-decayed
    cross_generator: float = 1.2    # result independently found by text and Lens
    multi_lens_mode: float = 0.4    # per additional Lens mode (capped)
    purchasable: float = 0.3        # provider supplied a price
    local_same_product: float = 2.4
    local_visual_similarity: float = 2.4
    local_category_match: float = 0.8
    min_brand_confidence: float = 0.5
    max_multi_query_bonus: int = 2


@dataclass
class ScoredCandidate:
    candidate: ProductCandidate
    best_position: int
    queries: list[str]
    retrieval_sources: list[str] = field(default_factory=list)
    text_rank: int | None = None
    visual_rank: int | None = None
    same_product_probability: float | None = None
    visual_similarity: float | None = None
    visual_category_match: float | None = None
    score: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)


class ProductReranker:
    """Explicit, deterministic scoring of merged candidates against the analysis.

    No LLM and no image similarity: only term evidence in provider titles/retailers,
    the analysis' own (confidence-weighted) identifications, and query provenance.
    """

    def __init__(self, weights: RerankWeights | None = None, results_per_query: int = 10) -> None:
        self._w = weights or RerankWeights()
        self._depth = max(1, results_per_query)

    def score(self, analysis: VisualProductAnalysis, item: ScoredCandidate, primary_query: str | None) -> None:
        w = self._w
        title = tokens(item.candidate.title)
        title_and_retailer = title | tokens(item.candidate.retailer)
        parts: dict[str, float] = {}

        parts["provider_rank"] = w.provider_rank * max(0.0, 1.0 - (item.best_position - 1) / self._depth)
        extra_queries = min(len(set(item.queries)) - 1, w.max_multi_query_bonus)
        parts["multi_query"] = w.multi_query * max(0, extra_queries)
        parts["primary_query"] = w.primary_query if primary_query and primary_query in item.queries else 0.0

        brand = analysis.brand
        if brand and brand.confidence >= w.min_brand_confidence:
            brand_terms = tokens(brand.value)
            if brand_terms and brand_terms <= title_and_retailer:
                parts["brand"] = w.brand * brand.confidence
        parts["product_type"] = w.product_type * _share(tokens(analysis.subcategory) | tokens(analysis.category), title)

        identity = tokens(" ".join(analysis.visibleText))
        if analysis.modelFamily:
            identity |= tokens(analysis.modelFamily.value)
        parts["identity_text"] = w.identity_text * _share(identity, title)

        descriptive = tokens(analysis.color) | tokens(" ".join(analysis.materials)) | tokens(
            " ".join(analysis.distinctiveFeatures)
        )
        parts["features"] = w.features * _share(descriptive, title)

        sources = set(item.retrieval_sources)
        lens_ranks = max(0.0, 1.0 - ((item.visual_rank or self._depth) - 1) / self._depth)
        if "lens_exact_match" in sources:
            parts["lens_exact"] = w.lens_exact * lens_ranks
        if "lens_products" in sources:
            parts["lens_product"] = w.lens_product * lens_ranks
        if "lens_visual_match" in sources:
            parts["lens_visual"] = w.lens_visual * lens_ranks
        lens_sources = sources & {"lens_exact_match", "lens_products", "lens_visual_match"}
        if "text_search" in sources and lens_sources:
            parts["cross_generator"] = w.cross_generator
        if len(lens_sources) > 1:
            parts["multi_lens_mode"] = w.multi_lens_mode * min(2, len(lens_sources) - 1)
        if item.candidate.price is not None or item.candidate.inStock is True:
            parts["purchasable"] = w.purchasable
        if item.same_product_probability is not None:
            parts["local_same_product"] = w.local_same_product * item.same_product_probability
        if item.visual_similarity is not None:
            parts["local_visual_similarity"] = w.local_visual_similarity * item.visual_similarity
        if item.visual_category_match is not None:
            parts["local_category_match"] = w.local_category_match * item.visual_category_match

        item.breakdown = {k: round(v, 4) for k, v in parts.items() if v}
        item.score = round(sum(parts.values()), 4)

    def rerank(
        self, analysis: VisualProductAnalysis, items: list[ScoredCandidate], primary_query: str | None
    ) -> list[ScoredCandidate]:
        for item in items:
            self.score(analysis, item, primary_query)
        return sorted(items, key=lambda i: (-i.score, i.best_position, i.candidate.title))


def _share(wanted: set[str], present: set[str]) -> float:
    if not wanted:
        return 0.0
    return len(wanted & present) / len(wanted)
