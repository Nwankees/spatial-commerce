from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import VisualProductAnalysis

DimensionsStatus = Literal["complete", "partial", "unavailable"]
ResultSource = Literal["live", "cache"]


class ProductDimensions(BaseModel):
    """Physical product dimensions, only ever populated from provider data.

    Values are never estimated. ``status`` is derived from which values are
    present so Milestone 5 can distinguish complete, partial, and unavailable
    dimensions without re-inspecting the numbers.
    """

    model_config = ConfigDict(extra="forbid")

    widthMeters: float | None = Field(default=None, gt=0)
    depthMeters: float | None = Field(default=None, gt=0)
    heightMeters: float | None = Field(default=None, gt=0)
    status: DimensionsStatus = "unavailable"
    source: str | None = Field(
        default=None,
        description="Where the dimensions came from; null when unavailable.",
    )

    @model_validator(mode="after")
    def derive_status(self) -> ProductDimensions:
        known = [
            value
            for value in (self.widthMeters, self.depthMeters, self.heightMeters)
            if value is not None
        ]
        if len(known) == 3:
            self.status = "complete"
        elif known:
            self.status = "partial"
        else:
            self.status = "unavailable"
            self.source = None
        return self


class ProductCandidate(BaseModel):
    """A real purchasable product returned by a search provider."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Stable candidate key: '<provider>:<providerProductId>'.")
    provider: str = Field(description="Search provider that returned this result, e.g. serpapi.")
    providerProductId: str
    position: int | None = Field(default=None, description="Provider ranking position.")
    title: str
    price: float = Field(ge=0)
    priceText: str | None = Field(default=None, description="Price exactly as the provider formatted it.")
    currency: str | None = Field(default=None, description="ISO 4217 code when unambiguous; otherwise null.")
    retailer: str | None = Field(default=None, description="Merchant selling the product, e.g. Wayfair.")
    imageUrl: str | None = None
    productUrl: str
    rating: float | None = Field(default=None, ge=0, le=5)
    reviewCount: int | None = Field(default=None, ge=0)
    dimensions: ProductDimensions = Field(default_factory=ProductDimensions)
    # Milestone 5.5 retrieval provenance.
    matchedQueries: list[str] = Field(default_factory=list, description="Searches that returned this product.")
    retrievalScore: float | None = Field(default=None, description="Deterministic rerank score (higher is better).")
    # Provider reference for fetching product details later (Milestone 5).
    # Kept server-side: excluded from API responses.
    detailPageToken: str | None = Field(default=None, exclude=True, max_length=4000)


class ProductSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: VisualProductAnalysis
    maxResults: int = Field(default=5, ge=1, le=10)


QueryStatus = Literal["ok", "empty", "failed"]


class QueryOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    status: QueryStatus
    resultCount: int = 0
    error: str | None = None


class ProductSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="The most specific query executed (kept for Milestone 4 clients).")
    queries: list[QueryOutcome] = Field(default_factory=list, description="Every search executed for this request.")
    provider: str
    resultSource: ResultSource
    cachedAt: datetime | None = Field(
        default=None,
        description="When a cached result was originally retrieved live; null for live results.",
    )
    products: list[ProductCandidate]
    message: str | None = None
