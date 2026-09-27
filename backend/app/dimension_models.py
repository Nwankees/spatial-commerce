from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DimensionStatus = Literal["verified", "partial", "unavailable"]
DimensionSourceType = Literal["json_ld", "structured_metadata", "spec_table", "page_text", "unavailable"]


class ResolvedDimensions(BaseModel):
    """Dimensions of one selected product, only from explicit source data.

    ``verified``: width, depth and height all explicitly stated by one source.
    ``partial``: some axes explicitly stated; the rest are null, never filled in.
    ``unavailable``: nothing trustworthy was found (see ``message``/``retryable``).
    """

    model_config = ConfigDict(extra="forbid")

    productId: str
    widthMeters: float | None = Field(default=None, gt=0)
    depthMeters: float | None = Field(default=None, gt=0)
    heightMeters: float | None = Field(default=None, gt=0)
    status: DimensionStatus = "unavailable"
    sourceType: DimensionSourceType = "unavailable"
    sourceUrl: str | None = None
    sourceName: str | None = Field(default=None, description="Human-readable source, e.g. a retailer name.")
    rawDimensions: str | None = Field(default=None, description="The source text the values were parsed from.")
    retryable: bool = False
    message: str | None = None

    @model_validator(mode="after")
    def derive_status(self) -> ResolvedDimensions:
        known = sum(v is not None for v in (self.widthMeters, self.depthMeters, self.heightMeters))
        if known == 3:
            self.status = "verified"
        elif known:
            self.status = "partial"
        else:
            self.status = "unavailable"
            self.sourceType = "unavailable"
            self.rawDimensions = None
        if self.status != "unavailable":
            self.retryable = False
        return self


class DimensionRequest(BaseModel):
    """Identifies a product previously returned by /products/search.

    Only identifiers are accepted; the backend looks the product up in its own
    record of provider results and never trusts client-supplied dimensions or URLs.
    """

    model_config = ConfigDict(extra="forbid")

    productId: str = Field(min_length=1, max_length=200)
    productUrl: str = Field(min_length=1, max_length=2000)
