from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DimensionStatus = Literal["verified", "partial", "unavailable"]
DimensionSourceType = Literal[
    "json_ld", "structured_metadata", "spec_table", "embedded_json", "page_text_llm", "page_text", "unavailable"
]


class AxisMapping(BaseModel):
    """Which of ``dimensionsMeters`` is width / depth / height. Indices are set only
    when the source itself labels the axis; otherwise they stay null (confidence 0)."""

    model_config = ConfigDict(extra="forbid")

    widthIndex: int | None = Field(default=None, ge=0, le=2)
    depthIndex: int | None = Field(default=None, ge=0, le=2)
    heightIndex: int | None = Field(default=None, ge=0, le=2)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str | None = None
    source: Literal["labels", "structured_fields", "none"] = "none"


class ResolvedDimensions(BaseModel):
    """Dimensions of one selected product, only from explicit source data.

    ``verified``: three overall dimensions explicitly stated by one source
        (``dimensionsMeters``); the axis order may still be unlabeled (``axisMapping``).
    ``partial``: one or two values explicitly stated; nothing is filled in.
    ``unavailable``: nothing trustworthy was found (see ``message``/``retryable``).
    ``widthMeters``/``depthMeters``/``heightMeters`` are set only for axes the source labels.
    """

    model_config = ConfigDict(extra="forbid")

    productId: str
    widthMeters: float | None = Field(default=None, gt=0)
    depthMeters: float | None = Field(default=None, gt=0)
    heightMeters: float | None = Field(default=None, gt=0)
    dimensionsMeters: list[float] | None = Field(default=None, max_length=3,
                                                 description="All stated values, in source order, in meters.")
    axisMapping: AxisMapping | None = None
    status: DimensionStatus = "unavailable"
    sourceType: DimensionSourceType = "unavailable"
    sourceUrl: str | None = None
    sourceName: str | None = Field(default=None, description="Human-readable source, e.g. a retailer name.")
    rawDimensions: str | None = Field(default=None, description="The source text the values were parsed from.")
    sourcePath: str | None = Field(default=None, description="Where on the page, e.g. 'Specs > Dimensions'.")
    extractionMethod: Literal["structured", "llm"] | None = None
    variantScope: Literal["exact_variant", "exact_variant_page", "product_family", "retailer_page"] | None = Field(
        default=None,
        description="Whether the values describe the exact selected variant or the product family.",
    )
    variantIdentity: str | None = Field(default=None, description="The retailer identity used for an exact-variant result.")
    retryable: bool = False
    message: str | None = None

    @model_validator(mode="after")
    def derive_status(self) -> ResolvedDimensions:
        axes = (self.widthMeters, self.depthMeters, self.heightMeters)
        if self.dimensionsMeters is None and any(v is not None for v in axes):
            # Separately labeled fields (structured W/D/H): values and mapping by construction.
            values, mapping = [], {}
            for name, value in zip(("widthIndex", "depthIndex", "heightIndex"), axes):
                if value is not None:
                    mapping[name] = len(values)
                    values.append(value)
            self.dimensionsMeters = values
            self.axisMapping = AxisMapping(**mapping, confidence=1.0, source="structured_fields",
                                           reason="separately labeled width/depth/height fields")
        if self.dimensionsMeters is not None and any(v <= 0 for v in self.dimensionsMeters):
            raise ValueError("dimensions must be positive")
        known = len(self.dimensionsMeters or [])
        if known == 3:
            self.status = "verified"
        elif known:
            self.status = "partial"
        else:
            self.status = "unavailable"
            self.sourceType = "unavailable"
            self.rawDimensions = None
            self.dimensionsMeters = None
            self.axisMapping = None
            self.sourcePath = None
            self.extractionMethod = None
            self.variantScope = None
            self.variantIdentity = None
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
