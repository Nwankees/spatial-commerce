from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class AnalyzeProductRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    imageBase64: str = Field(min_length=1, max_length=16_000_000)
    mimeType: Literal["image/jpeg", "image/png"] = "image/jpeg"
    rotationDegrees: Literal[0, 90, 180, 270] = 0
    userRequest: str | None = Field(default=None, max_length=500)

    @field_validator("userRequest")
    @classmethod
    def normalize_user_request(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

def _clean_list(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            cleaned.append(normalized)
    return cleaned


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


class Hypothesis(BaseModel):
    """An uncertain identification (e.g. a brand) with the model's confidence."""

    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1, max_length=80)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Literal["visible_text", "logo", "design_resemblance"] = Field(
        description="visible_text/logo when printed on the product; design_resemblance when only the look suggests it."
    )

    @field_validator("value")
    @classmethod
    def clean_value(cls, value: str) -> str:
        return " ".join(value.split())


class VisibleSpecification(BaseModel):
    """A specification printed/visible on the product. Never inferred."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=60)
    value: str = Field(min_length=1, max_length=80)
    evidence: str = Field(min_length=1, max_length=160, description="The visible text or feature that shows it.")


class VisualProductAnalysis(BaseModel):
    """Conservative, typed description of the product in view, built for shopping retrieval.

    Uncertain identifications are expressed as Hypothesis objects with a
    confidence; unknown fields stay null/empty rather than being guessed.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    objectDetected: bool = Field(
        description="Whether a principal, potentially shoppable physical object is clearly visible."
    )
    category: str | None = Field(default=None, max_length=80, description="Broad category, e.g. chair, laptop charger.")
    subcategory: str | None = Field(
        default=None, max_length=80, description="Specific product type, e.g. accent chair, USB-C wall charger."
    )
    brand: Hypothesis | None = Field(default=None, description="Brand only if a logo/text or strong design cue supports it.")
    modelFamily: Hypothesis | None = Field(
        default=None, description="Product line/family (not an exact model number unless printed)."
    )
    visibleText: list[str] = Field(
        default_factory=list, max_length=10, description="Legible text/logos exactly as printed on the product."
    )
    color: str | None = Field(default=None, max_length=60, description="Dominant visible product color.")
    materials: list[str] = Field(default_factory=list, max_length=8, description="Apparent materials only.")
    style: list[str] = Field(default_factory=list, max_length=8, description="Short visual style descriptors.")
    shape: str | None = Field(default=None, max_length=120, description="Form factor / dominant shape.")
    distinctiveFeatures: list[str] = Field(
        default_factory=list, max_length=8, description="Visually distinguishing features useful for finding it."
    )
    visibleSpecifications: list[VisibleSpecification] = Field(
        default_factory=list, max_length=6, description="Only specs printed or plainly visible on the product."
    )
    searchQueries: list[str] = Field(
        default_factory=list,
        max_length=6,
        validation_alias=AliasChoices("searchQueries", "searchKeywords"),
        description="Two to six diverse shopping-search queries, most specific first.",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the product identification overall.")
    uncertaintyNotes: str | None = Field(default=None, max_length=300, description="What is uncertain, briefly.")
    message: str | None = Field(
        default=None, max_length=300, description="Brief explanation only when no suitable product is detected."
    )

    @field_validator("materials", "style", "distinctiveFeatures", "visibleText")
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        return _clean_list(values)

    @field_validator("searchQueries")
    @classmethod
    def clean_queries(cls, values: list[str]) -> list[str]:
        return [query[:120] for query in _clean_list(values)]

    @field_validator("category", "subcategory", "color", "shape", "message", "uncertaintyNotes")
    @classmethod
    def clean_optional_strings(cls, value: str | None) -> str | None:
        return _clean_optional(value)

    @property
    def searchKeywords(self) -> list[str]:
        """Milestone 3/4 name for ``searchQueries``."""
        return self.searchQueries
