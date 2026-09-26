from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

class VisualProductAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objectDetected: bool = Field(
        description="Whether a principal, potentially shoppable physical object is clearly visible."
    )
    category: str | None = Field(
        default=None,
        description="Broad product category, such as chair or table.",
    )
    subcategory: str | None = Field(
        default=None,
        description="More specific product type, such as accent chair.",
    )
    color: str | None = Field(
        default=None,
        description="Dominant visible product color.",
    )
    materials: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Likely visible materials. Do not invent hidden materials.",
    )
    style: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Short visual style descriptors.",
    )
    shape: str | None = Field(
        default=None,
        description="Concise description of the product's dominant shape.",
    )
    searchKeywords: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Two to six concise shopping-search phrases grounded in the image.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    message: str | None = Field(
        default=None,
        description="Brief explanation only when no suitable product is detected.",
    )

    @field_validator("materials", "style", "searchKeywords")
    @classmethod
    def clean_string_lists(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            normalized = value.strip()
            if normalized and normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned

    @field_validator("category", "subcategory", "color", "shape", "message")
    @classmethod
    def clean_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None
