from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UserSessionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str = Field(min_length=1, max_length=120)
    conversationId: str | None = Field(default=None, max_length=160)
    analyzedObject: dict[str, Any] | None = None
    createdAt: datetime = Field(default_factory=_utc_now)
    updatedAt: datetime = Field(default_factory=_utc_now)


class SavedProductRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str = Field(min_length=1, max_length=120)
    productId: str = Field(min_length=1, max_length=300)
    title: str = Field(min_length=1, max_length=500)
    retailer: str | None = Field(default=None, max_length=160)
    productUrl: str | None = Field(default=None, max_length=4000)
    imageUrl: str | None = Field(default=None, max_length=4000)
    price: float | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, max_length=8)
    selectedVariant: dict[str, Any] | None = None
    dimensions: dict[str, Any] | None = None
    savedAt: datetime = Field(default_factory=_utc_now)


class FitHistoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str = Field(min_length=1, max_length=120)
    productId: str = Field(min_length=1, max_length=300)
    measuredWidthMeters: float = Field(gt=0)
    measuredDepthMeters: float = Field(gt=0)
    result: Literal["fits", "does_not_fit", "unknown"]
    checkedAt: datetime = Field(default_factory=_utc_now)


class PurchaseIntentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str = Field(min_length=1, max_length=120)
    intentId: str = Field(min_length=1, max_length=200)
    productId: str = Field(min_length=1, max_length=300)
    merchant: str = Field(min_length=1, max_length=160)
    merchantUrl: str = Field(min_length=1, max_length=4000)
    amount: float = Field(ge=0)
    currency: str = Field(min_length=1, max_length=8)
    quantity: int = Field(ge=1, le=10)
    status: str = Field(min_length=1, max_length=40)
    trustedAgentStatus: str = Field(min_length=1, max_length=80)
    commerceProvider: str = Field(min_length=1, max_length=120)
    isSimulation: bool = True
    updatedAt: datetime = Field(default_factory=_utc_now)


class MemoryWriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=1000)
    kind: Literal["preference", "favorite", "goal", "context"] = "context"

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        return " ".join(value.split())


class MemorySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str = Field(min_length=1, max_length=120)
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)


class MemoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    kind: str = "context"
    score: float | None = None


class VoiceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=1000)
    enabled: bool = True

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return " ".join(value.split())


class IntegrationWriteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    provider: str
    persistedRemotely: bool


class MemorySearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    persistedRemotely: bool
    memories: list[MemoryItem]


class VoiceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voiceAvailable: bool
    provider: str
    fallbackText: str
    audioBase64: str | None = None
    contentType: str | None = None
    audioByteCount: int | None = Field(default=None, ge=0)
    reason: str | None = None


class IntegrationStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    persistence: dict[str, str | bool]
    memory: dict[str, str | bool]
    voice: dict[str, str | bool]
    publicBaseUrl: str | None = None
