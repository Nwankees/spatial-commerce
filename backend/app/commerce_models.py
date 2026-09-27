from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


PurchaseStatus = Literal[
    "DRAFT",
    "AWAITING_CONFIRMATION",
    "CONFIRMED",
    "TRUST_VERIFIED",
    "PROCESSING",
    "COMPLETED_SIMULATED",
    "CANCELLED",
    "FAILED",
]


class PurchaseIntent(BaseModel):
    """Server-derived, confirmation-gated purchase state for the demo."""

    model_config = ConfigDict(extra="forbid")

    id: str
    sessionId: str
    shopperId: str
    productId: str
    title: str
    merchant: str
    merchantUrl: str
    checkoutUrl: str
    imageUrl: str | None = None
    price: float = Field(ge=0)
    currency: str
    quantity: int = Field(ge=1, le=10)
    total: float = Field(ge=0)
    selectedVariant: dict[str, str] = Field(default_factory=dict)
    sku: str | None = None
    model: str | None = None
    fitStatus: str | None = None
    status: PurchaseStatus
    requiresConfirmation: bool = True
    confirmationReference: str
    trustedAgentStatus: str = "not_requested"
    commerceProvider: str = "visa_intelligent_commerce_demo"
    isSimulation: bool = True
    createdAt: datetime = Field(default_factory=utc_now)
    confirmedAt: datetime | None = None
    completedAt: datetime | None = None
    failureReason: str | None = None


class TrustedAgentEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    authority: str
    path: str
    signatureInput: str
    signature: str
    intentId: str
    contentDigest: str


class TrustVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verified: bool
    protocol: str = "Visa Trusted Agent Protocol / RFC 9421 public profile"
    visaCertified: bool = False
    keySource: str = "ephemeral_demo"
    keyId: str | None = None
    reason: str | None = None


class MerchantVerificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    envelope: TrustedAgentEnvelope
    intent: PurchaseIntent
