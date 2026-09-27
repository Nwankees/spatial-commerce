from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from .models import VisualProductAnalysis
from .product_models import ProductCandidate, ProductSearchResponse
from .commerce_models import PurchaseIntent, TrustVerification


ConversationActionName = Literal[
    "find_similar_products",
    "refine_search",
    "select_product",
    "check_fit",
    "request_ar_preview",
    "get_product_details",
    "compare_products",
    "prepare_purchase",
    "confirm_purchase",
    "cancel_purchase",
    "get_purchase_status",
    "open_checkout",
    "reset_session",
    "clarify",
]


class SearchConstraints(BaseModel):
    """Shopping constraints the conversation has accumulated."""

    model_config = ConfigDict(extra="forbid")

    maxPrice: float | None = Field(default=None, ge=0)
    minPrice: float | None = Field(default=None, ge=0)
    color: str | None = Field(default=None, max_length=60)
    material: str | None = Field(default=None, max_length=60)
    retailer: str | None = Field(default=None, max_length=100)
    preferVisualSimilarity: bool = False

    @field_validator("color", "material", "retailer")
    @classmethod
    def clean_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        return cleaned or None

    def merged(self, other: SearchConstraints) -> SearchConstraints:
        values = self.model_dump()
        for key, value in other.model_dump().items():
            if value is not None and (not isinstance(value, bool) or value):
                values[key] = value
        return SearchConstraints.model_validate(values)


class MeasuredSpace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    widthMeters: float = Field(gt=0, le=100)
    depthMeters: float = Field(gt=0, le=100)


class FitAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    productId: str
    verdict: Literal["fits", "does_not_fit", "unknown", "needs_measurement"]
    productWidthMeters: float | None = Field(default=None, gt=0)
    productDepthMeters: float | None = Field(default=None, gt=0)
    availableWidthMeters: float | None = Field(default=None, gt=0)
    availableDepthMeters: float | None = Field(default=None, gt=0)
    rotated: bool | None = None
    sourceUrl: str | None = None


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    text: str = Field(min_length=1, max_length=1200)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationSession(BaseModel):
    """In-memory M7 state. No cloud persistence is required for this milestone."""

    model_config = ConfigDict(extra="forbid")

    id: str
    # A stable, pseudonymous device identity may be supplied by Android so
    # optional sponsor storage can span process-local conversation IDs.
    shopperId: str | None = Field(default=None, min_length=1, max_length=120)
    analyzedObject: VisualProductAnalysis | None = None
    # Prepared JPEG bytes stay backend-only. They are never serialized into the
    # planner prompt, conversation history, or an Android response.
    analyzedImageBytes: bytes | None = Field(default=None, exclude=True, repr=False)
    constraints: SearchConstraints = Field(default_factory=SearchConstraints)
    latestResults: list[ProductCandidate] = Field(default_factory=list, max_length=10)
    latestSearch: ProductSearchResponse | None = Field(default=None, exclude=True, repr=False)
    selectedProductId: str | None = None
    measuredSpace: MeasuredSpace | None = None
    latestFit: FitAssessment | None = None
    latestArPreviewProductId: str | None = None
    pendingPurchase: PurchaseIntent | None = None
    lastPurchase: PurchaseIntent | None = None
    # Backboard recall is planner-only context. It is deliberately excluded
    # from API serialization and is never treated as an instruction.
    rememberedPreferences: list[str] = Field(
        default_factory=list,
        max_length=5,
        exclude=True,
        repr=False,
    )
    activePhase: Literal["planning", "executing", "responding"] | None = None
    activeAction: ConversationActionName | None = None
    messages: list[ConversationMessage] = Field(default_factory=list, max_length=40)

    def result_ids(self) -> list[str]:
        return [product.id for product in self.latestResults]

    def selected_product(self) -> ProductCandidate | None:
        return next((p for p in self.latestResults if p.id == self.selectedProductId), None)


class ConversationStateView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shopperId: str | None = None
    analyzedObject: VisualProductAnalysis | None = None
    analyzedImageAvailable: bool = False
    constraints: SearchConstraints
    latestResultIds: list[str]
    selectedProductId: str | None = None
    measuredSpace: MeasuredSpace | None = None
    latestFit: FitAssessment | None = None
    latestArPreviewProductId: str | None = None
    purchase: PurchaseIntent | None = None
    activePhase: Literal["planning", "executing", "responding"] | None = None
    activeAction: ConversationActionName | None = None

    @classmethod
    def from_session(cls, session: ConversationSession) -> ConversationStateView:
        return cls(
            shopperId=session.shopperId,
            analyzedObject=session.analyzedObject,
            analyzedImageAvailable=session.analyzedImageBytes is not None,
            constraints=session.constraints,
            latestResultIds=session.result_ids(),
            selectedProductId=session.selectedProductId,
            measuredSpace=session.measuredSpace,
            latestFit=session.latestFit,
            latestArPreviewProductId=session.latestArPreviewProductId,
            purchase=session.pendingPurchase or session.lastPurchase,
            activePhase=session.activePhase,
            activeAction=session.activeAction,
        )


class CreateConversationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str
    message: str
    state: ConversationStateView


class ConversationContextRequest(BaseModel):
    """Optional state supplied by the existing button UI before a chat turn."""

    model_config = ConfigDict(extra="forbid")

    shopperId: str | None = Field(default=None, min_length=1, max_length=120)
    analysis: VisualProductAnalysis | None = None
    imageBase64: str | None = Field(default=None, max_length=16_000_000)
    mimeType: Literal["image/jpeg", "image/png"] = "image/jpeg"
    rotationDegrees: Literal[0, 90, 180, 270] = 0
    products: list[ProductCandidate] | None = Field(default=None, max_length=10)
    selectedProductId: str | None = Field(default=None, max_length=200)
    measuredSpace: MeasuredSpace | None = None


class ConversationTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=500)

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("Message cannot be blank.")
        return cleaned


class ProductReferenceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    productId: str | None = Field(default=None, max_length=200)
    resultNumber: int | None = Field(default=None, ge=1, le=10)


class FindSimilarArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maxResults: int = Field(default=5, ge=1, le=10)


class RefineSearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constraints: SearchConstraints = Field(
        default_factory=SearchConstraints,
        description="Only explicit constraints stated by the user; leave every other field null/false.",
    )
    relativePrice: Literal["none", "cheaper"] = Field(
        default="none",
        description="Use cheaper when the user says cheaper or less expensive; otherwise none.",
    )
    sizePreference: Literal["none", "smaller"] = Field(
        default="none",
        description="Use smaller when the user requests smaller products rather than a comparison; otherwise none.",
    )


class CompareProductsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resultNumbers: list[int] = Field(default_factory=list, max_length=10)


class PreparePurchaseArgs(ProductReferenceArgs):
    quantity: int = Field(default=1, ge=1, le=10)


class ConfirmPurchaseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intentId: str | None = Field(default=None, max_length=200)
    confirmationReference: str | None = Field(default=None, max_length=200)


class ClarifyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=240)


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FindSimilarAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["find_similar_products"]
    arguments: FindSimilarArgs = Field(default_factory=FindSimilarArgs)


class RefineSearchAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["refine_search"]
    arguments: RefineSearchArgs


class SelectProductAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["select_product"]
    arguments: ProductReferenceArgs


class CheckFitAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["check_fit"]
    arguments: ProductReferenceArgs = Field(default_factory=ProductReferenceArgs)


class RequestArPreviewAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["request_ar_preview"]
    arguments: ProductReferenceArgs = Field(default_factory=ProductReferenceArgs)


class GetProductDetailsAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["get_product_details"]
    arguments: ProductReferenceArgs = Field(default_factory=ProductReferenceArgs)


class CompareProductsAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["compare_products"]
    arguments: CompareProductsArgs = Field(default_factory=CompareProductsArgs)


class PreparePurchaseAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["prepare_purchase"]
    arguments: PreparePurchaseArgs = Field(default_factory=PreparePurchaseArgs)


class ConfirmPurchaseAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["confirm_purchase"]
    arguments: ConfirmPurchaseArgs = Field(default_factory=ConfirmPurchaseArgs)


class CancelPurchaseAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["cancel_purchase"]
    arguments: EmptyArgs = Field(default_factory=EmptyArgs)


class GetPurchaseStatusAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["get_purchase_status"]
    arguments: EmptyArgs = Field(default_factory=EmptyArgs)


class OpenCheckoutAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["open_checkout"]
    arguments: EmptyArgs = Field(default_factory=EmptyArgs)


class ResetSessionAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["reset_session"]
    arguments: EmptyArgs = Field(default_factory=EmptyArgs)


class ClarifyAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["clarify"]
    arguments: ClarifyArgs


AgentAction = Annotated[
    Union[
        FindSimilarAction,
        RefineSearchAction,
        SelectProductAction,
        CheckFitAction,
        RequestArPreviewAction,
        GetProductDetailsAction,
        CompareProductsAction,
        PreparePurchaseAction,
        ConfirmPurchaseAction,
        CancelPurchaseAction,
        GetPurchaseStatusAction,
        OpenCheckoutAction,
        ResetSessionAction,
        ClarifyAction,
    ],
    Field(discriminator="action"),
]
AGENT_ACTION_ADAPTER = TypeAdapter(AgentAction)


class ToolOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "needs_input", "unavailable", "error"]
    message: str = Field(min_length=1, max_length=1000)
    products: list[ProductCandidate] = Field(default_factory=list, max_length=10)
    searchResult: ProductSearchResponse | None = None
    selectedProduct: ProductCandidate | None = None
    fit: FitAssessment | None = None
    purchase: PurchaseIntent | None = None
    trustVerification: TrustVerification | None = None
    checkoutUrl: str | None = None
    uiDirective: Literal[
        "none", "analyze_object", "show_products", "select_product",
        "measure_space", "show_fit", "enter_ar_preview", "review_purchase",
        "purchase_complete", "open_checkout",
    ] = "none"


class ConversationTurnResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str
    message: str
    status: Literal["success", "needs_input", "unavailable", "error"]
    action: AgentAction
    planner: Literal["local_qwen", "deterministic_fallback"]
    responseWriter: Literal["local_qwen", "deterministic_fallback"]
    products: list[ProductCandidate] = Field(default_factory=list)
    searchResult: ProductSearchResponse | None = None
    selectedProduct: ProductCandidate | None = None
    fit: FitAssessment | None = None
    purchase: PurchaseIntent | None = None
    trustVerification: TrustVerification | None = None
    checkoutUrl: str | None = None
    uiDirective: str = "none"
    state: ConversationStateView


class ConversationHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sessionId: str
    messages: list[ConversationMessage]
    state: ConversationStateView
