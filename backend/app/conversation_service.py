from __future__ import annotations

from typing import Callable

from .conversation_models import (
    ClarifyAction,
    ConversationContextRequest,
    ConversationHistoryResponse,
    ConversationMessage,
    ConversationSession,
    ConversationStateView,
    ConversationTurnResponse,
    CreateConversationResponse,
    ResetSessionAction,
    ToolOutcome,
)
from .conversation_planner import (
    ConversationPlanner,
    ConversationResponseWriter,
    DeterministicFallbackPlanner,
    PlannerMalformedActionError,
    PlannerUnavailableError,
)
from .conversation_store import ConversationStore
from .conversation_sponsor import ConversationSponsorBridge
from .conversation_tools import ShoppingToolExecutor
from .product_models import ProductCandidate


class ConversationService:
    def __init__(
        self,
        store: ConversationStore,
        planner: ConversationPlanner,
        tools: ShoppingToolExecutor,
        *,
        fallback_planner: ConversationPlanner | None = None,
        response_writer: ConversationResponseWriter | None = None,
        product_lookup: Callable[[str], ProductCandidate | None] | None = None,
        sponsor_bridge: ConversationSponsorBridge | None = None,
    ) -> None:
        self._store = store
        self._planner = planner
        self._tools = tools
        self._fallback = fallback_planner or DeterministicFallbackPlanner()
        self._response_writer = response_writer
        self._product_lookup = product_lookup
        self._sponsor_bridge = sponsor_bridge

    def create(self) -> CreateConversationResponse:
        session = self._store.create()
        return CreateConversationResponse(
            sessionId=session.id,
            message="Ask me to find, refine, compare, check fit, preview, or prepare a product purchase.",
            state=ConversationStateView.from_session(session),
        )

    def history(self, session_id: str) -> ConversationHistoryResponse:
        session = self._store.get(session_id)
        return ConversationHistoryResponse(
            sessionId=session.id,
            messages=session.messages,
            state=ConversationStateView.from_session(session),
        )

    def sync_context(
        self,
        session_id: str,
        context: ConversationContextRequest,
        *,
        image_bytes: bytes | None = None,
    ) -> ConversationHistoryResponse:
        if context.products is not None and self._product_lookup is not None:
            # Restore the server-side provider references hidden from Android JSON.
            products: list[ProductCandidate] = []
            for supplied in context.products:
                known = self._product_lookup(supplied.id)
                if known is None:
                    raise ValueError(
                        "A conversation product is no longer known to the backend. Search again before selecting it."
                    )
                products.append(known)
            context = context.model_copy(update={"products": products})
            context.__pydantic_fields_set__.add("products")
        session = self._store.sync_context(session_id, context, image_bytes=image_bytes)
        return self.history(session.id)

    def reset(self, session_id: str) -> CreateConversationResponse:
        session = self._store.reset(session_id)
        return CreateConversationResponse(
            sessionId=session.id,
            message="Conversation reset. Your existing camera and button controls are unchanged.",
            state=ConversationStateView.from_session(session),
        )

    async def turn(self, session_id: str, message: str) -> ConversationTurnResponse:
        lock = self._store.lock(session_id)
        async with lock:
            state = self._store.get(session_id)
            try:
                state.messages.append(ConversationMessage(role="user", text=message))
                state.activePhase = "planning"
                state.activeAction = None
                if self._sponsor_bridge is not None:
                    await self._sponsor_bridge.recall(state, message)
                planner_mode = "local_qwen"
                try:
                    action = await self._planner.plan(message, state)
                except (PlannerUnavailableError, PlannerMalformedActionError, ValueError, TypeError):
                    action = await self._fallback.plan(message, state)
                    planner_mode = "deterministic_fallback"

                state.activePhase = "executing"
                state.activeAction = action.action
                if isinstance(action, ResetSessionAction):
                    state = self._store.reset(session_id)
                    outcome = ToolOutcome(
                        status="success",
                        message="Conversation reset. Point at an object or ask me to start shopping again.",
                    )
                    state.activePhase = "executing"
                    state.activeAction = action.action
                elif isinstance(action, ClarifyAction):
                    outcome = ToolOutcome(status="needs_input", message=action.arguments.question)
                else:
                    try:
                        outcome = await self._tools.execute(action, state)
                    except Exception:
                        outcome = ToolOutcome(
                            status="error",
                            message="That shopping action failed safely. Your previous results and selection are still available.",
                        )

                if self._sponsor_bridge is not None and outcome.status == "success":
                    self._sponsor_bridge.record_success(state, action, outcome)

                response_message = outcome.message
                response_writer_mode = "deterministic_fallback"
                if self._response_writer is not None:
                    state.activePhase = "responding"
                    try:
                        response_message = await self._response_writer.respond(message, action, outcome, state)
                        response_writer_mode = "local_qwen"
                    except (PlannerUnavailableError, PlannerMalformedActionError, ValueError, TypeError):
                        # The typed tool result remains the safe, grounded demo fallback.
                        pass

                state.messages.append(ConversationMessage(role="assistant", text=response_message))
                if len(state.messages) > 40:
                    state.messages = state.messages[-40:]
                state.activePhase = None
                state.activeAction = None
                return ConversationTurnResponse(
                    sessionId=state.id,
                    message=response_message,
                    status=outcome.status,
                    action=action,
                    planner=planner_mode,
                    responseWriter=response_writer_mode,
                    products=outcome.products,
                    searchResult=outcome.searchResult,
                    selectedProduct=outcome.selectedProduct,
                    fit=outcome.fit,
                    purchase=outcome.purchase,
                    trustVerification=outcome.trustVerification,
                    checkoutUrl=outcome.checkoutUrl,
                    uiDirective=outcome.uiDirective,
                    state=ConversationStateView.from_session(state),
                )
            finally:
                state.activePhase = None
                state.activeAction = None
