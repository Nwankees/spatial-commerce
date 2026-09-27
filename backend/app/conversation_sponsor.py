from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .conversation_models import (
    AgentAction,
    ConversationSession,
    RefineSearchAction,
    ToolOutcome,
)
from .integrations.models import FitHistoryRecord, PurchaseIntentRecord, SavedProductRecord, UserSessionRecord
from .integrations.service import SponsorIntegrationService

logger = logging.getLogger("app.conversation.sponsor")


class ConversationSponsorBridge:
    """Optional, fail-open sponsor side effects around the local Qwen agent.

    Conversation state remains process-local and authoritative. Backboard recall
    is a small hint supplied before planning; Atlas/Backboard writes run in the
    background after a typed action succeeds.
    """

    def __init__(
        self,
        service: SponsorIntegrationService,
        *,
        recall_timeout_seconds: float = 1.25,
        memory_limit: int = 5,
    ) -> None:
        self._service = service
        self._recall_timeout = recall_timeout_seconds
        self._memory_limit = memory_limit
        self._pending: set[asyncio.Task[None]] = set()

    async def recall(self, state: ConversationSession, query: str) -> None:
        """Populate bounded non-authoritative context without blocking chat."""

        state.rememberedPreferences = []
        try:
            response = await asyncio.wait_for(
                self._service.search_memory(self._identity(state), query, self._memory_limit),
                timeout=self._recall_timeout,
            )
        except Exception as exc:
            logger.info("Sponsor memory recall failed open: %s", type(exc).__name__)
            return

        state.rememberedPreferences = [
            " ".join(item.content.split())[:240]
            for item in response.memories[: self._memory_limit]
            if item.content.strip()
        ]

    def record_success(
        self,
        state: ConversationSession,
        action: AgentAction,
        outcome: ToolOutcome,
    ) -> None:
        """Schedule optional persistence; never delay or alter the tool result."""

        task = asyncio.create_task(self._record(state.model_copy(deep=True), action, outcome.model_copy(deep=True)))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def wait_pending(self) -> None:
        """Drain scheduled writes for graceful shutdown and focused tests."""

        pending = list(self._pending)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _record(
        self,
        state: ConversationSession,
        action: AgentAction,
        outcome: ToolOutcome,
    ) -> None:
        identity = self._identity(state)
        await self._fail_open(
            "session",
            self._service.save_session(
                UserSessionRecord(
                    sessionId=identity,
                    conversationId=state.id,
                    analyzedObject=(
                        state.analyzedObject.model_dump(mode="json")
                        if state.analyzedObject is not None
                        else None
                    ),
                )
            ),
        )

        selected = outcome.selectedProduct or state.selected_product()
        if selected is not None:
            await self._fail_open(
                "product",
                self._service.save_product(
                    SavedProductRecord(
                        sessionId=identity,
                        productId=selected.id,
                        title=selected.title,
                        retailer=selected.retailer,
                        productUrl=selected.productUrl,
                        imageUrl=selected.imageUrl,
                        price=selected.price,
                        currency=selected.currency,
                        dimensions=selected.dimensions.model_dump(mode="json"),
                    )
                ),
            )

        fit = outcome.fit
        if (
            fit is not None
            and fit.verdict in {"fits", "does_not_fit", "unknown"}
            and state.measuredSpace is not None
        ):
            await self._fail_open(
                "fit",
                self._service.save_fit(
                    FitHistoryRecord(
                        sessionId=identity,
                        productId=fit.productId,
                        measuredWidthMeters=state.measuredSpace.widthMeters,
                        measuredDepthMeters=state.measuredSpace.depthMeters,
                        result=fit.verdict,
                    )
                ),
            )

        purchase = outcome.purchase
        if purchase is not None:
            await self._fail_open(
                "purchase",
                self._service.save_purchase(
                    PurchaseIntentRecord(
                        sessionId=identity,
                        intentId=purchase.id,
                        productId=purchase.productId,
                        merchant=purchase.merchant,
                        merchantUrl=purchase.merchantUrl,
                        amount=purchase.total,
                        currency=purchase.currency,
                        quantity=purchase.quantity,
                        status=purchase.status,
                        trustedAgentStatus=purchase.trustedAgentStatus,
                        commerceProvider=purchase.commerceProvider,
                        isSimulation=purchase.isSimulation,
                    )
                ),
            )

        preference = self._typed_preference(action)
        if preference is not None:
            await self._fail_open(
                "memory",
                self._service.remember(identity, preference, "preference"),
            )

    async def _fail_open(self, label: str, operation: Any) -> None:
        try:
            await operation
        except Exception as exc:
            logger.info("Sponsor %s write failed open: %s", label, type(exc).__name__)

    @staticmethod
    def _identity(state: ConversationSession) -> str:
        return state.shopperId or state.id

    @staticmethod
    def _typed_preference(action: AgentAction) -> str | None:
        if not isinstance(action, RefineSearchAction):
            return None

        constraints = {
            key: value
            for key, value in action.arguments.constraints.model_dump().items()
            if value is not None and (not isinstance(value, bool) or value)
        }
        if action.arguments.relativePrice != "none":
            constraints["relativePrice"] = action.arguments.relativePrice
        if action.arguments.sizePreference != "none":
            constraints["sizePreference"] = action.arguments.sizePreference
        if not constraints:
            return None
        return json.dumps({"explicitRefinement": constraints}, separators=(",", ":"), sort_keys=True)
