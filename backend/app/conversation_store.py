from __future__ import annotations

import asyncio
import secrets
import threading

from .conversation_models import ConversationContextRequest, ConversationSession


class ConversationNotFoundError(LookupError):
    pass


class ConversationStore:
    """Small process-local store for hackathon conversation sessions."""

    def __init__(self, max_sessions: int = 200) -> None:
        self._max_sessions = max_sessions
        self._sessions: dict[str, ConversationSession] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = threading.Lock()

    def create(self) -> ConversationSession:
        session = ConversationSession(id=secrets.token_urlsafe(12))
        with self._guard:
            while len(self._sessions) >= self._max_sessions:
                oldest = next(iter(self._sessions))
                self._sessions.pop(oldest, None)
                self._locks.pop(oldest, None)
            self._sessions[session.id] = session
            self._locks[session.id] = asyncio.Lock()
        return session

    def get(self, session_id: str) -> ConversationSession:
        with self._guard:
            session = self._sessions.get(session_id)
        if session is None:
            raise ConversationNotFoundError(session_id)
        return session

    def lock(self, session_id: str) -> asyncio.Lock:
        self.get(session_id)
        with self._guard:
            return self._locks[session_id]

    def sync_context(
        self,
        session_id: str,
        context: ConversationContextRequest,
        *,
        image_bytes: bytes | None = None,
    ) -> ConversationSession:
        session = self.get(session_id)
        supplied = context.model_fields_set
        if "shopperId" in supplied:
            session.shopperId = context.shopperId
        if "analysis" in supplied:
            session.analyzedObject = context.analysis
        if "imageBase64" in supplied:
            session.analyzedImageBytes = image_bytes
        if "products" in supplied:
            previous_ids = session.result_ids()
            session.latestResults = list(context.products or [])
            if session.result_ids() != previous_ids:
                session.latestSearch = None
                self._cancel_pending_purchase(session)
            valid_ids = set(session.result_ids())
            if session.selectedProductId not in valid_ids:
                session.selectedProductId = None
        if "selectedProductId" in supplied:
            selected = context.selectedProductId
            if selected is not None and selected not in set(session.result_ids()):
                raise ValueError("Selected product is not in the current result list.")
            if selected != session.selectedProductId:
                self._cancel_pending_purchase(session)
            session.selectedProductId = selected
        if "measuredSpace" in supplied:
            session.measuredSpace = context.measuredSpace
        return session

    @staticmethod
    def _cancel_pending_purchase(session: ConversationSession) -> None:
        if session.pendingPurchase is not None and session.pendingPurchase.status == "AWAITING_CONFIRMATION":
            session.lastPurchase = session.pendingPurchase.model_copy(update={"status": "CANCELLED"})
            session.pendingPurchase = None

    def reset(self, session_id: str) -> ConversationSession:
        current = self.get(session_id)
        replacement = ConversationSession(id=session_id, shopperId=current.shopperId)
        with self._guard:
            self._sessions[session_id] = replacement
        return replacement
