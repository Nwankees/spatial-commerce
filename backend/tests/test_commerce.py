from __future__ import annotations

import asyncio

from app.commerce_service import CommerceService
from app.conversation_models import ConversationSession, ProductReferenceArgs, SearchConstraints
from app.product_models import ProductCandidate


def product(*, price: float = 129.99, retailer: str = "Example Shop") -> ProductCandidate:
    return ProductCandidate(
        id="serpapi:chair-1",
        provider="serpapi",
        providerProductId="chair-1",
        title="Black Modern Chair",
        price=price,
        priceText=f"${price:.2f}",
        currency="USD",
        retailer=retailer,
        productUrl="https://merchant.example/products/chair-1?variant=black",
        identifiers={"sku": "CHAIR-BLK"},
    )


def session() -> ConversationSession:
    item = product()
    return ConversationSession(
        id="conversation-1",
        shopperId="shopper-1",
        latestResults=[item],
        selectedProductId=item.id,
    )


def test_purchase_requires_separate_review_and_exact_confirmation() -> None:
    service = CommerceService()
    state = session()
    state.constraints = SearchConstraints(color="black")

    prepared = service.prepare(state, ProductReferenceArgs(), 2)
    assert prepared.purchase is not None
    assert prepared.purchase.status == "AWAITING_CONFIRMATION"
    assert prepared.purchase.total == 259.98
    assert prepared.purchase.checkoutUrl == state.latestResults[0].productUrl
    assert prepared.purchase.merchant == "Example Shop"
    assert prepared.purchase.price == 129.99
    assert prepared.purchase.sku == "CHAIR-BLK"
    assert prepared.purchase.selectedVariant["color"] == "black"
    assert prepared.purchase.fitStatus == "not_checked"

    stale = asyncio.run(service.confirm(state, prepared.purchase.id, "wrong-reference"))
    assert stale.status == "needs_input"
    assert state.pendingPurchase is not None

    completed = asyncio.run(service.confirm(
        state,
        prepared.purchase.id,
        prepared.purchase.confirmationReference,
    ))
    assert completed.purchase is not None
    assert completed.purchase.status == "COMPLETED_SIMULATED"
    assert completed.purchase.isSimulation is True
    assert completed.trustVerification is not None
    assert completed.trustVerification.verified is True
    assert completed.trustVerification.visaCertified is False
    assert state.pendingPurchase is None


def test_tap_demo_verifier_rejects_tampering_and_replay() -> None:
    service = CommerceService()
    state = session()
    intent = service.prepare(state, ProductReferenceArgs(), 1).purchase
    assert intent is not None

    envelope = service.signer.sign(intent)
    tampered = envelope.model_copy(update={"path": "/products/not-the-chair"})
    assert service.verifier.verify(tampered, intent).verified is False

    tampered_intent = intent.model_copy(update={"price": 1.00, "total": 1.00})
    assert service.verifier.verify(envelope, tampered_intent).verified is False

    invalid_signature = envelope.model_copy(update={"signature": "sig1=:AAAA:"})
    assert service.verifier.verify(invalid_signature, intent).verified is False
    assert "private" not in envelope.model_dump_json().lower()

    clean = service.signer.sign(intent)
    assert service.verifier.verify(clean, intent).verified is True
    replay = service.verifier.verify(clean, intent)
    assert replay.verified is False
    assert "Replay" in (replay.reason or "")


def test_constraints_and_context_change_block_unsafe_purchase() -> None:
    service = CommerceService()
    state = session()
    state.constraints = SearchConstraints(maxPrice=100)
    blocked = service.prepare(state, ProductReferenceArgs(), 1)
    assert blocked.status == "unavailable"

    state.constraints = SearchConstraints(retailer="Walmart")
    blocked_merchant = service.prepare(state, ProductReferenceArgs(), 1)
    assert blocked_merchant.status == "unavailable"

    state.constraints = SearchConstraints()
    prepared = service.prepare(state, ProductReferenceArgs(), 1)
    assert prepared.purchase is not None
    service.invalidate_pending(state)
    stale = asyncio.run(service.confirm(
        state,
        prepared.purchase.id,
        prepared.purchase.confirmationReference,
    ))
    assert stale.status == "needs_input"
    assert state.lastPurchase is not None
    assert state.lastPurchase.status == "CANCELLED"
