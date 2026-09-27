from __future__ import annotations

import secrets
from datetime import datetime, timezone

from .commerce_models import PurchaseIntent, TrustVerification, TrustedAgentEnvelope
from .conversation_models import ConversationSession, ProductReferenceArgs, ToolOutcome
from .product_models import ProductCandidate
from .trusted_agent import MerchantTrustedAgentVerifier, TrustedAgentSigner


class DemoVisaCommerceProvider:
    name = "visa_intelligent_commerce_demo"

    async def complete(self, intent: PurchaseIntent, verification: TrustVerification) -> PurchaseIntent:
        if not verification.verified or intent.status != "TRUST_VERIFIED":
            return intent.model_copy(update={"status": "FAILED", "failureReason": "Trust verification failed."})
        now = datetime.now(timezone.utc)
        return intent.model_copy(update={
            "status": "COMPLETED_SIMULATED",
            "completedAt": now,
            "trustedAgentStatus": "verified_demo_key",
        })


class CommerceService:
    def __init__(self) -> None:
        self.signer = TrustedAgentSigner()
        self.verifier = MerchantTrustedAgentVerifier({self.signer.key_id: self.signer.public_key})
        self.provider = DemoVisaCommerceProvider()

    def prepare(
        self,
        state: ConversationSession,
        reference: ProductReferenceArgs,
        quantity: int,
    ) -> ToolOutcome:
        product, error = self._resolve(reference, state)
        if product is None:
            return ToolOutcome(status="needs_input", message=error or "Choose a product before buying.")
        if product.price is None or not product.currency:
            return ToolOutcome(
                status="unavailable",
                message="I can't prepare checkout because this listing has no verified price and currency.",
                selectedProduct=product,
            )
        if not product.retailer or not product.productUrl:
            return ToolOutcome(status="unavailable", message="This listing has no verified merchant checkout link.")
        constraints = state.constraints
        if constraints.maxPrice is not None and product.price > constraints.maxPrice:
            return ToolOutcome(status="unavailable", message="That product exceeds your current price limit.")
        if constraints.retailer and constraints.retailer.lower() not in product.retailer.lower():
            return ToolOutcome(status="unavailable", message="That product does not match your retailer constraint.")
        if state.pendingPurchase and state.pendingPurchase.status == "AWAITING_CONFIRMATION":
            state.pendingPurchase.status = "CANCELLED"
            state.lastPurchase = state.pendingPurchase
        fit = state.latestFit.verdict if state.latestFit and state.latestFit.productId == product.id else "not_checked"
        identifiers = {str(key).lower(): value for key, value in product.identifiers.items()}
        selected_variant = dict(product.identifiers)
        if constraints.color and constraints.color.lower() in product.title.lower():
            selected_variant["color"] = constraints.color
        intent = PurchaseIntent(
            id="pi_" + secrets.token_urlsafe(12),
            sessionId=state.id,
            shopperId=state.shopperId or state.id,
            productId=product.id,
            title=product.title,
            merchant=product.retailer,
            merchantUrl=product.productUrl,
            checkoutUrl=product.productUrl,
            imageUrl=product.imageUrl,
            price=product.price,
            currency=product.currency,
            quantity=quantity,
            total=round(product.price * quantity, 2),
            selectedVariant=selected_variant,
            sku=identifiers.get("sku") or identifiers.get("item_id"),
            model=identifiers.get("model") or identifiers.get("mpn"),
            fitStatus=fit,
            status="AWAITING_CONFIRMATION",
            confirmationReference=secrets.token_urlsafe(10),
        )
        state.selectedProductId = product.id
        state.pendingPurchase = intent
        fit_note = {
            "fits": " It fits the measured space.",
            "does_not_fit": " Warning: it does not fit the measured space.",
            "unknown": " Fit could not be verified.",
            "needs_measurement": " Fit is not verified yet.",
            "not_checked": " Fit has not been checked.",
        }.get(fit, "")
        return ToolOutcome(
            status="success",
            message=(f"Review {quantity} × {product.title} from {product.retailer} for "
                     f"{intent.total:.2f} {intent.currency}.{fit_note} Confirm or cancel; nothing has been charged."),
            selectedProduct=product,
            purchase=intent,
            uiDirective="review_purchase",
        )

    async def confirm(self, state: ConversationSession, intent_id: str | None, confirmation_reference: str | None) -> ToolOutcome:
        intent = state.pendingPurchase
        if intent is None or intent.status != "AWAITING_CONFIRMATION":
            return ToolOutcome(status="needs_input", message="There is no active purchase to confirm. Ask me to buy a selected product first.")
        if intent_id != intent.id or confirmation_reference != intent.confirmationReference:
            return ToolOutcome(status="needs_input", message="That confirmation is stale. Review the current product again before confirming.")
        confirmed = intent.model_copy(update={"status": "CONFIRMED", "confirmedAt": datetime.now(timezone.utc)})
        envelope = self.signer.sign(confirmed)
        verification = self.verifier.verify(envelope, confirmed)
        if not verification.verified:
            failed = confirmed.model_copy(update={"status": "FAILED", "failureReason": verification.reason})
            state.pendingPurchase = None
            state.lastPurchase = failed
            return ToolOutcome(status="error", message="The merchant rejected the trusted-agent proof.", purchase=failed)
        trusted = confirmed.model_copy(update={"status": "TRUST_VERIFIED", "trustedAgentStatus": "verified_demo_key"})
        completed = await self.provider.complete(trusted, verification)
        state.pendingPurchase = None
        state.lastPurchase = completed
        return ToolOutcome(
            status="success",
            message=("Purchase completed in the hackathon simulation. The TAP-style signature was verified "
                     "with a demo key; no real payment was made."),
            purchase=completed,
            trustVerification=verification,
            uiDirective="purchase_complete",
        )

    def cancel(self, state: ConversationSession) -> ToolOutcome:
        intent = state.pendingPurchase
        if intent is None or intent.status != "AWAITING_CONFIRMATION":
            return ToolOutcome(status="needs_input", message="There is no active purchase to cancel.")
        cancelled = intent.model_copy(update={"status": "CANCELLED"})
        state.pendingPurchase = None
        state.lastPurchase = cancelled
        return ToolOutcome(status="success", message="Purchase cancelled. Nothing was charged.", purchase=cancelled)

    def status(self, state: ConversationSession) -> ToolOutcome:
        intent = state.pendingPurchase or state.lastPurchase
        if intent is None:
            return ToolOutcome(status="needs_input", message="There is no purchase in this conversation yet.")
        return ToolOutcome(status="success", message=f"Purchase status: {intent.status}.", purchase=intent)

    def open_checkout(self, state: ConversationSession) -> ToolOutcome:
        intent = state.pendingPurchase or state.lastPurchase
        if intent is None:
            return ToolOutcome(status="needs_input", message="Choose and review a product before opening its merchant page.")
        return ToolOutcome(
            status="success",
            message=f"Opening the original {intent.merchant} listing.",
            purchase=intent,
            checkoutUrl=intent.checkoutUrl,
            uiDirective="open_checkout",
        )

    def invalidate_pending(self, state: ConversationSession) -> None:
        if state.pendingPurchase and state.pendingPurchase.status == "AWAITING_CONFIRMATION":
            state.lastPurchase = state.pendingPurchase.model_copy(update={"status": "CANCELLED"})
            state.pendingPurchase = None

    @staticmethod
    def _resolve(reference: ProductReferenceArgs, state: ConversationSession) -> tuple[ProductCandidate | None, str | None]:
        if reference.productId:
            product = next((p for p in state.latestResults if p.id == reference.productId), None)
            return (product, None) if product else (None, "That product is no longer in the current results.")
        if reference.resultNumber is not None:
            index = reference.resultNumber - 1
            if 0 <= index < len(state.latestResults):
                return state.latestResults[index], None
            return None, f"There isn't a result #{reference.resultNumber}."
        product = state.selected_product()
        return (product, None) if product else (None, "Choose a product before buying.")
