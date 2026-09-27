from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .conversation_models import (
    AgentAction,
    CheckFitAction,
    CompareProductsAction,
    ConfirmPurchaseAction,
    CancelPurchaseAction,
    ConversationSession,
    FindSimilarAction,
    FitAssessment,
    GetProductDetailsAction,
    GetPurchaseStatusAction,
    OpenCheckoutAction,
    PreparePurchaseAction,
    ProductReferenceArgs,
    RefineSearchAction,
    RequestArPreviewAction,
    SearchConstraints,
    SelectProductAction,
    ToolOutcome,
)
from .commerce_service import CommerceService
from .dimension_models import ResolvedDimensions
from .models import VisualProductAnalysis
from .product_models import ProductCandidate, ProductSearchResponse


class ProductSearchGateway(Protocol):
    async def search(
        self,
        analysis: VisualProductAnalysis,
        max_results: int,
        image_bytes: bytes | None = None,
    ) -> ProductSearchResponse: ...


class DimensionGateway(Protocol):
    async def resolve(self, candidate: ProductCandidate) -> ResolvedDimensions: ...


@dataclass(frozen=True)
class ArPreviewGatewayResult:
    ready: bool
    message: str


class ArPreviewGateway(Protocol):
    async def request(self, product: ProductCandidate) -> ArPreviewGatewayResult: ...


class UnavailableArPreviewGateway:
    """M7 contract boundary until the isolated M6 implementation is merged."""

    async def request(self, product: ProductCandidate) -> ArPreviewGatewayResult:
        return ArPreviewGatewayResult(
            ready=False,
            message="Real-product AR preview is not connected to this M7 worktree yet.",
        )


class ShoppingToolExecutor:
    """Executes validated actions; the model never mutates state directly."""

    def __init__(
        self,
        search: ProductSearchGateway,
        dimensions: DimensionGateway,
        ar_preview: ArPreviewGateway | None = None,
        product_lookup: Callable[[str], ProductCandidate | None] | None = None,
        commerce: CommerceService | None = None,
    ) -> None:
        self._search = search
        self._dimensions = dimensions
        self._ar = ar_preview or UnavailableArPreviewGateway()
        self._lookup = product_lookup
        self._commerce = commerce

    async def execute(self, action: AgentAction, state: ConversationSession) -> ToolOutcome:
        if self._commerce is not None:
            if isinstance(action, PreparePurchaseAction):
                return self._commerce.prepare(state, action.arguments, action.arguments.quantity)
            if isinstance(action, ConfirmPurchaseAction):
                return await self._commerce.confirm(
                    state,
                    action.arguments.intentId,
                    action.arguments.confirmationReference,
                )
            if isinstance(action, CancelPurchaseAction):
                return self._commerce.cancel(state)
            if isinstance(action, GetPurchaseStatusAction):
                return self._commerce.status(state)
            if isinstance(action, OpenCheckoutAction):
                return self._commerce.open_checkout(state)
            # A changed selection/search context makes an old "yes" unsafe.
            if isinstance(action, (FindSimilarAction, RefineSearchAction, SelectProductAction)):
                self._commerce.invalidate_pending(state)
        if isinstance(action, FindSimilarAction):
            return await self._find(action, state)
        if isinstance(action, RefineSearchAction):
            return await self._refine(action, state)
        if isinstance(action, SelectProductAction):
            return self._select(action, state)
        if isinstance(action, CheckFitAction):
            return await self._check_fit(action, state)
        if isinstance(action, RequestArPreviewAction):
            return await self._request_ar(action, state)
        if isinstance(action, GetProductDetailsAction):
            return self._details(action, state)
        if isinstance(action, CompareProductsAction):
            return self._compare(action, state)
        return ToolOutcome(status="error", message="That action is not supported.")

    async def _find(self, action: FindSimilarAction, state: ConversationSession) -> ToolOutcome:
        if state.analyzedObject is None or not state.analyzedObject.objectDetected:
            return ToolOutcome(
                status="needs_input",
                message="Point the camera at a product and tap Analyze object first.",
                uiDirective="analyze_object",
            )
        if state.latestResults:
            return ToolOutcome(
                status="success",
                message=f"You already have {len(state.latestResults)} current matches. Tap one, or ask me to refine them.",
                products=state.latestResults,
                searchResult=state.latestSearch,
                uiDirective="show_products",
            )
        try:
            response = await self._search.search(
                state.analyzedObject,
                action.arguments.maxResults,
                state.analyzedImageBytes,
            )
        except Exception:
            return ToolOutcome(
                status="error",
                message="Product search is unavailable right now. The existing buttons still work; please retry.",
            )
        state.latestResults = list(response.products)
        state.latestSearch = response
        state.selectedProductId = None
        state.latestFit = None
        if not state.latestResults:
            return ToolOutcome(status="unavailable", message="I couldn't find purchasable matches for that object.")
        return ToolOutcome(
            status="success",
            message=f"I found {len(state.latestResults)} matches. Tap one, or ask for a cheaper/color/material option.",
            products=state.latestResults,
            searchResult=response,
            uiDirective="show_products",
        )

    async def _refine(self, action: RefineSearchAction, state: ConversationSession) -> ToolOutcome:
        args = action.arguments
        constraints = state.constraints.merged(args.constraints)
        if args.relativePrice == "cheaper":
            reference = state.selected_product() or (state.latestResults[0] if state.latestResults else None)
            if reference is None:
                return ToolOutcome(
                    status="needs_input",
                    message="Find products first so I know what 'cheaper' should be compared with.",
                )
            if reference.price is None:
                return ToolOutcome(
                    status="unavailable",
                    message="The selected product has no verified price, so I can't compare cheaper options safely.",
                    selectedProduct=reference,
                )
            cheaper_than = max(0.0, reference.price - 0.01)
            constraints = constraints.merged(SearchConstraints(maxPrice=cheaper_than))
        state.constraints = constraints

        if not state.latestResults:
            found = await self._find(
                FindSimilarAction(action="find_similar_products", arguments={"maxResults": 10}),
                state,
            )
            if found.status != "success":
                return found

        filtered = _apply_constraints(state.latestResults, constraints)
        if args.sizePreference == "smaller":
            measurable = [p for p in filtered if _footprint(p) is not None]
            filtered = sorted(measurable, key=lambda p: _footprint(p) or float("inf"))

        if not filtered and state.analyzedObject is not None:
            # A local refinement found nothing. Reuse the normal retrieval pipeline with
            # constraint words added to its already-grounded analysis queries.
            refined_analysis = _analysis_with_constraints(state.analyzedObject, constraints)
            try:
                response = await self._search.search(refined_analysis, 10, state.analyzedImageBytes)
                filtered = _apply_constraints(list(response.products), constraints)
                state.latestSearch = response
            except Exception:
                filtered = []

        if not filtered:
            return ToolOutcome(
                status="unavailable",
                message="None of the current purchasable results match that refinement. Try relaxing one constraint.",
            )

        # Keep the previous relative order unless a user explicitly asked for smaller.
        state.latestResults = filtered[:10]
        if state.latestSearch is not None:
            state.latestSearch = state.latestSearch.model_copy(update={"products": state.latestResults})
        if state.selectedProductId not in state.result_ids():
            state.selectedProductId = None
        state.latestFit = None
        summary = _constraint_summary(
            constraints,
            args.relativePrice == "cheaper",
            args.sizePreference == "smaller",
        )
        return ToolOutcome(
            status="success",
            message=f"Showing {len(state.latestResults)} result(s){summary}.",
            products=state.latestResults,
            searchResult=state.latestSearch,
            uiDirective="show_products",
        )

    def _select(self, action: SelectProductAction, state: ConversationSession) -> ToolOutcome:
        product, error = self._resolve_reference(action.arguments, state)
        if product is None:
            return ToolOutcome(status="needs_input", message=error or "Choose a product first.")
        state.selectedProductId = product.id
        state.latestFit = None
        return ToolOutcome(
            status="success",
            message=f"Selected {product.title} for {_display_price(product)}.",
            selectedProduct=product,
            uiDirective="select_product",
        )

    async def _check_fit(self, action: CheckFitAction, state: ConversationSession) -> ToolOutcome:
        product, error = self._resolve_reference(action.arguments, state)
        if product is None:
            return ToolOutcome(status="needs_input", message=error or "Choose a product first.")
        state.selectedProductId = product.id
        if state.measuredSpace is None:
            state.latestFit = FitAssessment(productId=product.id, verdict="needs_measurement")
            return ToolOutcome(
                status="needs_input",
                message="Measure the available floor space first; I only need width and depth.",
                selectedProduct=product,
                fit=state.latestFit,
                uiDirective="measure_space",
            )

        width, depth, source_url = await self._dimensions_for(product)
        if width is None or depth is None:
            state.latestFit = FitAssessment(
                productId=product.id,
                verdict="unknown",
                availableWidthMeters=state.measuredSpace.widthMeters,
                availableDepthMeters=state.measuredSpace.depthMeters,
            )
            return ToolOutcome(
                status="unavailable",
                message="I couldn't verify two horizontal product dimensions, so I won't guess whether it fits.",
                selectedProduct=product,
                fit=state.latestFit,
                uiDirective="show_fit",
            )

        space = state.measuredSpace
        normal = width <= space.widthMeters and depth <= space.depthMeters
        rotated = depth <= space.widthMeters and width <= space.depthMeters
        fits = normal or rotated
        fit = FitAssessment(
            productId=product.id,
            verdict="fits" if fits else "does_not_fit",
            productWidthMeters=width,
            productDepthMeters=depth,
            availableWidthMeters=space.widthMeters,
            availableDepthMeters=space.depthMeters,
            rotated=(not normal and rotated) if fits else None,
            sourceUrl=source_url,
        )
        state.latestFit = fit
        verdict = "FITS" if fits else "DOES NOT FIT"
        rotation_note = " when rotated" if fit.rotated else ""
        return ToolOutcome(
            status="success",
            message=(
                f"{verdict}{rotation_note}: product footprint {width:.2f} × {depth:.2f} m; "
                f"measured space {space.widthMeters:.2f} × {space.depthMeters:.2f} m."
            ),
            selectedProduct=product,
            fit=fit,
            uiDirective="show_fit",
        )

    async def _request_ar(self, action: RequestArPreviewAction, state: ConversationSession) -> ToolOutcome:
        product, error = self._resolve_reference(action.arguments, state)
        if product is None:
            return ToolOutcome(status="needs_input", message=error or "Choose a product first.")
        state.selectedProductId = product.id
        try:
            result = await self._ar.request(product)
        except Exception:
            return ToolOutcome(
                status="error",
                message="AR preview preparation failed. Your product selection is still saved; please retry.",
                selectedProduct=product,
            )
        if not result.ready:
            return ToolOutcome(
                status="unavailable",
                message=result.message,
                selectedProduct=product,
            )
        state.latestArPreviewProductId = product.id
        return ToolOutcome(
            status="success",
            message=result.message,
            selectedProduct=product,
            uiDirective="enter_ar_preview",
        )

    def _details(self, action: GetProductDetailsAction, state: ConversationSession) -> ToolOutcome:
        product, error = self._resolve_reference(action.arguments, state)
        if product is None:
            return ToolOutcome(status="needs_input", message=error or "Choose a product first.")
        merchant = f" at {product.retailer}" if product.retailer else ""
        return ToolOutcome(
            status="success",
            message=f"{product.title} is {_display_price(product)}{merchant}.",
            selectedProduct=product,
        )

    def _compare(self, action: CompareProductsAction, state: ConversationSession) -> ToolOutcome:
        if action.arguments.resultNumbers:
            products = [
                state.latestResults[number - 1]
                for number in action.arguments.resultNumbers
                if 1 <= number <= len(state.latestResults)
            ]
        else:
            products = state.latestResults
        measurable = [(product, _footprint(product)) for product in products]
        measurable = [(product, area) for product, area in measurable if area is not None]
        if len(measurable) < 2:
            return ToolOutcome(
                status="unavailable",
                message="I need verified width and depth for at least two results before I can say which is smaller.",
            )
        product, area = min(measurable, key=lambda item: item[1])
        return ToolOutcome(
            status="success",
            message=f"{product.title} has the smallest verified footprint ({area:.2f} m²).",
            selectedProduct=product,
        )

    def _resolve_reference(
        self,
        reference: ProductReferenceArgs,
        state: ConversationSession,
    ) -> tuple[ProductCandidate | None, str | None]:
        product: ProductCandidate | None = None
        if reference.productId:
            product = next((p for p in state.latestResults if p.id == reference.productId), None)
            if product is None:
                return None, "That product is no longer in the current results. Search again or choose another one."
        elif reference.resultNumber is not None:
            index = reference.resultNumber - 1
            if index < 0 or index >= len(state.latestResults):
                return None, f"There isn't a result #{reference.resultNumber} in the current list."
            product = state.latestResults[index]
        else:
            product = state.selected_product()
        if product is not None and self._lookup is not None:
            known = self._lookup(product.id)
            if known is None:
                return None, "That product is no longer known to the backend. Search again before using it."
            product = known
        return product, None

    async def _dimensions_for(self, product: ProductCandidate) -> tuple[float | None, float | None, str | None]:
        if product.dimensions.widthMeters and product.dimensions.depthMeters:
            return product.dimensions.widthMeters, product.dimensions.depthMeters, None
        try:
            result = await self._dimensions.resolve(product)
        except Exception:
            return None, None, None
        return result.widthMeters, result.depthMeters, result.sourceUrl


def _apply_constraints(products: list[ProductCandidate], constraints: SearchConstraints) -> list[ProductCandidate]:
    result: list[ProductCandidate] = []
    for product in products:
        searchable = f"{product.title} {product.retailer or ''}".lower()
        if constraints.maxPrice is not None and (product.price is None or product.price > constraints.maxPrice):
            continue
        if constraints.minPrice is not None and (product.price is None or product.price < constraints.minPrice):
            continue
        if constraints.retailer and constraints.retailer.lower() not in (product.retailer or "").lower():
            continue
        if constraints.color and constraints.color.lower() not in searchable:
            continue
        material = constraints.material
        if material:
            aliases = {"wooden": ("wood", "wooden"), "wood": ("wood", "wooden")}.get(material.lower(), (material.lower(),))
            if not any(alias in searchable for alias in aliases):
                continue
        result.append(product)
    return result


def _analysis_with_constraints(
    analysis: VisualProductAnalysis,
    constraints: SearchConstraints,
) -> VisualProductAnalysis:
    suffix = " ".join(
        part for part in (constraints.color, constraints.material, constraints.retailer) if part
    )
    if not suffix:
        return analysis
    base_queries = analysis.searchQueries or [analysis.subcategory or analysis.category or "product"]
    return analysis.model_copy(update={"searchQueries": [f"{query} {suffix}"[:120] for query in base_queries]})


def _footprint(product: ProductCandidate) -> float | None:
    width = product.dimensions.widthMeters
    depth = product.dimensions.depthMeters
    return width * depth if width and depth else None


def _display_price(product: ProductCandidate) -> str:
    if product.priceText:
        return product.priceText
    if product.price is not None:
        return f"${product.price:.2f}"
    return "an unavailable price"


def _constraint_summary(constraints: SearchConstraints, cheaper: bool, smaller: bool) -> str:
    parts: list[str] = []
    if constraints.maxPrice is not None:
        parts.append(f"under ${constraints.maxPrice:.2f}")
    if constraints.color:
        parts.append(constraints.color)
    if constraints.material:
        parts.append(constraints.material)
    if constraints.retailer:
        parts.append(f"from {constraints.retailer}")
    if smaller:
        parts.append("smallest first")
    elif cheaper and not any(part.startswith("under") for part in parts):
        parts.append("cheaper")
    return " " + ", ".join(parts) if parts else ""
