from __future__ import annotations

from typing import Awaitable, Callable, Protocol

from .conversation_tools import ArPreviewGatewayResult
from .dimension_models import ResolvedDimensions
from .dimension_resolver import ResolutionCache
from .product_models import ProductCandidate


class M6ArPreviewResponse(Protocol):
    """Small structural view of M6's ``ArAssetResponse``.

    Keeping this adapter independent of the M6 modules lets the isolated M7
    worktree remain testable. After M6 is committed, ``ArPreviewService.request``
    satisfies this protocol directly.
    """

    status: str
    sourceProductId: str
    message: str | None


ArPreviewRequester = Callable[[ProductCandidate], Awaitable[M6ArPreviewResponse]]


class M6ArPreviewGateway:
    """Connects the conversational tool to M6's real reconstruction service.

    ``generating`` is an accepted handoff: it means M6 validated the selected
    product and started the job. Android then opens the existing M6 preview
    flow, which polls that same job and downloads the GLB when it is ready.
    """

    def __init__(self, request_preview: ArPreviewRequester) -> None:
        self._request_preview = request_preview

    async def request(self, product: ProductCandidate) -> ArPreviewGatewayResult:
        response = await self._request_preview(product)
        if response.sourceProductId != product.id:
            return ArPreviewGatewayResult(
                ready=False,
                message="The AR service returned a different product, so the preview was stopped safely.",
            )
        if response.status in {"generating", "ready"}:
            return ArPreviewGatewayResult(
                ready=True,
                message=(
                    response.message
                    or (
                        "The real-scale AR preview is ready."
                        if response.status == "ready"
                        else "Preparing the real-scale AR preview now."
                    )
                ),
            )
        return ArPreviewGatewayResult(
            ready=False,
            message=response.message or "Real-scale AR preview is unavailable for this product.",
        )


class M6DimensionResolver(Protocol):
    async def resolve(self, product: ProductCandidate) -> ResolvedDimensions: ...


class M6CachedDimensionGateway:
    """Shares M6's verified dimension cache with conversational fit and AR."""

    def __init__(self, resolver: M6DimensionResolver, cache: ResolutionCache) -> None:
        self._resolver = resolver
        self._cache = cache

    async def resolve(self, product: ProductCandidate) -> ResolvedDimensions:
        cached = self._cache.get(product.id)
        if cached is not None:
            return cached
        resolved = await self._resolver.resolve(product)
        self._cache.put(resolved)
        return resolved
