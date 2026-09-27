from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.dimension_content import DimensionContent, PragmaticDimensionResolver, _provider_dimensions
from app.dimension_models import AxisMapping, ResolvedDimensions
from app.page_representation import build_text_representation
from app.dimension_semantic import SemanticExtraction
from app.product_models import ProductCandidate
from app.variant_context import RetailerIds, VariantContext
from scripts.dimension_debug import fit_report


def candidate() -> ProductCandidate:
    return ProductCandidate(
        id="serpapi:chair-1",
        provider="serpapi",
        providerProductId="chair-1",
        title="Exact Demo Chair",
        retailer="Demo Store",
        productUrl="https://store.example/chair-1",
        identifiers={"model": "CHAIR-1"},
    )


class Fast:
    def __init__(self, result: ResolvedDimensions) -> None:
        self.result = result
        self.last_trace = {}

    async def resolve(self, _candidate):
        return self.result


class Provider:
    name = "crawl4ai"

    def __init__(self) -> None:
        self.calls = 0

    async def collect(self, _candidate):
        self.calls += 1
        return [DimensionContent(
            provider=self.name,
            source_url="https://manufacturer.example/chair-1",
            source_label="Manufacturer",
            text="Specifications\nOverall Width: 27 in\nOverall Length: 30 in",
        )]


class Extractor:
    async def extract(self, _rep, _page_json):
        return SemanticExtraction.model_validate({
            "status": "partial",
            "dimensions": [
                {"value": 27, "unit": "in", "sourceId": "T2", "label": "width"},
                {"value": 30, "unit": "in", "sourceId": "T3", "label": "length"},
            ],
            "rawText": "Overall Width: 27 in / Overall Length: 30 in",
            "axisMapping": {
                "widthIndex": 0, "depthIndex": 1, "heightIndex": None,
                "confidence": 1, "reason": "explicit labels",
            },
            "reason": None,
        })


def test_complete_structured_result_skips_fallback_provider() -> None:
    product = candidate()
    provider = Provider()
    result = asyncio.run(PragmaticDimensionResolver(
        Fast(ResolvedDimensions(productId=product.id, widthMeters=.6, depthMeters=.7, heightMeters=.9)),
        Extractor(),
        [provider],
        None,
        page_max_chars=4000,
    ).resolve(product))
    assert result.status == "verified"
    assert provider.calls == 0


def test_rendered_width_and_length_enable_fit_without_height() -> None:
    product = candidate()
    result = asyncio.run(PragmaticDimensionResolver(
        Fast(ResolvedDimensions(productId=product.id)),
        Extractor(),
        [Provider()],
        None,
        page_max_chars=4000,
    ).resolve(product))
    assert result.status == "partial"
    assert (result.widthMeters, result.depthMeters, result.heightMeters) == (0.6858, 0.762, None)
    assert result.sourceUrl == "https://manufacturer.example/chair-1"


def test_title_dimensions_require_a_real_google_shopping_candidate() -> None:
    untrusted = candidate().model_copy(update={
        "provider": "debug",
        "title": 'Demo Chair 24" W x 26" D x 34" H',
    })
    assert _provider_dimensions(untrusted) is None

    shopping = untrusted.model_copy(update={
        "provider": "serpapi",
        "productUrl": "https://www.google.com/search?ibp=oshop&q=demo-chair",
    })
    resolved = _provider_dimensions(shopping)
    assert resolved is not None
    assert (resolved.widthMeters, resolved.depthMeters, resolved.heightMeters) == (
        0.6096,
        0.6604,
        0.8636,
    )


@pytest.mark.parametrize(
    ("scope", "expected_scope", "expects_identity"),
    [
        ("exact_record", "exact_variant", True),
        ("family", "product_family", False),
    ],
)
def test_semantic_fallback_preserves_validated_variant_scope(
    scope: str,
    expected_scope: str,
    expects_identity: bool,
) -> None:
    product = candidate()
    rep = build_text_representation(
        "Specifications\nOverall Width: 27 in\nOverall Length: 30 in",
        product.productUrl,
        product_title=product.title,
        retailer=product.retailer,
    )
    rep.sections[0].scope = scope
    for entry in rep.sections[0].text:
        entry.scope = scope
    context = VariantContext(
        retailerDomain="store.example",
        retailerIds=RetailerIds(itemId="chair-1"),
        identity="exact_item",
    )
    fast = Fast(ResolvedDimensions(productId=product.id))
    fast.last_trace = {
        "variant_context": context,
        "attempts": [SimpleNamespace(name="Demo Store", url=product.productUrl, rep=rep)],
    }

    result = asyncio.run(PragmaticDimensionResolver(
        fast,
        Extractor(),
        [],
        None,
        page_max_chars=4000,
    ).resolve(product))

    assert result.variantScope == expected_scope
    assert (result.variantIdentity == context.describe()) is expects_identity


def test_debug_fit_report_allows_width_depth_without_ar_height() -> None:
    fit_only = ResolvedDimensions(productId="fit-only", widthMeters=0.6, depthMeters=0.7)
    report = fit_report(fit_only, 0.7, 0.8)
    assert report["status"] == "FITS"
    assert report["checkFitEligible"] is True
    assert report["arPreviewEligible"] is False

    mesh_mappable = ResolvedDimensions(
        productId="mesh-mappable",
        dimensionsMeters=[0.6, 0.7, 0.9],
        axisMapping=AxisMapping(reason="unlabeled source order"),
    )
    report = fit_report(mesh_mappable, 0.7, 0.8)
    assert report["status"] == "UNKNOWN"
    assert report["checkFitEligible"] is False
    assert report["arPreviewEligible"] is True
