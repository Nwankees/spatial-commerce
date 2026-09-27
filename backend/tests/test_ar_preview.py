from __future__ import annotations

import asyncio
import base64
import io
import math
from pathlib import Path

import httpx
import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image

from app.ar_geometry import (AssetValidationError, compute_scale, export_glb, footprint_yaw_degrees,
                             load_single_mesh, map_axes_by_mesh, normalize_mesh)
from app.ar_preview import (
    UNVERIFIED_MESSAGE,
    ArAssetStore,
    ArPreviewService,
    DimensionLookupResult,
    asset_fingerprint,
    dimension_fingerprint,
)
from app.dimension_models import AxisMapping, ResolvedDimensions
from app.main import app, get_ar_preview_service, get_product_cache
from app.product_cache import ProductSearchCache
from app.product_image import ProductImage, ProductImageError, download_product_image
from app.product_models import ProductCandidate
from app.reconstruction import ReconstructionError, ReconstructionOutput, SidecarReconstructionProvider


def run(coro):
    return asyncio.run(coro)


def textured_box(extents=(2.0, 1.0, 4.0), translate=(0, 0, 0), yaw_degrees=0.0) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    uv = np.column_stack([np.linspace(0, 1, len(mesh.vertices)), np.linspace(1, 0, len(mesh.vertices))])
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.new("RGB", (8, 8), (180, 60, 40)))
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    if yaw_degrees:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(yaw_degrees), [0, 1, 0]))
    mesh.apply_translation(translate)
    return mesh


def glb_of(mesh) -> bytes:
    return mesh.export(file_type="glb", include_normals=True)


# ---- Geometry ------------------------------------------------------------

def test_bounds_and_normalization_center_and_floor() -> None:
    mesh = load_single_mesh(glb_of(textured_box(translate=(5, 3, -2))))
    normalized, info = normalize_mesh(mesh)
    assert info.original_bounds.min == pytest.approx((4, 2.5, -4)) and info.original_bounds.size == pytest.approx((2, 1, 4))
    lo, hi = normalized.bounds
    assert lo[1] == pytest.approx(0) and (lo[0] + hi[0]) == pytest.approx(0) and (lo[2] + hi[2]) == pytest.approx(0)
    assert info.normalized_bounds.size == pytest.approx((2, 1, 4))  # aspect ratio preserved, no scaling
    assert info.textured and info.yaw_degrees == 0


def test_rotated_footprint_is_straightened_without_swapping_axes() -> None:
    mesh = load_single_mesh(glb_of(textured_box(extents=(4, 1, 2), yaw_degrees=30)))
    normalized, info = normalize_mesh(mesh)
    assert abs(info.yaw_degrees) == pytest.approx(30, abs=0.01)
    assert info.normalized_bounds.size == pytest.approx((4, 1, 2), abs=1e-4)


def test_round_footprint_is_left_alone() -> None:
    points = np.array([[math.cos(t), math.sin(t)] for t in np.linspace(0, 2 * math.pi, 64, endpoint=False)])
    assert footprint_yaw_degrees(points) == 0.0


def test_verified_dimension_scaling_math() -> None:
    s = compute_scale((2.0, 1.0, 4.0), width_m=1.0, depth_m=2.0, height_m=0.5)
    assert (s.scale_x, s.scale_y, s.scale_z, s.axes_swapped, s.max_axis_distortion) == (0.5, 0.5, 0.5, False, 0.0)
    stretched = compute_scale((2.0, 1.0, 4.0), width_m=1.0, depth_m=2.0, height_m=1.0)
    assert stretched.scale_y == 1.0 and stretched.max_axis_distortion == pytest.approx(1.0)
    # Mesh long axis is Z but verified width is the longer dimension -> axes swap (90° yaw on device).
    swapped = compute_scale((2.0, 1.0, 4.0), width_m=2.0, depth_m=1.0, height_m=0.5)
    assert swapped.axes_swapped and swapped.scale_x == 0.5 and swapped.scale_z == 0.5
    # Nearly square footprints never swap.
    assert not compute_scale((1.0, 1.0, 1.05), width_m=1.0, depth_m=0.95, height_m=1.0).axes_swapped


def test_mesh_axis_mapping_tall_wide_ambiguous_and_poor() -> None:
    tall = map_axes_by_mesh((0.30, 1.0, 0.29), [13.39 * 0.0254, 4.13 * 0.0254, 4.13 * 0.0254])
    assert tall.accepted and tall.height_index == 0 and tall.height == pytest.approx(13.39 * 0.0254)
    assert tall.confidence > 0.5 and tall.mismatch < 0.15

    # Nearly equal footprint measurements produce distinct permutations but
    # effectively the same physical X/Y/Z scale, so they collapse together.
    near = map_axes_by_mesh((0.30, 1.0, 0.29), [0.340, 0.105, 0.107])
    assert near.accepted and near.height_index == 0
    assert "collapsed to 3 physical outcomes" in near.reason

    wide = map_axes_by_mesh((2.0, 1.0, 0.8), [0.8, 2.0, 1.0])
    assert wide.accepted and wide.permutation == [1, 0, 2]

    ambiguous = map_axes_by_mesh((1.0, 1.0, 1.0), [0.5, 0.8, 1.2])
    assert not ambiguous.accepted and "do not distinguish" in ambiguous.reason

    poor = map_axes_by_mesh((0.2, 3.0, 0.2), [0.8, 0.7, 0.6])
    assert not poor.accepted and "disagree" in poor.reason


def test_mesh_axis_mapping_respects_labeled_height() -> None:
    mapped = map_axes_by_mesh((0.5, 0.9, 0.7), [0.7290, 0.6401, 0.9703], height_index=2)
    assert mapped.accepted and mapped.height_index == 2 and mapped.height == pytest.approx(0.9703)


@pytest.mark.parametrize("dims", [(1.0, 2.0, None), (None, 2.0, 1.0), (1.0, 0.0, 1.0)])
def test_scale_requires_all_verified_dimensions(dims) -> None:
    with pytest.raises(ValueError):
        compute_scale((2.0, 1.0, 4.0), *dims)


def test_invalid_assets_rejected() -> None:
    with pytest.raises(AssetValidationError):
        load_single_mesh(b"not a glb")
    scene = trimesh.Scene([textured_box(), textured_box(translate=(5, 0, 0))])
    with pytest.raises(AssetValidationError, match="exactly one mesh"):
        load_single_mesh(scene.export(file_type="glb"))
    flat = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 0, 1]], faces=[[0, 1, 2]])
    with pytest.raises(AssetValidationError, match="degenerate"):
        load_single_mesh(glb_of(flat))


def test_exported_glb_round_trips() -> None:
    normalized, _ = normalize_mesh(load_single_mesh(glb_of(textured_box())))
    again = load_single_mesh(export_glb(normalized))
    assert len(again.faces) == 12


# ---- Service ---------------------------------------------------------------

PRODUCT = ProductCandidate(id="serpapi:1", provider="serpapi", providerProductId="1", title="Chair", price=71.99,
                           retailer="Target", imageUrl="https://img.example.com/chair.jpg",
                           productUrl="https://www.google.com/search?prds=productid:1", detailPageToken="tok")
VERIFIED = ResolvedDimensions(productId="serpapi:1", widthMeters=0.55, depthMeters=0.56, heightMeters=1.0,
                              sourceType="page_text_llm", sourceName="Target (selected variant)", variantScope="exact_variant")


def image(data=b"image-1") -> ProductImage:
    return ProductImage("https://img.example.com/chair.jpg", data, str(abs(hash(data))).ljust(64, "0")[:64], 512, 512, "JPEG", 0.1)


class FakeProvider:
    name, version = "sf3d", "test"

    def __init__(self, outcome=None):
        self.calls = 0
        self.outcome = outcome
        self.last_image = None

    async def health(self):
        return {"reachable": True}

    async def reconstruct(self, image_bytes):
        self.calls += 1
        self.last_image = image_bytes
        if isinstance(self.outcome, Exception):
            raise self.outcome
        glb = self.outcome if isinstance(self.outcome, bytes) else glb_of(textured_box(extents=(0.6, 1.1, 0.6), translate=(1, 1, 1)))
        return ReconstructionOutput(glb, b"png", {"reconstructionSeconds": 1.5})


def service(tmp_path: Path, provider=None, dims=VERIFIED, downloader=None):
    async def lookup(product):
        return dims

    async def download(url):
        return image()

    return ArPreviewService(ArAssetStore(tmp_path / "assets"), provider or FakeProvider(), lookup, downloader or download)


async def settle(svc, asset_id):
    for _ in range(100):
        job = svc._jobs.get(asset_id)
        if job and job.task and job.task.done():
            return
        await asyncio.sleep(0.01)


def test_generation_then_ready_with_verified_scale_and_cache_hit(tmp_path: Path) -> None:
    provider = FakeProvider()
    downloaded_urls = []

    async def selected_image(url):
        downloaded_urls.append(url)
        return image(b"selected-retailer-product-image")

    async def flow():
        svc = service(tmp_path, provider, downloader=selected_image)
        first = await svc.request(PRODUCT)
        assert first.status == "generating" and first.assetId
        await settle(svc, first.assetId)
        ready = svc.status(first.assetId)
        again = await svc.request(PRODUCT)
        return first, ready, again, svc

    first, ready, again, svc = run(flow())
    assert ready.status == "ready" and ready.assetUrl == f"/api/v1/ar-assets/{first.assetId}/model.glb"
    assert ready.normalizedBounds.min[1] == pytest.approx(0)
    assert ready.scale.widthMeters == 0.55 and ready.scale.scaleY == pytest.approx(1.0 / 1.1, rel=1e-4)
    assert first.cacheOutcome == "miss"
    assert ready.cacheOutcome == "miss"
    assert again.status == "ready" and again.cached and again.cacheOutcome == "hit" and provider.calls == 1
    assert downloaded_urls == [PRODUCT.imageUrl, PRODUCT.imageUrl]
    assert provider.last_image == b"selected-retailer-product-image"
    assert ready.sourceImageUrl == PRODUCT.imageUrl
    assert svc.glb_path(first.assetId).read_bytes()[:4] == b"glTF"
    assert {
        "provider.reconstructionSeconds",
        "meshLoadSeconds",
        "normalizeSeconds",
        "normalizedExportSeconds",
        "validationSeconds",
        "writeSeconds",
        "totalGenerationSeconds",
        "imageDownloadSeconds",
        "dimensionResolutionSeconds",
        "requestTotalSeconds",
    } <= ready.timings.keys()
    assert ready.glbBytes > 0


def test_unlabeled_verified_values_are_mapped_after_reconstruction(tmp_path: Path) -> None:
    dims = ResolvedDimensions(
        productId=PRODUCT.id,
        dimensionsMeters=[1.1, 0.6, 0.6],
        axisMapping=AxisMapping(reason="axis order not labeled"),
        sourceType="spec_table",
    )

    async def flow():
        svc = service(tmp_path, dims=dims)
        started = await svc.request(PRODUCT)
        await settle(svc, started.assetId)
        return svc.status(started.assetId)

    ready = run(flow())
    assert ready.status == "ready" and ready.scale.axisMappingSource == "mesh_assisted"
    assert ready.scale.heightIndex == 0 and ready.scale.axisMappingConfidence > 0.5
    assert ready.scale.heightMeters == pytest.approx(1.1)


def test_ambiguous_mesh_does_not_guess_unlabeled_axis_order(tmp_path: Path) -> None:
    dims = ResolvedDimensions(
        productId=PRODUCT.id,
        dimensionsMeters=[1.2, 0.8, 0.5],
        axisMapping=AxisMapping(reason="axis order not labeled"),
        sourceType="spec_table",
    )

    async def flow():
        svc = service(tmp_path, provider=FakeProvider(glb_of(textured_box(extents=(1, 1, 1)))), dims=dims)
        started = await svc.request(PRODUCT)
        await settle(svc, started.assetId)
        return svc.status(started.assetId)

    result = run(flow())
    assert result.status == "unavailable" and result.scale is None and not result.retryable
    assert "could not map their axis order safely" in result.message


def test_cache_key_changes_with_source_image() -> None:
    dimensions = dimension_fingerprint(VERIFIED)
    args = (PRODUCT.id, PRODUCT.productUrl, "a" * 64, dimensions, "sf3d", "v1")
    a = asset_fingerprint(*args)
    assert a != asset_fingerprint(PRODUCT.id, PRODUCT.productUrl, "b" * 64, dimensions, "sf3d", "v1")
    assert a != asset_fingerprint("serpapi:2", PRODUCT.productUrl, "a" * 64, dimensions, "sf3d", "v1")
    assert a != asset_fingerprint(PRODUCT.id, PRODUCT.productUrl, "a" * 64, dimensions, "triposr", "v1")
    changed = VERIFIED.model_copy(update={"widthMeters": 0.75, "dimensionsMeters": [0.75, 0.56, 1.0]})
    assert a != asset_fingerprint(
        PRODUCT.id,
        PRODUCT.productUrl,
        "a" * 64,
        dimension_fingerprint(changed),
        "sf3d",
        "v1",
    )


def test_dimension_lookup_diagnostics_show_check_fit_reuse(tmp_path: Path) -> None:
    async def lookup(product):
        return DimensionLookupResult(
            VERIFIED,
            cache_hit=True,
            path="check_fit_cache",
            timings={"cacheLookupSeconds": 0.002},
        )

    async def download(url):
        return image()

    async def flow():
        svc = ArPreviewService(ArAssetStore(tmp_path / "assets"), FakeProvider(), lookup, download)
        started = await svc.request(PRODUCT)
        await settle(svc, started.assetId)
        return svc.status(started.assetId)

    ready = run(flow())
    assert ready.status == "ready"
    assert ready.dimensionSource["lookupCacheHit"] is True
    assert ready.dimensionSource["lookupPath"] == "check_fit_cache"
    assert ready.timings["dimension.cacheLookupSeconds"] == pytest.approx(0.002)


@pytest.mark.parametrize("dims", [
    ResolvedDimensions(productId="serpapi:1", message="none"),
    ResolvedDimensions(productId="serpapi:1", widthMeters=0.5, depthMeters=0.5, sourceType="spec_table"),
])
def test_unverified_dimensions_never_generate(tmp_path: Path, dims) -> None:
    provider = FakeProvider()
    result = run(service(tmp_path, provider, dims=dims).request(PRODUCT))
    assert result.status == "unavailable" and result.message.startswith(UNVERIFIED_MESSAGE)
    assert provider.calls == 0 and result.scale is None


@pytest.mark.parametrize(("error", "status", "retryable"), [
    (ProductImageError("The selected product has no image.", retryable=False), "unavailable", False),
    (ProductImageError("Downloading the product image timed out.", retryable=True), "failed", True),
])
def test_image_problems(tmp_path: Path, error, status, retryable) -> None:
    async def download(url):
        raise error

    provider = FakeProvider()
    result = run(service(tmp_path, provider, downloader=download).request(PRODUCT))
    assert result.status == status and result.retryable is retryable and provider.calls == 0


@pytest.mark.parametrize("outcome", [
    ReconstructionError("3D reconstruction timed out.", retryable=True, code="timeout"),
    ReconstructionError("The GPU ran out of memory.", retryable=True, code="gpu_out_of_memory"),
    b"glTF-garbage",
])
def test_provider_failures_fail_safely_and_are_not_cached(tmp_path: Path, outcome) -> None:
    provider = FakeProvider(outcome)

    async def flow():
        svc = service(tmp_path, provider)
        first = await svc.request(PRODUCT)
        await settle(svc, first.assetId)
        failed = svc.status(first.assetId)
        svc._jobs[first.assetId].finished_at -= 60  # after the short failure memory, a new request retries
        retry = await svc.request(PRODUCT)
        await settle(svc, first.assetId)
        return failed, retry

    failed, retry = run(flow())
    assert failed.status == "failed" and failed.retryable and failed.message
    assert retry.status in ("generating", "failed") and provider.calls == 2
    assert not list((tmp_path / "assets").glob("*/model.glb"))


# ---- Endpoints & safe retrieval ---------------------------------------------

def test_endpoints_and_asset_id_safety(tmp_path: Path) -> None:
    cache = ProductSearchCache(tmp_path / "c.json")
    cache.put("serpapi", "chair", "chair", [PRODUCT])
    svc = service(tmp_path)
    app.dependency_overrides[get_product_cache] = lambda: cache
    app.dependency_overrides[get_ar_preview_service] = lambda: svc
    try:
        with TestClient(app) as client:
            assert client.post("/api/v1/products/ar-preview", json={"productId": "serpapi:x", "productUrl": "u"}).status_code == 404
            assert client.post("/api/v1/products/ar-preview", json={"productId": PRODUCT.id, "productUrl": PRODUCT.productUrl,
                                                                     "widthMeters": 9}).status_code == 422
            started = client.post("/api/v1/products/ar-preview", json={"productId": PRODUCT.id, "productUrl": PRODUCT.productUrl}).json()
            asset_id = started["assetId"]
            for _ in range(200):
                body = client.get(f"/api/v1/ar-assets/{asset_id}").json()
                if body["status"] != "generating":
                    break
            assert body["status"] == "ready"
            glb = client.get(body["assetUrl"])
            assert glb.status_code == 200 and glb.content[:4] == b"glTF"
            for bad in ("..%2F..%2Fmain.py", "a" * 32, "ABC", "0" * 31 + "g", f"{asset_id}%2F..%2Fmeta.json"):
                assert client.get(f"/api/v1/ar-assets/{bad}/model.glb").status_code == 404
                assert client.get(f"/api/v1/ar-assets/{bad}").status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_store_rejects_paths_outside_root(tmp_path: Path) -> None:
    store = ArAssetStore(tmp_path)
    assert store.glb_path("../etc") is None and store.load_meta("a" * 32) is None
    with pytest.raises(ValueError):
        store.save("../../x", {"model.glb": b"x"}, {})


# ---- Image download ----------------------------------------------------------

def png_bytes(size=(256, 256)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_image_download_validation() -> None:
    def handler(request):
        path = request.url.path
        if path == "/ok.png":
            return httpx.Response(200, content=png_bytes())
        if path == "/tiny.png":
            return httpx.Response(200, content=png_bytes((40, 40)))
        if path == "/html":
            return httpx.Response(200, text="<html>nope</html>")
        if path == "/slow":
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    ok = run(download_product_image("https://img.example.com/ok.png", transport=transport))
    assert ok.width == 256 and ok.format == "PNG" and len(ok.sha256) == 64
    for path, retryable in (("/tiny.png", False), ("/html", False), ("/missing", False), ("/slow", True)):
        with pytest.raises(ProductImageError) as error:
            run(download_product_image(f"https://img.example.com{path}", transport=transport))
        assert error.value.retryable is retryable
    for url in (None, "http://127.0.0.1/x.png", "file:///etc/passwd"):
        with pytest.raises(ProductImageError):
            run(download_product_image(url, transport=transport))


# ---- Sidecar provider wire format ----------------------------------------------

def test_sidecar_provider_maps_errors_and_decodes() -> None:
    glb = glb_of(textured_box())

    def handler(request):
        mode = request.headers.get("x-mode", "")
        return {
            "oom": httpx.Response(507, json={"code": "gpu_out_of_memory"}),
            "bad": httpx.Response(400, json={"code": "invalid_image", "detail": "too small"}),
            "junk": httpx.Response(200, json={"nope": 1}),
        }.get(mode, httpx.Response(200, json={"glbBase64": base64.b64encode(glb).decode(), "timings": {"totalSeconds": 2.0},
                                                "model": "sf3d", "version": "x", "device": "cuda"}))

    class Header(httpx.MockTransport):
        def __init__(self, mode):
            super().__init__(lambda r: handler(httpx.Request(r.method, r.url, headers={"x-mode": mode})))

    ok = run(SidecarReconstructionProvider("http://127.0.0.1:8010", transport=Header("")).reconstruct(b"img"))
    assert ok.glb == glb and ok.timings == {"totalSeconds": 2.0}
    for mode, code, retryable in (("oom", "gpu_out_of_memory", True), ("bad", "invalid_image", False), ("junk", "malformed_response", True)):
        with pytest.raises(ReconstructionError) as error:
            run(SidecarReconstructionProvider("http://127.0.0.1:8010", transport=Header(mode)).reconstruct(b"img"))
        assert error.value.code == code and error.value.retryable is retryable

    def down(request):
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ReconstructionError) as error:
        run(SidecarReconstructionProvider("http://127.0.0.1:8010", transport=httpx.MockTransport(down)).reconstruct(b"img"))
    assert error.value.code == "service_unavailable" and error.value.retryable
