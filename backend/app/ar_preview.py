"""Real-product AR preview: selected product -> image -> reconstruction -> normalized GLB.

Physical size never comes from the mesh: the client scales the normalized asset to
the M5 verified width/depth/height returned alongside it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from .ar_geometry import AssetValidationError, compute_scale, export_glb, load_single_mesh, map_axes_by_mesh, normalize_mesh
from .ar_models import ArAssetResponse, ArScaleModel, MeshBoundsModel
from .dimension_models import ResolvedDimensions
from .product_image import ProductImage, ProductImageError, download_product_image
from .product_models import ProductCandidate
from .reconstruction import ReconstructionError, ReconstructionProvider

logger = logging.getLogger(__name__)
NORMALIZATION_VERSION = "norm-v1"
DIMENSION_IDENTITY_VERSION = "dimensions-v1"
ASSET_ID_RE = re.compile(r"^[a-f0-9]{32}$")
FAILURE_MEMORY_SECONDS = 20.0
UNVERIFIED_MESSAGE = "Real-scale preview unavailable — product dimensions could not be verified."


def dimension_fingerprint(dims: ResolvedDimensions) -> str:
    """Stable identity for the exact measurements used to scale an AR asset."""
    payload = {
        "version": DIMENSION_IDENTITY_VERSION,
        "productId": dims.productId,
        "values": dims.dimensionsMeters,
        "width": dims.widthMeters,
        "depth": dims.depthMeters,
        "height": dims.heightMeters,
        "axis": dims.axisMapping.model_dump() if dims.axisMapping else None,
        "sourceUrl": dims.sourceUrl,
        "sourcePath": dims.sourcePath,
        "variantScope": dims.variantScope,
        "variantIdentity": dims.variantIdentity,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def asset_fingerprint(
    product_id: str,
    product_url: str,
    image_sha256: str,
    dimension_identity: str,
    provider: str,
    provider_version: str,
) -> str:
    raw = "|".join((
        product_id,
        product_url,
        image_sha256,
        DIMENSION_IDENTITY_VERSION,
        dimension_identity,
        provider,
        provider_version,
        NORMALIZATION_VERSION,
    ))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class ArAssetStore:
    """Generated assets on disk (git-ignored). Only IDs matching ASSET_ID_RE are ever used as paths."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _dir(self, asset_id: str) -> Path | None:
        if not ASSET_ID_RE.fullmatch(asset_id or ""):
            return None
        path = (self.root / asset_id).resolve()
        return path if path.parent == self.root else None

    def glb_path(self, asset_id: str) -> Path | None:
        directory = self._dir(asset_id)
        if directory is None:
            return None
        path = directory / "model.glb"
        return path if path.is_file() else None

    def load_meta(self, asset_id: str) -> dict | None:
        directory = self._dir(asset_id)
        if directory is None or self.glb_path(asset_id) is None:
            return None
        try:
            meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return meta if isinstance(meta, dict) and meta.get("status") == "ready" else None

    def save(self, asset_id: str, files: dict[str, bytes], meta: dict, *, generation_started: float | None = None) -> None:
        directory = self._dir(asset_id)
        if directory is None:
            raise ValueError("invalid asset id")
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{asset_id}-", dir=self.root))
        try:
            write_started = time.perf_counter()
            for name, data in files.items():
                (staging / name).write_bytes(data)
            timings = meta.setdefault("timings", {})
            timings["writeSeconds"] = round(time.perf_counter() - write_started, 3)
            if generation_started is not None:
                timings["totalGenerationSeconds"] = round(time.perf_counter() - generation_started, 3)
            (staging / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            if directory.exists():
                shutil.rmtree(directory, ignore_errors=True)
            os.replace(staging, directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


@dataclass
class _Job:
    status: str = "generating"
    message: str | None = None
    retryable: bool = False
    finished_at: float | None = None
    task: asyncio.Task | None = None
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class _RequestContext:
    product: ProductCandidate
    dims: ResolvedDimensions
    image_url: str
    request_started: float
    dimension_seconds: float = 0.0
    dimension_cache_hit: bool = False
    dimension_path: str = "service_lookup"
    dimension_timings: dict[str, float] = field(default_factory=dict)
    image_seconds: float = 0.0
    asset_cache_seconds: float = 0.0
    cache_outcome: str = "miss"


@dataclass(frozen=True)
class DimensionLookupResult:
    """Dimension lookup diagnostics supplied by the app-level shared cache."""

    dimensions: ResolvedDimensions
    cache_hit: bool
    path: str
    timings: dict[str, float] = field(default_factory=dict)


DimensionLookup = Callable[[ProductCandidate], Awaitable[ResolvedDimensions | DimensionLookupResult]]
ImageDownloader = Callable[[str | None], Awaitable[ProductImage]]


class ArPreviewService:
    def __init__(self, store: ArAssetStore, provider: ReconstructionProvider, dimensions: DimensionLookup,
                 image_downloader: ImageDownloader = download_product_image) -> None:
        self._store = store
        self._provider = provider
        self._dimensions = dimensions
        self._download = image_downloader
        self._jobs: dict[str, _Job] = {}
        self._contexts: dict[str, _RequestContext] = {}
        self._gpu = asyncio.Semaphore(1)

    async def request(self, product: ProductCandidate) -> ArAssetResponse:
        request_started = time.perf_counter()
        logger.info(
            "AR preview selected product id=%s title=%r imageUrl=%s",
            product.id, product.title, product.imageUrl,
        )
        base = {"sourceProductId": product.id, "sourceImageUrl": product.imageUrl}
        dimension_started = time.perf_counter()
        lookup = await self._dimensions(product)
        dimension_seconds = round(time.perf_counter() - dimension_started, 3)
        if isinstance(lookup, DimensionLookupResult):
            dims = lookup.dimensions
            dimension_cache_hit = lookup.cache_hit
            dimension_path = lookup.path
            dimension_timings = {
                f"dimension.{key}": float(value)
                for key, value in lookup.timings.items()
                if isinstance(value, (int, float))
            }
        else:
            dims = lookup
            dimension_cache_hit = False
            dimension_path = "service_lookup"
            dimension_timings = {}
        initial_timings = {
            **dimension_timings,
            "dimensionResolutionSeconds": dimension_seconds,
        }
        dimension_source = _dimension_source(
            dims,
            lookup_path=dimension_path,
            lookup_cache_hit=dimension_cache_hit,
        )
        logger.info(
            "AR dimension lookup product=%s cacheHit=%s path=%s seconds=%s status=%s sourcePath=%s",
            product.id, dimension_cache_hit, dimension_path, dimension_seconds, dims.status, dims.sourcePath,
        )
        if len(dims.dimensionsMeters or []) != 3:
            detail = "" if dims.status == "unavailable" else \
                f" (only {len(dims.dimensionsMeters or [])} of 3 dimensions verified)"
            return ArAssetResponse(status="unavailable", message=UNVERIFIED_MESSAGE + detail,
                                   retryable=dims.retryable, dimensionSource=dimension_source,
                                   timings={**initial_timings,
                                            "requestTotalSeconds": round(time.perf_counter() - request_started, 3)},
                                   **base)
        try:
            image = await self._download(product.imageUrl)
        except ProductImageError as exc:
            return ArAssetResponse(status="unavailable" if not exc.retryable else "failed",
                                   message=f"3D preview unavailable: {exc}", retryable=exc.retryable,
                                   dimensionSource=dimension_source,
                                   timings={**initial_timings,
                                            "requestTotalSeconds": round(time.perf_counter() - request_started, 3)},
                                   **base)
        logger.info(
            "AR reconstruction image downloaded product=%s url=%s sha256=%s pixels=%sx%s bytes=%s",
            product.id, image.url, image.sha256, image.width, image.height, len(image.data),
        )
        dimension_identity = dimension_fingerprint(dims)
        asset_id = asset_fingerprint(
            product.id,
            product.productUrl,
            image.sha256,
            dimension_identity,
            self._provider.name,
            self._provider.version,
        )
        context = _RequestContext(
            product=product,
            dims=dims,
            image_url=image.url,
            request_started=request_started,
            dimension_seconds=dimension_seconds,
            dimension_cache_hit=dimension_cache_hit,
            dimension_path=dimension_path,
            dimension_timings=dimension_timings,
            image_seconds=image.download_seconds,
        )
        self._contexts[asset_id] = context
        cache_started = time.perf_counter()
        meta = self._store.load_meta(asset_id)
        context.asset_cache_seconds = round(time.perf_counter() - cache_started, 3)
        if meta is not None:
            context.cache_outcome = "hit"
            logger.info("AR asset cache product=%s asset=%s outcome=hit seconds=%s",
                        product.id, asset_id, context.asset_cache_seconds)
            return self._ready(asset_id, meta, context, cached=True)
        job = self._jobs.get(asset_id)
        if job and job.status == "failed" and job.finished_at and time.monotonic() - job.finished_at > FAILURE_MEMORY_SECONDS:
            job = None  # failures are not cached: a later request retries
        if job is None:
            context.cache_outcome = "miss"
            job = _Job()
            self._jobs[asset_id] = job
            job.task = asyncio.create_task(self._generate(asset_id, image, dimension_identity, job))
        else:
            context.cache_outcome = "joined"
        logger.info("AR asset cache product=%s asset=%s outcome=%s seconds=%s",
                    product.id, asset_id, context.cache_outcome, context.asset_cache_seconds)
        return self._from_job(asset_id, job, context)

    def status(self, asset_id: str) -> ArAssetResponse | None:
        if not ASSET_ID_RE.fullmatch(asset_id or ""):
            return None
        context = self._contexts.get(asset_id)
        if context is None:
            return None  # unknown in this backend session: the client re-POSTs
        meta = self._store.load_meta(asset_id)
        if meta is not None:
            return self._ready(asset_id, meta, context, cached=False)
        job = self._jobs.get(asset_id)
        return self._from_job(asset_id, job, context) if job else None

    def glb_path(self, asset_id: str) -> Path | None:
        return self._store.glb_path(asset_id)

    async def _generate(self, asset_id: str, image: ProductImage, dimension_identity: str, job: _Job) -> None:
        started = time.perf_counter()
        try:
            logger.info(
                "SF3D input asset=%s imageSha256=%s pixels=%sx%s bytes=%s",
                asset_id, image.sha256, image.width, image.height, len(image.data),
            )
            queue_started = time.perf_counter()
            async with self._gpu:
                queue_seconds = round(time.perf_counter() - queue_started, 3)
                provider_started = time.perf_counter()
                output = await self._provider.reconstruct(image.data)
                provider_call_seconds = round(time.perf_counter() - provider_started, 3)

            mark = time.perf_counter()
            raw_mesh = load_single_mesh(output.glb)
            mesh_load_seconds = round(time.perf_counter() - mark, 3)

            mark = time.perf_counter()
            normalized, info = normalize_mesh(raw_mesh)
            normalize_seconds = round(time.perf_counter() - mark, 3)
            logger.info(
                "SF3D mesh asset=%s originalBounds=%s normalizedBounds=%s",
                asset_id, info.original_bounds.as_dict(), info.normalized_bounds.as_dict(),
            )

            mark = time.perf_counter()
            glb = export_glb(normalized)
            normalized_export_seconds = round(time.perf_counter() - mark, 3)

            mark = time.perf_counter()
            load_single_mesh(glb)  # the served file must itself be a valid single-mesh GLB
            validation_seconds = round(time.perf_counter() - mark, 3)
            timings = {
                **{f"provider.{k}": v for k, v in output.timings.items()},
                "gpuQueueSeconds": queue_seconds,
                "providerCallSeconds": provider_call_seconds,
                "meshLoadSeconds": mesh_load_seconds,
                "normalizeSeconds": normalize_seconds,
                "normalizedExportSeconds": normalized_export_seconds,
                "validationSeconds": validation_seconds,
            }
            job.timings = timings
            meta = {
                "status": "ready", "assetId": asset_id, "provider": self._provider.name,
                "providerVersion": self._provider.version, "normalizationVersion": NORMALIZATION_VERSION,
                "dimensionIdentityVersion": DIMENSION_IDENTITY_VERSION,
                "dimensionIdentity": dimension_identity,
                "generatedAt": datetime.now(timezone.utc).isoformat(), "sourceImageUrl": image.url,
                "sourceImageSha256": image.sha256, "sourceImageSize": [image.width, image.height],
                "originalBounds": info.original_bounds.as_dict(), "normalizedBounds": info.normalized_bounds.as_dict(),
                "normalization": {"steps": info.steps, "yawDegrees": info.yaw_degrees, "vertexCount": info.vertex_count,
                                  "faceCount": info.face_count, "textured": info.textured, "upAxis": "+Y",
                                  "axisMapping": {"x": "width", "y": "height", "z": "depth"}},
                "timings": timings, "glbBytes": len(glb), "providerMeta": output.provider_meta,
            }
            files = {"model.glb": glb, "raw.glb": output.glb, "source-image": image.data}
            if output.processed_image_png:
                files["processed.png"] = output.processed_image_png
            self._store.save(asset_id, files, meta, generation_started=started)
            job.timings = dict(meta["timings"])
            job.status = "ready"
            logger.info("AR asset %s ready: %s, %d bytes", asset_id, timings, len(glb))
        except ReconstructionError as exc:
            job.status, job.message, job.retryable = "failed", str(exc), exc.retryable
        except AssetValidationError as exc:
            job.status, job.message, job.retryable = "failed", f"The generated 3D model was unusable: {exc}", True
        except Exception as exc:  # never crash the backend over one preview
            logger.exception("AR asset generation failed")
            job.status, job.message, job.retryable = "failed", f"3D preview generation failed ({type(exc).__name__}).", True
        finally:
            job.timings.setdefault("totalGenerationSeconds", round(time.perf_counter() - started, 3))
            job.finished_at = time.monotonic()

    def _from_job(self, asset_id: str, job: _Job, context: _RequestContext) -> ArAssetResponse:
        if job.status == "ready":
            meta = self._store.load_meta(asset_id)
            if meta is not None:
                return self._ready(asset_id, meta, context, cached=False)
        return ArAssetResponse(
            assetId=asset_id, status="failed" if job.status == "failed" else "generating",
            sourceProductId=context.product.id, sourceImageUrl=context.image_url,
            reconstructionProvider=self._provider.name, providerVersion=self._provider.version,
            dimensionSource=_context_dimension_source(context),
            timings={**job.timings, **_request_timings(context)},
            cacheOutcome=context.cache_outcome,
            message=job.message if job.status == "failed" else "Generating 3D preview…",
            retryable=job.retryable if job.status == "failed" else False,
        )

    def _ready(self, asset_id: str, meta: dict, context: _RequestContext, cached: bool) -> ArAssetResponse:
        dims = context.dims
        size = tuple(meta["normalizedBounds"]["size"])
        if dims.widthMeters and dims.depthMeters and dims.heightMeters:
            width, depth, height, mapping = dims.widthMeters, dims.depthMeters, dims.heightMeters, "labeled"
            axis = dims.axisMapping
            width_index = axis.widthIndex if axis else 0
            depth_index = axis.depthIndex if axis else 1
            height_index = axis.heightIndex if axis else 2
            mapping_confidence = axis.confidence if axis else 1.0
            mapping_mismatch = 0.0
            mapping_reason = axis.reason if axis else "axes explicitly labeled by the retailer source"
        else:
            source_axis = dims.axisMapping
            mapped = map_axes_by_mesh(
                size,
                list(dims.dimensionsMeters or []),
                width_index=source_axis.widthIndex if source_axis else None,
                depth_index=source_axis.depthIndex if source_axis else None,
                height_index=source_axis.heightIndex if source_axis else None,
            )
            logger.info(
                "Mesh axis mapping asset=%s sourceValues=%s meshSize=%s accepted=%s permutation=%s "
                "mismatch=%s confidence=%s reason=%s",
                asset_id, dims.dimensionsMeters, size, mapped.accepted, mapped.permutation,
                mapped.mismatch, mapped.confidence, mapped.reason,
            )
            if not mapped.accepted:
                return ArAssetResponse(
                    assetId=asset_id,
                    status="unavailable",
                    sourceProductId=context.product.id,
                    sourceImageUrl=meta.get("sourceImageUrl"),
                    reconstructionProvider=meta.get("provider"),
                    providerVersion=meta.get("providerVersion"),
                    dimensionSource={
                        **_context_dimension_source(context),
                        "meshAxisMapping": {
                            "accepted": False,
                            "permutation": mapped.permutation,
                            "confidence": mapped.confidence,
                            "mismatch": mapped.mismatch,
                            "reason": mapped.reason,
                        },
                    },
                    message="Retailer measurements were verified, but the generated mesh could not map their axis order safely. "
                            + mapped.reason.capitalize() + ".",
                    # This normalized asset is cached, so the same request would
                    # produce the same ambiguous mapping rather than recover.
                    retryable=False,
                    cached=cached,
                    cacheOutcome=context.cache_outcome,
                    timings={**(meta.get("timings") or {}), **_request_timings(context)},
                )
            width, depth, height = mapped.width, mapped.depth, mapped.height
            width_index, depth_index, height_index = mapped.permutation
            mapping_confidence, mapping_mismatch, mapping_reason = mapped.confidence, mapped.mismatch, mapped.reason
            mapping = "mesh_assisted"
        scale = compute_scale(size, width, depth, height)
        return ArAssetResponse(
            assetId=asset_id, status="ready", sourceProductId=context.product.id,
            sourceImageUrl=meta.get("sourceImageUrl"), reconstructionProvider=meta.get("provider"),
            providerVersion=meta.get("providerVersion"), assetUrl=f"/api/v1/ar-assets/{asset_id}/model.glb",
            generatedAt=meta.get("generatedAt"), originalBounds=MeshBoundsModel(**meta["originalBounds"]),
            normalizedBounds=MeshBoundsModel(**meta["normalizedBounds"]), normalization=meta.get("normalization"),
            scale=ArScaleModel(widthMeters=width, depthMeters=depth, heightMeters=height,
                               scaleX=scale.scale_x, scaleY=scale.scale_y, scaleZ=scale.scale_z,
                               axesSwapped=scale.axes_swapped, maxAxisDistortion=scale.max_axis_distortion,
                               axisMappingSource=mapping, dimensionsMeters=list(dims.dimensionsMeters or []),
                               widthIndex=width_index, depthIndex=depth_index, heightIndex=height_index,
                               axisMappingConfidence=mapping_confidence,
                               axisMappingMismatch=mapping_mismatch, axisMappingReason=mapping_reason),
            dimensionSource=_context_dimension_source(context),
            timings={**(meta.get("timings") or {}), **_request_timings(context)},
            cacheOutcome=context.cache_outcome,
            glbBytes=meta.get("glbBytes"), cached=cached,
            previewLabel=(
                "AI-generated 3D preview scaled to verified retailer dimensions; "
                "axis order inferred from the generated shape."
                if mapping == "mesh_assisted"
                else "AI-generated 3D preview scaled to verified retailer dimensions."
            ),
        )


def _dimension_source(
    dims: ResolvedDimensions,
    *,
    lookup_path: str | None = None,
    lookup_cache_hit: bool | None = None,
) -> dict[str, object]:
    source: dict[str, object] = {
        "status": dims.status,
        "sourceType": dims.sourceType,
        "sourceName": dims.sourceName,
        "variantScope": dims.variantScope,
        "rawDimensions": dims.rawDimensions,
        "sourcePath": dims.sourcePath,
        "extractionMethod": dims.extractionMethod,
        "axisMapping": dims.axisMapping.model_dump() if dims.axisMapping else None,
    }
    if lookup_path is not None:
        source["lookupPath"] = lookup_path
    if lookup_cache_hit is not None:
        source["lookupCacheHit"] = lookup_cache_hit
    return source


def _context_dimension_source(context: _RequestContext) -> dict[str, object]:
    return _dimension_source(
        context.dims,
        lookup_path=context.dimension_path,
        lookup_cache_hit=context.dimension_cache_hit,
    )


def _request_timings(context: _RequestContext) -> dict[str, float]:
    return {
        **context.dimension_timings,
        "dimensionResolutionSeconds": context.dimension_seconds,
        "imageDownloadSeconds": context.image_seconds,
        "assetCacheLookupSeconds": context.asset_cache_seconds,
        "requestTotalSeconds": round(time.perf_counter() - context.request_started, 3),
    }
