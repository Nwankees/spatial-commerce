from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ArAssetStatus = Literal["ready", "generating", "unavailable", "failed"]


class ArPreviewRequest(BaseModel):
    """Identifies a product this backend returned from /products/search (same rule as M5)."""

    model_config = ConfigDict(extra="forbid")

    productId: str = Field(min_length=1, max_length=200)
    productUrl: str = Field(min_length=1, max_length=2000)


class MeshBoundsModel(BaseModel):
    min: list[float]
    max: list[float]
    size: list[float]


class ArScaleModel(BaseModel):
    """How the normalized mesh (mesh-local axes) is scaled to verified meters.

    The client renders: anchor * yaw(user) * yaw(90° if axesSwapped) * scale(scaleX, scaleY, scaleZ).
    """

    widthMeters: float
    depthMeters: float
    heightMeters: float
    scaleX: float
    scaleY: float
    scaleZ: float
    axesSwapped: bool
    maxAxisDistortion: float
    # "labeled": the source labels width/depth/height. "mesh_assisted": the source gives three
    # unlabeled values; which value is which axis was decided from the mesh's proportions only.
    axisMappingSource: Literal["labeled", "mesh_assisted"] = "labeled"
    dimensionsMeters: list[float] | None = None
    widthIndex: int | None = None
    depthIndex: int | None = None
    heightIndex: int | None = None
    axisMappingConfidence: float = Field(default=1.0, ge=0.0, le=1.0)
    axisMappingMismatch: float = Field(default=0.0, ge=0.0)
    axisMappingReason: str | None = None


class ArAssetResponse(BaseModel):
    assetId: str | None = None
    status: ArAssetStatus
    sourceProductId: str
    sourceImageUrl: str | None = None
    reconstructionProvider: str | None = None
    providerVersion: str | None = None
    assetFormat: Literal["glb"] = "glb"
    assetUrl: str | None = Field(default=None, description="Relative URL of the normalized GLB; never a filesystem path.")
    generatedAt: str | None = None
    originalBounds: MeshBoundsModel | None = None
    normalizedBounds: MeshBoundsModel | None = None
    normalization: dict[str, object] | None = None
    scale: ArScaleModel | None = None
    dimensionSource: dict[str, object] | None = None
    timings: dict[str, float] = Field(default_factory=dict)
    cacheOutcome: Literal["hit", "miss", "joined"] | None = Field(
        default=None,
        description="Whether this request reused a finished asset, started generation, or joined an existing job.",
    )
    glbBytes: int | None = None
    cached: bool = False
    previewLabel: str = "AI-generated 3D preview scaled to verified product dimensions."
    message: str | None = None
    retryable: bool = False
