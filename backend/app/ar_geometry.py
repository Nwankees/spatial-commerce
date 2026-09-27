"""Deterministic geometry handling for generated AR preview assets.

The generated mesh only provides *shape*; physical size always comes from M5
verified dimensions. Conventions (glTF / OpenGL / ARCore world): +Y is up,
units are arbitrary until scaled, the reconstruction camera looked along -Z, so
the image's horizontal axis is X (product width) and the viewing direction is
Z (product depth).
"""
from __future__ import annotations

import io
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import trimesh

MAX_VERTICES = 2_000_000
MAX_FACES = 4_000_000
MIN_EXTENT = 1e-6
YAW_MIN_AREA_GAIN = 0.03  # only straighten the footprint when it is clearly rotated
# Two mappings are physically equivalent when every target mesh-axis extent is
# within 3%. This is small enough to preserve materially different shapes while
# absorbing retailer rounding and swaps between nearly equal dimensions.
PHYSICAL_SCALE_EQUIVALENCE_TOLERANCE = 0.03


class AssetValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Bounds:
    min: tuple[float, float, float]
    max: tuple[float, float, float]

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(float(b - a) for a, b in zip(self.min, self.max))  # type: ignore[return-value]

    def as_dict(self) -> dict[str, list[float]]:
        return {"min": [round(v, 6) for v in self.min], "max": [round(v, 6) for v in self.max],
                "size": [round(v, 6) for v in self.size]}


@dataclass
class NormalizationInfo:
    original_bounds: Bounds
    normalized_bounds: Bounds
    yaw_degrees: float
    steps: list[str] = field(default_factory=list)
    vertex_count: int = 0
    face_count: int = 0
    textured: bool = False


@dataclass(frozen=True)
class ScaleResult:
    scale_x: float
    scale_y: float
    scale_z: float
    axes_swapped: bool
    max_axis_distortion: float  # max(scale)/min(scale) - 1: how much the mesh proportions were stretched


@dataclass(frozen=True)
class MeshAxisMapping:
    """Permutation of retailer-backed values onto width/depth/height."""

    width: float
    depth: float
    height: float
    width_index: int
    depth_index: int
    height_index: int
    mismatch: float
    confidence: float
    accepted: bool
    reason: str

    @property
    def permutation(self) -> list[int]:
        return [self.width_index, self.depth_index, self.height_index]


@dataclass(frozen=True)
class _AxisCandidate:
    mismatch: float
    permutation: tuple[int, int, int]
    assigned: tuple[float, float, float]  # width, depth, height

    @property
    def target_xyz(self) -> tuple[float, float, float]:
        # Generated mesh axes: X=width, Y=height, Z=depth.
        return self.assigned[0], self.assigned[2], self.assigned[1]


def bounds_of(mesh: trimesh.Trimesh) -> Bounds:
    lo, hi = mesh.bounds
    return Bounds(tuple(float(v) for v in lo), tuple(float(v) for v in hi))  # type: ignore[arg-type]


def load_single_mesh(glb: bytes) -> trimesh.Trimesh:
    """Loads a GLB containing exactly one triangle mesh (scene transforms applied)."""
    if not glb or glb[:4] != b"glTF":
        raise AssetValidationError("Generated asset is not a GLB file.")
    try:
        loaded = trimesh.load(io.BytesIO(glb), file_type="glb", process=False)
    except Exception as exc:  # trimesh raises many types on bad input
        raise AssetValidationError(f"Generated GLB could not be parsed ({type(exc).__name__}).") from None
    if isinstance(loaded, trimesh.Scene):
        meshes = []
        for node in loaded.graph.nodes_geometry:
            transform, name = loaded.graph[node]
            geometry = loaded.geometry[name]
            if isinstance(geometry, trimesh.Trimesh):
                copy = geometry.copy()
                copy.apply_transform(transform)
                meshes.append(copy)
        if len(meshes) != 1:
            raise AssetValidationError(f"Generated asset must contain exactly one mesh (found {len(meshes)}).")
        mesh = meshes[0]
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise AssetValidationError("Generated asset contains no triangle mesh.")
    validate_mesh(mesh)
    return mesh


def validate_mesh(mesh: trimesh.Trimesh) -> None:
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise AssetValidationError("Generated mesh is empty.")
    if len(mesh.vertices) > MAX_VERTICES or len(mesh.faces) > MAX_FACES:
        raise AssetValidationError("Generated mesh is too large.")
    if not np.all(np.isfinite(mesh.vertices)):
        raise AssetValidationError("Generated mesh has non-finite vertices.")
    if np.any(mesh.faces >= len(mesh.vertices)) or np.any(mesh.faces < 0):
        raise AssetValidationError("Generated mesh has invalid face indices.")
    if min(bounds_of(mesh).size) <= MIN_EXTENT:
        raise AssetValidationError("Generated mesh is flat or degenerate.")


def normalize_mesh(mesh: trimesh.Trimesh, straighten_yaw: bool = True) -> tuple[trimesh.Trimesh, NormalizationInfo]:
    """Up axis stays +Y (provider convention). Steps, all deterministic:
    1. optionally rotate about +Y by the smallest angle (|angle| <= 45°) that makes the
       footprint's minimum-area rectangle axis-aligned (never swaps width/depth);
    2. center the bounding box horizontally (x = z = 0);
    3. put the lowest point on the floor (min y = 0).
    Aspect ratio is preserved; no scaling happens here."""
    mesh = mesh.copy()
    original = bounds_of(mesh)
    steps: list[str] = []
    yaw = 0.0
    if straighten_yaw:
        yaw = footprint_yaw_degrees(np.asarray(mesh.vertices)[:, [0, 2]])
        if yaw:
            # A +Y rotation by a turns (x, z) footprint points by -a, undoing the footprint angle.
            mesh.apply_transform(trimesh.transformations.rotation_matrix(math.radians(yaw), [0, 1, 0]))
            steps.append(f"rotated {yaw:.2f} deg about +Y to align footprint")
    b = bounds_of(mesh)
    offset = np.array([-(b.min[0] + b.max[0]) / 2, -b.min[1], -(b.min[2] + b.max[2]) / 2])
    mesh.apply_translation(offset)
    steps.append("centered horizontally; bottom on y=0")
    textured = isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals) and getattr(mesh.visual, "uv", None) is not None
    info = NormalizationInfo(original, bounds_of(mesh), round(yaw, 4), steps, len(mesh.vertices), len(mesh.faces), textured)
    return mesh, info


def footprint_yaw_degrees(points_xz: np.ndarray) -> float:
    """Angle (degrees, in (-45, 45]) of the footprint's minimum-area bounding rectangle,
    or 0 when straightening would not shrink the axis-aligned footprint noticeably."""
    hull = _convex_hull(np.unique(np.round(points_xz, 7), axis=0))
    if len(hull) < 3:
        return 0.0
    aligned_area = _aabb_area(hull, 0.0)
    best_angle, best_area = 0.0, aligned_area
    edges = np.roll(hull, -1, axis=0) - hull
    for dx, dz in edges:
        angle = math.degrees(math.atan2(dz, dx))
        angle = ((angle + 45.0) % 90.0) - 45.0  # equivalent rectangle orientation within (-45, 45]
        area = _aabb_area(hull, angle)
        if area < best_area - 1e-12:
            best_angle, best_area = angle, area
    if aligned_area <= 0 or (aligned_area - best_area) / aligned_area < YAW_MIN_AREA_GAIN:
        return 0.0
    return round(best_angle, 4)


def _aabb_area(points: np.ndarray, angle_degrees: float) -> float:
    a = math.radians(-angle_degrees)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    p = points @ rot.T
    span = p.max(axis=0) - p.min(axis=0)
    return float(span[0] * span[1])


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain (no scipy dependency)."""
    pts = sorted(map(tuple, points))
    if len(pts) <= 2:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def export_glb(mesh: trimesh.Trimesh) -> bytes:
    data = mesh.export(file_type="glb", include_normals=True)
    if not isinstance(data, (bytes, bytearray)) or data[:4] != b"glTF":
        raise AssetValidationError("Normalized asset could not be exported as GLB.")
    return bytes(data)


def assign_axes_by_mesh(normalized_size: tuple[float, float, float], values: list[float],
                        height_index: int | None = None) -> tuple[float, float, float]:
    """Axis assignment for verified values whose axis order the source does not label.

    Only the mesh's *proportions* (which extent is longest/shortest) are used to decide
    which verified number belongs to which axis; every size is still a verified value,
    never a mesh measurement. With a labeled height, only width vs depth is decided.
    Returns (width, depth, height) in meters.
    """
    mapping = map_axes_by_mesh(normalized_size, values, height_index=height_index)
    if not mapping.accepted:
        raise ValueError(mapping.reason)
    return mapping.width, mapping.depth, mapping.height


def map_axes_by_mesh(
    normalized_size: tuple[float, float, float],
    values: list[float],
    *,
    width_index: int | None = None,
    depth_index: int | None = None,
    height_index: int | None = None,
    max_mismatch: float = 0.55,
    min_margin: float = 0.08,
    physical_scale_tolerance: float = PHYSICAL_SCALE_EQUIVALENCE_TOLERANCE,
) -> MeshAxisMapping:
    """Score every source-compatible permutation using only mesh aspect ratios.

    The three retailer values remain unchanged. Mesh X/Y/Z correspond to
    width/height/depth. Permutations are first grouped by their resulting
    physical X/Y/Z target extents; equivalent or near-equivalent scale outcomes
    do not count as ambiguity. A poor match or a near-tie between materially
    different outcomes is still rejected instead of guessed.
    """
    if len(values) != 3 or any(v <= 0 or not math.isfinite(v) for v in values):
        raise ValueError("Three positive verified dimensions are required.")
    ex, ey, ez = normalized_size
    if min(ex, ey, ez) <= MIN_EXTENT or not all(math.isfinite(v) for v in normalized_size):
        raise ValueError("The normalized mesh has invalid extents.")
    if not 0 <= physical_scale_tolerance < 1:
        raise ValueError("Physical-scale equivalence tolerance must be between 0 and 1.")
    fixed = (width_index, depth_index, height_index)
    known = [index for index in fixed if index is not None]
    if any(index not in (0, 1, 2) for index in known) or len(set(known)) != len(known):
        raise ValueError("Source axis indices are invalid.")

    candidates: list[_AxisCandidate] = []
    for permutation in itertools.permutations(range(3)):
        if width_index is not None and permutation[0] != width_index:
            continue
        if depth_index is not None and permutation[1] != depth_index:
            continue
        if height_index is not None and permutation[2] != height_index:
            continue
        assigned = tuple(values[index] for index in permutation)
        candidates.append(_AxisCandidate(_shape_mismatch((ex, ez, ey), assigned), permutation, assigned))
    if not candidates:
        raise ValueError("No axis permutation is compatible with the source labels.")
    candidates.sort(key=lambda item: item.mismatch)
    outcome_classes = _physical_outcome_classes(candidates, physical_scale_tolerance)
    representatives = sorted(
        (min(group, key=lambda item: item.mismatch) for group in outcome_classes),
        key=lambda item: item.mismatch,
    )
    best = representatives[0]
    best_score, best_permutation, best_values = best.mismatch, best.permutation, best.assigned
    second_score = representatives[1].mismatch if len(representatives) > 1 else math.inf
    margin = second_score - best_score
    poor = best_score > max_mismatch
    ambiguous = math.isfinite(second_score) and margin < min_margin
    accepted = not poor and not ambiguous
    shape_quality = max(0.0, 1.0 - best_score / max_mismatch)
    separation = 1.0 if not math.isfinite(second_score) else min(1.0, max(0.0, margin / 0.30))
    confidence = round(shape_quality * separation, 4)
    if poor:
        reason = f"mesh aspect ratios disagree with every retailer-dimension permutation (mismatch {best_score:.3f})"
    elif ambiguous:
        reason = (
            "mesh proportions do not distinguish materially different physical scale outcomes "
            f"(score margin {margin:.3f}; {len(outcome_classes)} distinct outcomes)"
        )
    else:
        second = "n/a" if not math.isfinite(second_score) else f"{margin:.3f}"
        reason = (
            f"mesh aspect-ratio match selected source indices "
            f"W={best_permutation[0]}, D={best_permutation[1]}, H={best_permutation[2]} "
            f"(mismatch {best_score:.3f}, distinct-outcome margin {second}; "
            f"{len(candidates)} permutations collapsed to {len(outcome_classes)} physical outcomes "
            f"at {physical_scale_tolerance:.1%} tolerance)"
        )
    return MeshAxisMapping(
        width=best_values[0],
        depth=best_values[1],
        height=best_values[2],
        width_index=best_permutation[0],
        depth_index=best_permutation[1],
        height_index=best_permutation[2],
        mismatch=round(best_score, 6),
        confidence=confidence,
        accepted=accepted,
        reason=reason,
    )


def _physical_outcome_classes(
    candidates: list[_AxisCandidate],
    tolerance: float,
) -> list[list[_AxisCandidate]]:
    """Groups only outcomes whose target X/Y/Z extents are pairwise equivalent.

    Complete-link grouping avoids transitive drift where a chain of individually
    close outcomes could collapse two endpoints that differ materially.
    """
    groups: list[list[_AxisCandidate]] = []
    for candidate in candidates:
        group = next(
            (items for items in groups
             if all(_scale_vectors_equivalent(candidate.target_xyz, item.target_xyz, tolerance)
                    for item in items)),
            None,
        )
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)
    return groups


def _scale_vectors_equivalent(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
    tolerance: float,
) -> bool:
    return all(abs(a - b) / max(a, b) <= tolerance for a, b in zip(left, right))


def _shape_mismatch(mesh_semantic: tuple[float, float, float],
                    dimensions_semantic: tuple[float, float, float]) -> float:
    mesh_logs = np.log(np.asarray(mesh_semantic, dtype=float))
    dimension_logs = np.log(np.asarray(dimensions_semantic, dtype=float))
    mesh_logs -= mesh_logs.mean()
    dimension_logs -= dimension_logs.mean()
    return float(np.sqrt(np.mean(np.square(mesh_logs - dimension_logs))))


def compute_scale(normalized_size: tuple[float, float, float], width_m: float, depth_m: float,
                  height_m: float, swap_tolerance: float = 0.15) -> ScaleResult:
    """Per-axis scale that makes the normalized mesh exactly W x H x D meters.

    Axis mapping: mesh X -> width, mesh Y -> height, mesh Z -> depth (reconstruction
    camera convention). If the mesh footprint's long axis clearly disagrees with the
    verified dimensions' long axis (both differ by more than ``swap_tolerance``), the
    horizontal axes are swapped (the client applies a 90° yaw) — the mapping follows
    verified data, never the mesh's own size. Physical size comes only from W/D/H.
    """
    ex, ey, ez = normalized_size
    if min(ex, ey, ez) <= MIN_EXTENT:
        raise AssetValidationError("Normalized mesh has a degenerate extent.")
    for value in (width_m, depth_m, height_m):
        if not (value and value > 0):
            raise ValueError("Verified width, depth and height are all required.")
    mesh_long_is_x = ex >= ez
    dims_long_is_width = width_m >= depth_m
    mesh_clear = abs(ex - ez) / max(ex, ez) > swap_tolerance
    dims_clear = abs(width_m - depth_m) / max(width_m, depth_m) > swap_tolerance
    swapped = mesh_clear and dims_clear and mesh_long_is_x != dims_long_is_width
    width_extent, depth_extent = (ez, ex) if swapped else (ex, ez)
    # After a 90° yaw the mesh's Z axis lies along world width: scale factors are in
    # mesh-local axes, applied before the yaw.
    sx = (depth_m / ex) if swapped else (width_m / width_extent)
    sz = (width_m / ez) if swapped else (depth_m / depth_extent)
    sy = height_m / ey
    factors = (sx, sy, sz)
    return ScaleResult(round(sx, 6), round(sy, 6), round(sz, 6), swapped, round(max(factors) / min(factors) - 1, 4))
