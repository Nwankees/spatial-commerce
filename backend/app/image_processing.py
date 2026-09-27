from __future__ import annotations

import base64
import binascii
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from .models import AnalyzeProductRequest
from .product_models import ProductSearchRequest


class InvalidImageError(ValueError):
    pass


def prepare_image(request: AnalyzeProductRequest, max_image_bytes: int) -> bytes:
    return _prepare_image(
        request.imageBase64,
        request.rotationDegrees,
        max_image_bytes,
    )


def prepare_search_image(request: ProductSearchRequest, max_image_bytes: int) -> bytes | None:
    """Validates the original analyzed frame for visual retrieval.

    Older/text-only clients may omit the image and continue to work.
    """
    if request.imageBase64 is None:
        return None
    return _prepare_image(request.imageBase64, request.rotationDegrees, max_image_bytes)


def prepare_lens_upload(image_bytes: bytes, max_bytes: int = 500_000) -> bytes:
    """Produces a Lens-compatible JPEG without changing the source kept by Android.

    SerpApi's Image API currently caps uploads at 500 KB. Quality is reduced
    first; oversized camera frames are then downscaled in bounded steps.
    """
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The visual-search image is not readable.") from exc

    for _ in range(7):
        for quality in (86, 76, 66, 56, 46):
            output = BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            data = output.getvalue()
            if len(data) <= max_bytes:
                return data
        width, height = image.size
        if min(width, height) <= 320:
            break
        image.thumbnail((max(320, int(width * 0.8)), max(320, int(height * 0.8))), Image.Resampling.LANCZOS)
    raise InvalidImageError("The camera image could not be prepared for visual search.")


def _prepare_image(image_base64: str, rotation_degrees: int, max_image_bytes: int) -> bytes:
    try:
        raw = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("imageBase64 is not valid Base64 data.") from exc

    if not raw:
        raise InvalidImageError("The captured image is empty.")
    if len(raw) > max_image_bytes:
        raise InvalidImageError(
            f"The captured image exceeds the {max_image_bytes // 1_000_000} MB limit."
        )

    try:
        with Image.open(BytesIO(raw)) as source:
            source.load()
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The uploaded data is not a readable image.") from exc

    if rotation_degrees:
        image = image.rotate(-rotation_degrees, expand=True)

    output = BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()
