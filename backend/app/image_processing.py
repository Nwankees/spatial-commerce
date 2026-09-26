from __future__ import annotations

import base64
import binascii
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from .models import AnalyzeProductRequest


class InvalidImageError(ValueError):
    pass


def prepare_image(request: AnalyzeProductRequest, max_image_bytes: int) -> bytes:
    try:
        raw = base64.b64decode(request.imageBase64, validate=True)
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

    if request.rotationDegrees:
        image = image.rotate(-request.rotationDegrees, expand=True)

    output = BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()
