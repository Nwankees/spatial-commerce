from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass

import httpx
from PIL import Image, UnidentifiedImageError

from .page_fetcher import BROWSER_USER_AGENT, is_public_http_url

MAX_IMAGE_BYTES = 12_000_000
MIN_IMAGE_SIDE = 128
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP"}


class ProductImageError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ProductImage:
    url: str
    data: bytes
    sha256: str
    width: int
    height: int
    format: str
    download_seconds: float


async def download_product_image(url: str | None, *, timeout_seconds: float = 10.0,
                                 transport: httpx.AsyncBaseTransport | None = None) -> ProductImage:
    """Downloads and validates the selected product's image. Never modifies it."""
    if not url:
        raise ProductImageError("The selected product has no image.", retryable=False)
    if not is_public_http_url(url):
        raise ProductImageError("The product image URL is not a public web address.", retryable=False)
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True, max_redirects=5,
                                     transport=transport) as client:
            async with client.stream("GET", url, headers={"User-Agent": BROWSER_USER_AGENT,
                                                          "Accept": "image/*"}) as response:
                if not is_public_http_url(str(response.url)):
                    raise ProductImageError("The product image redirected to a non-public address.", retryable=False)
                if response.status_code != 200:
                    raise ProductImageError(f"The product image returned HTTP {response.status_code}.",
                                            retryable=response.status_code in (408, 429) or response.status_code >= 500)
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_IMAGE_BYTES:
                        raise ProductImageError("The product image is too large.", retryable=False)
    except ProductImageError:
        raise
    except httpx.TimeoutException:
        raise ProductImageError("Downloading the product image timed out.", retryable=True) from None
    except httpx.HTTPError:
        raise ProductImageError("The product image could not be downloaded.", retryable=True) from None
    data = bytes(body)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            fmt, (width, height) = image.format or "", image.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise ProductImageError("The product image is not a valid image.", retryable=False) from None
    if fmt not in SUPPORTED_FORMATS:
        raise ProductImageError(f"Unsupported product image format ({fmt or 'unknown'}).", retryable=False)
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ProductImageError(f"The product image is too small ({width}x{height}).", retryable=False)
    if width * height > MAX_IMAGE_PIXELS:
        raise ProductImageError("The product image is too large.", retryable=False)
    return ProductImage(url, data, hashlib.sha256(data).hexdigest(), width, height, fmt,
                        round(time.perf_counter() - started, 3))
