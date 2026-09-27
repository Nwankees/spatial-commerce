from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)
MAX_PAGE_BYTES = 4_000_000


class PageFetchError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class FetchedPage:
    url: str
    html: str


class PageFetcher(Protocol):
    async def fetch(self, url: str) -> FetchedPage: ...


def is_public_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host or "." not in host or host.endswith(".local"):
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        ipaddress.ip_address(host)
        return False  # Literal IPs are never retailer product pages.
    except ValueError:
        return True


class HttpPageFetcher:
    """Plain HTTP GET of a retailer product page. No JavaScript execution."""

    def __init__(self, timeout_seconds: float = 8.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._timeout = timeout_seconds
        self._transport = transport

    async def fetch(self, url: str) -> FetchedPage:
        if not is_public_http_url(url):
            raise PageFetchError("Retailer URL is not a public web page.", retryable=False)
        headers = {
            "User-Agent": BROWSER_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=True, max_redirects=5, transport=self._transport
            ) as client:
                async with client.stream("GET", url, headers=headers) as response:
                    if not is_public_http_url(str(response.url)):
                        raise PageFetchError("Retailer redirected to a non-public address.", retryable=False)
                    if response.status_code in (401, 403, 429) or response.status_code == 451:
                        raise PageFetchError(
                            f"Retailer blocked automated access (HTTP {response.status_code}).",
                            retryable=response.status_code == 429,
                        )
                    if response.status_code >= 500:
                        raise PageFetchError(f"Retailer returned HTTP {response.status_code}.", retryable=True)
                    if response.status_code != 200:
                        raise PageFetchError(f"Retailer returned HTTP {response.status_code}.", retryable=False)
                    content_type = response.headers.get("content-type", "").lower()
                    if "html" not in content_type:
                        raise PageFetchError("Retailer did not return an HTML page.", retryable=False)
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > MAX_PAGE_BYTES:
                            break
                    encoding = response.encoding or "utf-8"
                    return FetchedPage(str(response.url), bytes(body).decode(encoding, errors="replace"))
        except PageFetchError:
            raise
        except httpx.TimeoutException:
            raise PageFetchError("Retailer page timed out.", retryable=True) from None
        except httpx.HTTPError as exc:
            logger.info("Retailer fetch failed: %s", type(exc).__name__)
            raise PageFetchError("Retailer page could not be reached.", retryable=True) from None
