"""Isolated rendered-page helper for the M6 dimension resolver.

Run in its own virtual environment because Crawl4AI, Browser Use, and Playwright
have a much larger dependency surface than the main FastAPI backend.
"""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Spatial Commerce dimension helper", version="0.1.0")


class CrawlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(pattern=r"^https?://", max_length=3000)


class BrowseRequest(CrawlRequest):
    title: str = Field(max_length=600)
    retailer: str | None = Field(default=None, max_length=200)
    identifiers: dict[str, str] = Field(default_factory=dict)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "crawl4ai": _available("crawl4ai"),
        "browserUse": _available("browser_use"),
        "browserModel": os.getenv("BROWSER_USE_MODEL", "qwen3-vl:8b"),
    }


@app.post("/crawl")
async def crawl(request: CrawlRequest) -> dict[str, str]:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import PruningContentFilter
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    except ImportError as exc:
        raise HTTPException(503, "Crawl4AI is not installed in the helper environment") from exc

    browser = BrowserConfig(
        headless=True,
        verbose=False,
        enable_stealth=True,
        user_agent_mode="random",
        viewport_width=1440,
        viewport_height=1000,
    )
    markdown = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.45, threshold_type="fixed")
    )
    config = CrawlerRunConfig(
        markdown_generator=markdown,
        excluded_tags=["nav", "footer", "aside", "form"],
        exclude_external_links=True,
        remove_overlay_elements=True,
        remove_consent_popups=True,
        simulate_user=True,
        override_navigator=True,
        magic=True,
        wait_until="domcontentloaded",
        page_timeout=45_000,
    )
    try:
        async with AsyncWebCrawler(config=browser) as crawler:
            result = await crawler.arun(url=request.url, config=config)
    except Exception as exc:
        raise HTTPException(502, f"Rendered page failed: {type(exc).__name__}") from exc
    if not getattr(result, "success", False):
        raise HTTPException(502, str(getattr(result, "error_message", "Rendered page failed"))[:300])
    rendered = getattr(result, "markdown", None)
    content = (
        getattr(rendered, "fit_markdown", None)
        or getattr(rendered, "raw_markdown", None)
        or str(rendered or "")
    )
    if not content.strip():
        raise HTTPException(404, "Rendered page contained no useful text")
    final_url = getattr(result, "url", None) or request.url
    return {"url": final_url, "content": content[:80_000]}


@app.post("/browse")
async def browse(request: BrowseRequest) -> dict[str, str]:
    try:
        from browser_use import Agent, ChatOllama
    except ImportError as exc:
        raise HTTPException(503, "Browser Use is not installed in the helper environment") from exc

    identity = ", ".join(f"{key}={value}" for key, value in request.identifiers.items()) or "none"
    browser_url = quote(request.url, safe=":/?&=,%|+")
    task = f"""Open {browser_url} and find the overall physical dimensions of this exact product:
Title: {request.title}
Retailer: {request.retailer or 'unknown'}
Identifiers: {identity}

Scroll and, when necessary, expand specification/details accordions or click View full specifications.
Return only copied page evidence containing overall Width, Depth or Length, and Height when available,
plus the page URL. Never use package/shipping dimensions or seat/arm/back/component measurements.
If the exact product has no overall dimensions, say unavailable. Do not estimate."""
    try:
        agent = Agent(
            task=task,
            llm=ChatOllama(
                model=os.getenv("BROWSER_USE_MODEL", "qwen3-vl:8b"),
                host=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                ollama_options={
                    # Browser Use sends the current DOM plus a screenshot.  Ollama's
                    # 4k default truncates ordinary retailer pages before the model
                    # can act, so reserve enough context for the last-resort browser
                    # fallback while keeping generation deterministic.
                    "num_ctx": int(os.getenv("BROWSER_USE_NUM_CTX", "16384")),
                    "temperature": 0,
                },
            ),
            use_thinking=False,
            use_judge=False,
            enable_planning=False,
            max_actions_per_step=2,
            max_clickable_elements_length=18_000,
            vision_detail_level="low",
            llm_timeout=90,
            step_timeout=120,
        )
        history = await agent.run(max_steps=int(os.getenv("BROWSER_USE_MAX_STEPS", "12")))
        content = history.final_result() or ""
        urls = history.urls() if hasattr(history, "urls") else []
    except Exception as exc:
        raise HTTPException(502, f"Interactive browser failed: {type(exc).__name__}") from exc
    if not content.strip():
        raise HTTPException(404, "Interactive browser found no dimension evidence")
    final_url = next((url for url in reversed(urls or []) if isinstance(url, str) and url.startswith("http")), request.url)
    return {"url": final_url, "content": content[:20_000]}


def _available(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False
