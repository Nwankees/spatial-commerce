# Visual analysis backend

This local FastAPI service keeps the Gemini credential off the Android device and returns a validated `VisualProductAnalysis` JSON object.

## Local setup

From `backend/`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# Put the SerpApi key in .env (never commit this file). Vision runs locally in Ollama:
#   ollama pull qwen3-vl:30b   (OLLAMA_MODEL / OLLAMA_BASE_URL are configurable)
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`. For the USB-connected Android device, run:

```powershell
adb reverse tcp:8000 tcp:8000
```

The debug Android app then reaches the service at `http://127.0.0.1:8000` through the USB connection.

## Local vision (Milestone 5.5)

`POST /api/v1/analyze` calls Ollama's `/api/chat` with `stream: false`, `think: false`, and the `VisualProductAnalysis` JSON schema as `format`. `/health` reports `vision.reachable` and `vision.modelInstalled`. Gemini code remains in `app/gemini_service.py` but is unused.

## Dimension extraction model

Retailer-page dimension extraction uses a second, text-only Ollama model: `ollama pull qwen3:4b-instruct` (`OLLAMA_DIMENSION_MODEL`). It only runs when structured sources lack width and depth; its output is validated deterministically against the page text before use (see the main README). Test a page with `.venv\Scripts\python ..\tmp\m5_5_dimension_check.py <retailer URL>`.

## Product search

`POST /api/v1/products/search` accepts `{"analysis": <VisualProductAnalysis>, "maxResults": 5}` and returns the query used, the provider (`serpapi`), `resultSource` (`live` or `cache`), `cachedAt`, and normalized `ProductCandidate` results from SerpApi Google Shopping. It requires `SERPAPI_API_KEY`; `/health` reports `productSearchConfigured`. Successful live results are cached in `.cache/product_search_cache.json` and are only served, clearly labeled as cached, when a later live search for the same query fails.

## Product dimensions

`POST /api/v1/products/dimensions` accepts `{"productId", "productUrl"}` for a product previously returned by product search and returns `ResolvedDimensions` (meters, `verified`/`partial`/`unavailable`, source type, source URL/name, raw text, `retryable`). It uses SerpApi's Google Immersive Product API (same `SERPAPI_API_KEY`) plus plain HTTP fetches of up to two retailer pages (`RETAILER_FETCH_TIMEOUT_SECONDS`, default 8). Dimensions are never estimated; see the main README for the trust rules.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

The API documentation is available at `http://127.0.0.1:8000/docs` while the service is running.
