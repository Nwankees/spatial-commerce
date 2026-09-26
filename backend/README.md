# Visual analysis backend

This local FastAPI service keeps the Gemini credential off the Android device and returns a validated `VisualProductAnalysis` JSON object.

## Local setup

From `backend/`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# Put the HackGT Gemini key and the SerpApi key in .env (never commit this file).
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health`. For the USB-connected Android device, run:

```powershell
adb reverse tcp:8000 tcp:8000
```

The debug Android app then reaches the service at `http://127.0.0.1:8000` through the USB connection.

## Product search

`POST /api/v1/products/search` accepts `{"analysis": <VisualProductAnalysis>, "maxResults": 5}` and returns the query used, the provider (`serpapi`), `resultSource` (`live` or `cache`), `cachedAt`, and normalized `ProductCandidate` results from SerpApi Google Shopping. It requires `SERPAPI_API_KEY`; `/health` reports `productSearchConfigured`. Successful live results are cached in `.cache/product_search_cache.json` and are only served, clearly labeled as cached, when a later live search for the same query fails.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

The API documentation is available at `http://127.0.0.1:8000/docs` while the service is running.
