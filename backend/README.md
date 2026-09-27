# Visual analysis backend

This local FastAPI service keeps the Gemini credential off the Android device and returns a validated `VisualProductAnalysis` JSON object.

## Local setup

From `backend/`:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# Put the SerpApi key in .env (never commit this file). Vision runs locally in Ollama:
#   ollama pull qwen3-vl:8b   (OLLAMA_MODEL / OLLAMA_BASE_URL are configurable)
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

Retailer-page dimension extraction uses a second, text-only Ollama model: `ollama pull qwen3:4b-instruct` (`OLLAMA_DIMENSION_MODEL`). A generic parser preserves product JSON-LD, embedded product/application data, spec tables, name/value lists, headings and visible product text while removing obvious page noise. Explicit separate Width/Depth/Height fields use a deterministic fast path; otherwise Qwen selects the overall-product evidence from that structured representation. Code then verifies every cited entry, value, unit and variant scope before normalizing to meters. Unlabeled triples remain verified in source order instead of being discarded.

For a verbose live trace from `backend/`, run:

```powershell
.venv\Scripts\python scripts\dimension_debug.py --url "https://retailer.example/product/..." --title "Selected product" --retailer "Retailer"
```

Add `--html ..\tmp\saved-page.html` to use a captured fixture without refetching. The script prints the selected variant, exact model input, raw result, source path, deterministic validation, normalized values, axis mapping, and fit/M6 eligibility.

## Conversational shopping agent (Milestone 7)

M7 reuses local `qwen3:4b-instruct` (`OLLAMA_AGENT_MODEL`) as a schema-constrained action planner. The normal path is: user message + session state → local Qwen → validated typed action → backend tool execution → a second schema-constrained Qwen call grounded strictly in the typed tool result. The model cannot directly mutate the session or change a tool's status/facts. The small deterministic router and canonical tool-result wording are used only if Ollama is unavailable or returns malformed JSON, so the core demo remains recoverable without pretending a tool succeeded.

Conversation endpoints:

- `POST /api/v1/conversations` creates an in-memory session.
- `PUT /api/v1/conversations/{id}/context` synchronizes the existing button UI's current analysis, analyzed frame, ordered results, selected product, and measured space. Prepared image bytes remain backend-only and never enter Qwen context.
- `POST /api/v1/conversations/{id}/messages` plans and executes one typed shopping action.
- `GET /api/v1/conversations/{id}` reads state/history.
- `POST /api/v1/conversations/{id}/reset` clears that session.

Implemented tools cover visual + text product search/refinement/selection, details/comparison, two-axis fit checks, and M6's real selected-product reconstruction service. Conversational fit and AR share M6's dimension cache; the AR action starts or joins the real SF3D job and Android continues through the normal real-scale placement flow. It never substitutes the built-in chair.

## Visa-aligned agentic commerce (Milestone 8)

M8 extends the same Qwen planner with typed `prepare_purchase`, `confirm_purchase`,
`cancel_purchase`, `get_purchase_status`, and `open_checkout` actions. Products, prices,
merchants, variants, and URLs are resolved from backend session/search state. Preparing a purchase
only creates a review; a separate confirmation tied to the active intent ID and nonce is required.
Changing the result set or selection cancels the pending review, so a stale “yes” cannot authorize it.

The backend signs the confirmed intent with an ephemeral Ed25519 development key using an
RFC 9421-style signature over authority, path, and content digest. The controlled merchant verifier
checks timestamps, the eight-minute lifetime, key ID, signature, payload integrity, and nonce replay.
Completion is deliberately labeled `COMPLETED_SIMULATED`, `isSimulation=true`, and
`visaCertified=false`; no Visa Intelligent Commerce credential, PAN, CVV, or real payment is used.
The exact merchant URL is preserved for an explicit handoff. No additional M8 API key is required.

## Product search

`POST /api/v1/products/search` accepts the validated `analysis`, the same `imageBase64`/rotation used for analysis, and `maxResults`. Text queries run through SerpApi Google Shopping while the photographed object is uploaded directly to SerpApi's Image API and searched through Google Lens `products`, `visual_matches`, and `exact_matches`; the image never needs a public URL. Results are deduplicated and reranked together, with bounded one-candidate-at-a-time local Qwen3-VL comparisons over the top-three shortlist. Either retrieval path can fail independently. `SERPAPI_API_KEY` stays server-side; `LENS_SEARCH_ENABLED`, `LENS_MODES`, `VISUAL_RERANK_ENABLED`, and `VISUAL_RERANK_MAX_CANDIDATES` tune the feature. Successful live results are cached in `.cache/product_search_cache.json` for dimension/AR handoff and failure fallback.

## Product dimensions

`POST /api/v1/products/dimensions` accepts `{"productId", "productUrl"}` for a product previously returned by product search and returns `ResolvedDimensions` (source-order meters, axis mapping, `verified`/`partial`/`unavailable`, source path/provenance, variant scope, and retry state). It uses SerpApi's Google Immersive Product API (same `SERPAPI_API_KEY`) plus plain HTTP fetches of up to two retailer pages (`RETAILER_FETCH_TIMEOUT_SECONDS`, default 8). Dimensions are never estimated; see the main README for the trust rules.

## AR preview (Milestone 6)

`POST /api/v1/products/ar-preview`, `GET /api/v1/ar-assets/{id}` and
`GET /api/v1/ar-assets/{id}/model.glb` turn a selected product's image into a normalized GLB via the
SF3D sidecar (`RECONSTRUCTION_SERVICE_URL`, default `http://127.0.0.1:8010`; see
`../reconstruction_service/README.md`). Assets are cached in `AR_ASSET_CACHE_DIR`
(default `.cache/ar-assets`, git-ignored). The cache identity includes the selected product/URL,
image hash, verified-dimension identity, reconstruction provider/version, and normalization version.
Dimensions already resolved by Check Fit are reused rather than sending Browser Use or another slow
fallback through the preview path. Responses expose compact dimension, download, cache, SF3D,
normalization, write, and total timing fields plus a `hit`/`miss`/`joined` cache outcome. Scaling uses
only verified retailer measurements. If the retailer supplies an unlabeled triple, the generated mesh
may assign its W/D/H permutation from shape proportions, but it never supplies or changes the physical
meter values; ambiguous mappings are rejected.

## Tests

```powershell
.venv\Scripts\python -m pytest
```

The API documentation is available at `http://127.0.0.1:8000/docs` while the service is running.
