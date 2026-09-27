# Spatial Commerce

HackGT 13 project exploring spatially aware conversational commerce, with Android/ARCore measurement tools, a physically verified anchored product preview, and on-demand Gemini visual product understanding.

## Milestone 1: Android/ARCore foundation

The prototype provides:

- an ARCore camera session on Android;
- an explicit ARCore Depth API capability check and `AUTOMATIC` depth mode;
- plane, feature-point, and depth-point hit testing;
- two anchored measurement markers;
- 3D Euclidean distance calculated from the anchors' world coordinates;
- a clear centimeter/meter readout;
- reset and re-measure behavior; and
- placement of a visible, anchored 10 cm test cube.

The rendering layer uses ARCore with a small OpenGL ES 2 renderer. There are no backend, AI, commerce, authentication, database, or voice integrations in this milestone.

## Milestone 2: true-scale product preview

The Milestone 2 branch adds a separate **Preview product** mode with a locally authored furniture model:

- `ProductPreview` holds the product ID, name, model reference, and authoritative width/depth/height;
- the Lighthouse Lounge Chair is defined as 0.68 m wide × 0.74 m deep × 0.84 m high;
- the chair is assembled from meter-scaled OpenGL primitives, so its geometry reaches those declared outer dimensions without an arbitrary display scale;
- product placement accepts tracked upward-facing horizontal planes so the chair stays upright and rests on its legs;
- tapping another valid surface replaces its ARCore anchor and repositions it;
- ±15° controls rotate the chair around its vertical axis; and
- the overlay shows the product name and metric/imperial dimensions.

The model reference is `primitive://lighthouse-lounge-chair-v1`. It is original project geometry built from code, so there is no third-party model or attribution requirement.

## Milestone 3: Gemini visual product understanding

The Android app adds an intentional **Analyze object** action without opening a second camera session. It acquires one CPU camera image from the current ARCore frame, encodes it as JPEG, and sends it to the local backend. The backend:

- keeps `GEMINI_API_KEY` outside the APK and Git;
- validates and normalizes the image before sending it to Gemini;
- uses Gemini's schema-constrained JSON output;
- validates the result again as a typed `VisualProductAnalysis`; and
- returns category, subcategory, color, materials, style, shape, search keywords, confidence, and a safe no-object result.

The Android client displays progress, structured results, and retryable errors while leaving measurement and product-preview modes intact. It uses `http://127.0.0.1:8000` for USB development; cleartext networking is enabled only for this local Milestone 3 workflow and must be replaced by HTTPS before deployment.

## Milestone 4: real product retrieval

After a successful analysis, a separate **Find similar products** action sends the typed `VisualProductAnalysis` to the backend (`POST /api/v1/products/search`). The backend:

- builds one deterministic shopping query (`ProductQueryBuilder`: color + primary style + primary material + product type, capped for recall, falling back to Gemini's `searchKeywords`) without a second Gemini call;
- queries SerpApi Google Shopping through a replaceable `ProductSearchProvider` (`SerpApiProductSearchProvider`); `SERPAPI_API_KEY` stays on the backend;
- normalizes up to five real results into `ProductCandidate` (numeric price, raw price text, explicit currency only when unambiguous, retailer, provider, provider product ID, image URL, product URL, rating, review count), dropping entries without a title, numeric price, or product link and removing duplicates;
- reports dimensions as `unavailable` because Google Shopping search results do not contain structured dimensions; they are never estimated (`status` is `complete`, `partial`, or `unavailable` for Milestone 5); and
- caches successful live results by normalized query in `backend/.cache/` (git-ignored). If the provider later fails for the same query, the cached real results are returned with `resultSource: "cache"` and the original retrieval time; otherwise a retryable error is returned.

The Android app shows the results in a scrollable panel above the lower overlay, with thumbnails (loaded with Coil), title, price, retailer, and rating. Tapping a result selects it and keeps it in app state for later milestones. The AR preview still renders only the locally authored Lighthouse chair; retrieved products are not rendered in AR.

## Milestone 5.5: local vision + multi-query retrieval

Gemini is no longer on the active path. `/api/v1/analyze` now calls a **local Ollama vision model** (default `qwen3-vl:30b`), and product search runs **several searches per object** and merges them.

### Local vision architecture

Android → FastAPI `/api/v1/analyze` → `OllamaVisionService` → Ollama `POST /api/chat` on the same PC (`OLLAMA_BASE_URL`, default `http://127.0.0.1:11434`). The request uses `stream: false`, `think: false`, `temperature: 0.1`, `keep_alive: 15m`, the JPEG as a base64 image, and `format` set to the analysis JSON schema so Ollama constrains the output. The backend then validates the JSON against `VisualProductAnalysis` and applies deterministic guards:

- a detection below `VISION_MIN_CONFIDENCE` (0.2) becomes a safe "no product" result;
- a detection with no product type and no queries is rejected as incomplete (retryable 502);
- `visibleSpecifications` are kept only if their value appears in the legible `visibleText`;
- a brand/model hypothesis that claims `visible_text` evidence but is not in `visibleText` is downgraded to `design_resemblance` (its confidence is kept).

Ollama unreachable, model not installed, timeouts, malformed JSON and schema mismatches all become retryable 502/504 errors. `/health` reports whether Ollama is reachable and whether the configured model is installed. Each analysis logs model, outcome, duration and Ollama's own timings; the response carries `X-Vision-Model` and `X-Analysis-Duration-Ms` headers. There is no automatic fallback to Gemini.

Required setup: install Ollama, `ollama pull qwen3-vl:30b` (or set `OLLAMA_MODEL=qwen3-vl:8b` for lower latency; no code change), and keep Ollama running. Environment variables: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS` (180), `OLLAMA_KEEP_ALIVE`, `VISION_MIN_CONFIDENCE`, `PRODUCT_SEARCH_MAX_QUERIES` (3), `PRODUCT_SEARCH_RESULTS_PER_QUERY` (10).

### Richer analysis contract

`VisualProductAnalysis` now carries `category`, `subcategory`, `brand` and `modelFamily` as optional hypotheses (`value`, `confidence`, `evidence` = `visible_text` | `logo` | `design_resemblance`), `visibleText`, `color`, `materials`, `style`, `shape`, `distinctiveFeatures`, text-backed `visibleSpecifications`, 2–6 `searchQueries`, `confidence`, `uncertaintyNotes`, and `message`. Unknown fields stay null or empty. The prompt tells the model its output feeds a shopping search, asks it to keep distinguishing details and printed text, and forbids guessed dimensions, wattage, connector standards, model numbers or brands. **Exact technical specs are never inferred from images.** The old `searchKeywords` name is still accepted as input.

### Multi-query retrieval and reranking

`QueryPlanner` takes the model's queries (most specific first) plus the deterministic Milestone 4 query as a fallback, removes near-duplicates, and caps the list (`PRODUCT_SEARCH_MAX_QUERIES`, default 3). The searches run concurrently through the existing `ProductSearchProvider` (SerpApi). Results are merged and deduplicated by provider product ID, product URL, and normalized title + retailer, keeping each product's best-ranked occurrence and recording every query that returned it (`matchedQueries`).

`ProductReranker` then scores each candidate with explicit, testable terms: provider rank, extra queries that also found it, the most specific query, brand match (× brand confidence, only at ≥ 0.5), product-type term overlap, model-family/visible-text overlap, and color/material/feature overlap. The top five by score are returned with `retrievalScore`. If some searches fail, the others' results are returned with a note. If all fail, cached results are returned (the same query set first, otherwise individually cached queries), clearly labeled as cached; otherwise a retryable error. **Image-similarity reranking is not implemented yet.**

Milestone 5 compatibility: merged products keep their SerpApi product ID, detail token (server-side), links, retailer, price and image, and are recorded in the product cache, so **Check fit** works on them unchanged.

### Known limitations

- `qwen3-vl:30b` may not fit fully in GPU memory; first analysis after a model load can take minutes. The phone waits up to 190 s.
- Reranking is term-based: listings with sparse or generic titles can rank lower than they deserve.
- Up to 3 SerpApi searches per **Find similar products** (plus one detail call per **Check fit**).

### Manual test procedure

1. Start Ollama and confirm `ollama list` shows the configured model; start the backend; `adb reverse tcp:8000 tcp:8000`.
2. Optional: `.venv\Scripts\python ..\tmp\m5_5_integration.py photo.jpg` from `backend\` to see analysis, queries and ranked products.
3. On the phone: Analyze a branded product (e.g. a laptop charger or headphones) and a piece of furniture; confirm type, likely brand with confidence (only when visible), features and searches.
4. Find similar products: confirm the panel lists the searches run and shows real, relevant products.
5. Select a product and Check fit; confirm Milestone 5 still works.

## Milestone 5: spatial fit

Selected real product → trustworthy product dimensions → measured available space → deterministic fit result.

### Architecture

- `POST /api/v1/products/dimensions` takes only `{"productId", "productUrl"}` of a product the backend itself returned from `/products/search`. The backend looks the product up in its own record (the product-search cache, which also stores the provider's server-side detail reference); unknown or mismatched products get a 404, and client-supplied dimensions or URLs are never trusted.
- `DimensionResolver` gathers evidence from replaceable sources:
  - `ProductDetailSource` → `SerpApiImmersiveProductDetailSource` (SerpApi Google Immersive Product API, keyed by the search result's `immersive_product_page_token`): explicit spec fields (`about_the_product.features`) and the list of stores with direct retailer links.
  - `PageFetcher` → `HttpPageFetcher`: one plain HTTP GET per retailer page (no JavaScript), at most two pages, fetched concurrently, the result's own retailer first. Only public `http(s)` hosts are fetched (no IP literals, localhost, or Google URLs).
  - `dimension_extraction` parses each page into per-source findings; `dimension_parsing` holds the pure, conservative parsing and unit rules.
- Android adds **Measure space** mode, a **Check fit** action on the selected product, a `FitPanel`, and the pure `FitEngine`.

### Dimension resolution update: retailer text extraction

Physical testing found retailer pages (e.g. Target) that state dimensions as `Dimensions (Overall): 39.4 inches (H) x 21.65 inches (W) x 22 inches (D)`. The deterministic parser did not recognize parenthesized `(H)`/`(W)`/`(D)` labels or the `Dimensions (Overall)` label, and on such pages the text is often only inside embedded page data (scripts), which it never read. Instead of adding more retailer-specific regexes, the resolver now has an extraction step backed by a small local text model plus deterministic validation.

Hierarchy (highest first; sources are never mixed):

1. Provider product-detail structured specs (SerpApi Immersive Product `about_the_product` fields). If they give width and depth, no retailer page is fetched.
2. Retailer JSON-LD.
3. Retailer structured metadata (microdata) and specification tables.
4. Cleaned retailer spec/page text → local LLM extraction (`OLLAMA_DIMENSION_MODEL`, default `qwen3:4b-instruct`) → deterministic validation. Only runs when 1–3 lack width and depth. Source type `page_text_llm`, shown in the app as "retailer specifications".
5. Deterministic labeled page-text parsing (legacy fallback).
6. Unavailable (with `retryable` when a timeout/network/model error was involved).

Page preprocessing (`dimension_evidence.py`, generic, not retailer-specific): product JSON-LD (name, width/depth/height/size/additionalProperty, dimension sentences from the description), spec-table rows and `<dl>` pairs with dimension keywords, visible text lines with dimension keywords plus the next line, and keyword windows from embedded page scripts (JS string escapes and tags removed). Navigation, header/footer, forms, cookie/consent banners and review/Q&A sections are skipped; duplicates are removed; the result is capped at `DIMENSION_EVIDENCE_MAX_CHARS` (6000).

**The model is an extractor, not a source of truth.** The prompt forbids using titles, product type, images, typical sizes or world knowledge; requires explicit W/D/H labels or an explicitly declared order; rejects unlabeled triples and `L x W x H`; excludes package and part dimensions; and asks for numbers copied verbatim with their unit plus the verbatim evidence text. The request is text only with `stream: false`, `think: false`, `temperature: 0`, and a JSON schema `format`.

Deterministic post-validation (`validate_extraction`), per claimed axis:

- the evidence text must be verbatim source text (otherwise the whole source is searched instead of trusting it);
- the exact number must occur in the source, each occurrence used for at most one axis;
- the unit right after that number (or the group's trailing unit) must equal the claimed unit (mm, cm, m, in/″/inches, ft/feet);
- the number's own segment must carry exactly the claimed axis label (`W`, `(W)`, `Width`, `wide`, …) and no other, or the source must declare the order (`(W x D x H)`, never with `L`) and the number's position must match;
- package/shipping/box/seat/arm/leg/shelf/drawer/interior contexts are rejected, as are non-positive and implausible (outside 0.01–10 m) values;
- meters are computed in code. The model's own `status` is ignored except that `unavailable` stays unavailable.
- if the same labeled statement appears elsewhere in the source with a different value for that axis (e.g. several product variants each listing `Dimensions (Overall)` with different widths), that axis is dropped rather than picking one;
- if the first pass quoted a real statement but some axes failed validation, one second pass is run on just that quoted statement; the pass with more validated axes is kept (passes are never merged).

Setup: `ollama pull qwen3:4b-instruct`. Env: `OLLAMA_DIMENSION_MODEL`, `OLLAMA_DIMENSION_TIMEOUT_SECONDS` (60), `DIMENSION_LLM_ENABLED`, `DIMENSION_EVIDENCE_MAX_CHARS`. `/health` reports `dimensionModel.installed`. Only resolved (verified/partial) results are cached in memory, so installing the model or a transient failure is re-checked on the next **Check fit**.

Known retailer limitations: no browser automation, so pages that block plain requests (HTTP 403, e.g. west elm/Staples) or render specs only with client-side JavaScript that is not in the server HTML still end up unavailable; `L x W x H` listings (common on Walmart) remain deliberately unresolved.

### Selected-variant matching

SerpApi's product-detail specs describe Google's product *family*; a probe of seven retailers showed they can describe a different variant than the offer the user picked (e.g. "Color: Black" while the selected Target/Walmart offers were Gray). The resolver therefore builds a `VariantContext` for the selected result (`variant_context.py`) from real identifiers only:

- Google IDs parsed from the result's `product_link` (`catalogid`, `productid`, `headlineOfferDocid`, `gpcid`, `mid`);
- the concrete offer: the product-detail `stores[]` entry with the selected retailer name **and** the selected price (ambiguous → no offer);
- retailer IDs read from that offer's URL by a small rule table (Target `A-<item>`, Walmart `/ip/<item>` + `selectedOfferId`, Wayfair `~<sku>` + `PiID`, Home Depot, Kohl's `skuid`, Staples, Best Buy `/sku/`, Office Depot, Quill, Lowe's, Costco, Amazon) plus generic query parameters (`variant`, `sku`, `skuid`, `itemId`, `pid`, `preselect`, `offerId`, …);
- explicitly selected options: SerpApi `variants[].items[].selected == true`, or an explicit `Color: X`/`Size: Y` in the offer title.

Identity is `exact_item` (retailer item/SKU/variant ID), `exact_offer` (offer ID), `options_only`, or `product_only`.

Policy:

1. **Exact identity**: the selected offer's page is fetched and its embedded data (JSON in `<script>` tags) is searched for the object whose identifier key (`tcin`, `usItemId`, `sku`, `variantId`, `id`, …) *exactly* equals the retailer ID (`variant_scope.py`). Only that object's content is used; nested objects with a different ID (sibling variants), AI-generated summaries, reviews, media and package/shipping data are skipped. Labeled name/value specs there are parsed deterministically; otherwise the scoped text goes to the local extraction model and deterministic validation. If no scoped record has dimensions, the exact item's page is used (sibling conflicts still drop axes).
2. **Options only**: options scope a record only if they match exactly one item ID in the retailer's own data; zero or several matches leave the variant unresolved.
3. Exact-variant results outrank SerpApi family specs; a disagreement between them is not treated as a conflict. Without exact identity (or when the exact page has no width+depth), SerpApi family specs remain first, as before.

`ResolvedDimensions` now includes `variantScope` (`exact_variant`, `exact_variant_page`, `product_family`, `retailer_page`) and `variantIdentity`. Search deduplication no longer merges two results with the same title and retailer when their Google offer IDs (`headlineOfferDocid`) differ, so variants are not merged. AI-generated retailer summaries (e.g. `genAi…Summary`) are excluded from all dimension evidence.

### Dimension trust rules

- Values are only taken from explicit source data, in the priority order listed above (updated: provider product-detail specs now rank first, and LLM extraction with deterministic validation sits before the legacy page-text parser).
- A source stating all three axes wins by priority; otherwise the highest-priority source stating both width and depth; otherwise any partial source. Values from different sources are never mixed.
- Never used: images, category averages, Gemini, product titles, untyped JSON embedded in page scripts (it often describes related products).
- A number needs an explicit unit (mm, cm, m, in/″, ft, UN/CEFACT `MMT`/`CMT`/`MTR`/`INH`/`FOT`) or a unit declared by its label, e.g. `Width (in)`. Bare numbers are rejected.
- Axes are mapped only from explicit labels: `Width`/`Overall Width`/…, `W`/`D`/`H` letters on each value, or an order declared in the label such as `Dimensions (W x D x H)`. Unlabeled `30 x 28 x 35 in`, any `L x W x H` order, a fourth value, fragments like the `6 in` in `2 ft 6 in`, and package/shipping/box/seat/arm/leg/interior/adjustable contexts are rejected. Two different values for the same axis in one source drop that axis.
- Values outside 0.01–10 m are rejected as misparses.

### Dimension provenance model

`ResolvedDimensions`: `widthMeters`, `depthMeters`, `heightMeters` (null when not explicitly stated), `status` (`verified` = all three stated, `partial` = some stated, `unavailable`), `sourceType` (`json_ld`, `structured_metadata`, `spec_table`, `page_text`, `unavailable`), `sourceUrl`, `sourceName`, `rawDimensions` (the text the values were parsed from), `retryable`, and `message`. Timeouts, network errors, provider failures, and HTTP 429/5xx are retryable; blocked pages (401/403), non-HTML, and pages without dimensions are definitive `unavailable`. Definitive results are cached in memory for an hour to spare provider calls.

### Fit formula

```
fits ⇔ productWidth + clearance ≤ availableWidth  AND  productDepth + clearance ≤ availableDepth
widthRemaining = availableWidth − productWidth − clearance   (not clamped; negative = over)
depthRemaining = availableDepth − productDepth − clearance
```

Milestone 5 uses `clearance = 0` (exact footprint) and compares the product in its stated orientation. Missing product width or depth → `UNKNOWN`; missing available width or depth → `NEEDS_MEASUREMENT`. No LLM participates.

### Manual test procedure

1. Start the backend and `adb reverse tcp:8000 tcp:8000`; launch the app.
2. Analyze a real object, tap **Find similar products**, select a result, tap **Check fit**.
3. Confirm the panel shows either product dimensions with their source, or **UNKNOWN — dimensions unavailable** with a reason (never invented values).
4. Tap **Measure space**: tap the left then right edge of the available width, then the front then back edge of the available depth. Confirm `Available width` / `Available depth` readouts.
5. Confirm **FITS** / **DOES NOT FIT** with per-axis spare or excess, then **Reset** and confirm the measurements clear and the result returns to **NEEDS MEASUREMENT**.

### Milestone 5 and 5.5 physical verification

Milestones 5 and 5.5 were physically verified end to end on September 27, 2026 using the Samsung Galaxy S25:

- local Ollama vision analysis (`qwen3-vl:30b`) worked without Gemini: pass;
- multi-query SerpApi retrieval returned merged, ranked real products: pass;
- exact selected-variant dimension resolution worked; the selected Target Gray chair (item 93144009) resolved to 21.65 in W × 22 in D × 39.4 in H from its own variant record: pass;
- Measure space recorded available width and depth: pass;
- FITS, DOES NOT FIT and UNKNOWN results behaved as specified: pass;
- Reset cleared the space measurements and the fit result returned to NEEDS MEASUREMENT: pass; and
- Milestone 1–4 Measure, Preview product, Analyze object and Find similar products still worked: pass.

### Milestone 4 physical verification

Milestone 4 was physically verified on September 26, 2026 using the Samsung Galaxy S25:

- backend health reported Gemini and product search configured, and `adb reverse` was active: pass;
- Milestone 1 measurement and reset, and Milestone 2 placement, rotation, repositioning, and reset still worked: pass;
- Gemini analysis of a real chair returned sensible attributes and enabled **Find similar products**: pass;
- several real purchasable products appeared, labeled live, with sensible titles, prices, and retailers: pass;
- product thumbnails loaded for the results: pass;
- tapping a product visibly selected it, and selection survived hiding and reopening the panel: pass;
- a significantly different product category returned matching results: pass;
- with the backend stopped, search failed safely with a retry that succeeded after restart: pass;
- with live search forced to fail, a previously searched query returned clearly labeled cached results: pass;
- the no-product path kept the search action hidden: pass; and
- previous AR functionality still worked afterward: pass.

### Deferred issues (not addressed in Milestone 4)

- Cleartext HTTP is allowed app-wide for the local `adb reverse` backend; scope it or move to HTTPS before deployment.
- The manifest requires the Depth feature although the code falls back without it.
- `ArRenderer.setMode` reads measurement anchors from the UI thread.
- The backend creates a Gemini client per request.
- There are no Android unit tests.

### Milestone 3 physical verification

Milestone 3 was physically verified on September 26, 2026 using the Samsung Galaxy S25:

- intentional one-shot analysis of several real objects completed successfully;
- Gemini returned reasonable structured product attributes and search keywords: pass;
- the no-obvious-product path completed safely and allowed retrying: pass;
- the AR camera remained active and the application did not crash: pass;
- Milestone 1 measurement and reset still worked: pass; and
- Milestone 2 placement, rotation, repositioning, and reset still worked: pass.

### Milestone 2 physical verification

Milestone 2 was physically verified on September 26, 2026 using the Samsung Galaxy S25:

- the chair was recognizable, upright, and resting on the detected floor: pass;
- the 0.68 × 0.74 × 0.84 m rendered dimensions closely matched real-world measurements: pass;
- the product remained anchored while the phone moved around it: pass;
- the chair maintained a convincing 3D presence as people moved behind it: pass;
- both ±15° rotation controls worked: pass;
- tapping a new floor location moved the chair and removed the old instance: pass;
- reset removed the product: pass; and
- Milestone 1 measurement, markers, distance display, and reset still worked: pass.

## Verified device state

Milestone 1 was physically verified on September 26, 2026 using a Samsung Galaxy S25 (`SM-S931U1`) running Android 16:

- camera permission and ARCore session startup: pass;
- Depth API support and `Depth: ON` state: pass;
- two valid physical hit points and visible markers: pass;
- measured distance display: pass (24.4 cm during the acceptance run);
- reset and second measurement readiness: pass;
- visible purple test cube placement: pass; and
- stable anchored cube while the phone moved: pass.

## Build and run

Requirements:

- JDK 17;
- Android SDK Platform 37 and Build Tools 36.0.0;
- an ARCore- and Depth-capable Android device; and
- Google Play Services for AR installed on the device.

Build the debug APK:

```powershell
.\gradlew.bat :app:assembleDebug
```

Install it on a connected device:

```powershell
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

Set up and start the local Gemini backend by following [backend/README.md](backend/README.md), then connect the USB device to it:

```powershell
adb reverse tcp:8000 tcp:8000
```

The launcher label is **Spatial Commerce**.

## Interaction

For measurement:

1. Move the phone slowly until ARCore tracks the environment.
2. In **Measure** mode, tap point A and point B on tracked physical surfaces.
3. Read the 3D distance shown in centimeters and meters.
4. Tap **Reset** to clear all anchors.

For product preview:

1. Select **Preview product** and move slowly until ARCore tracks an upward-facing floor or tabletop.
2. Tap the surface to place the Lighthouse Lounge Chair at its declared physical scale.
3. Use **Rotate -15°** and **Rotate +15°** to adjust its orientation.
4. Tap another valid surface to reposition the single product instance.
5. Select **Measure** to return to two-point measurement, or tap **Reset** to clear every anchor.

For product retrieval:

1. Complete a visual analysis that detects a product.
2. Tap **Find similar products** and wait for the results panel.
3. Scroll the results and tap one to select it; **Hide** closes the panel and **Retry search** appears after failures.

For visual analysis:

1. Start the local backend and establish `adb reverse` as described above.
2. Point the AR camera at one clear physical product.
3. Tap **Analyze object** once and hold the phone steady briefly.
4. Review the structured attributes and generated shopping-search phrases in the lower overlay.
5. Tap **Analyze object** again to analyze the current view, or **Retry analysis** after a recoverable error.
