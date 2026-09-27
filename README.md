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

## Milestone 6: real-product AR preview

"View in my space" shows the selected real product in AR: its product image is turned into a
textured 3D model by Stable Fast 3D (SF3D, open source, running locally on the GPU), normalized,
scaled to the product's **verified retailer measurements** and placed on a detected floor or tabletop.

The generated geometry is a visual approximation, not CAD truth. Physical size never comes from the
mesh: the backend only scales to three measurements from its own verified dimension record. Source labels,
an exact labeled duplicate, or—when the result is unambiguous—mesh proportions assign those values to
width/depth/height. Without all three, or when an unlabeled set cannot be mapped safely, the app reports
that real-scale preview is unavailable. A selected real product is never replaced by the built-in chair.

### Architecture

```
Android (ArPreviewClient)
  POST /api/v1/products/ar-preview {productId, productUrl}
    -> product registry lookup (selected result only) -> three verified measurements (M5 resolver, cached)
    -> safe image download (public http(s) only, redirects re-checked, 12 MB, decoded + size-checked)
    -> cache key sha256(product id + URL, selected-image hash, verified-dimension identity/version,
       provider/version, normalization version)
    -> ready (cache hit) | generating (job) | unavailable | failed
  GET  /api/v1/ar-assets/{assetId}            poll status (32-hex ids only)
  GET  /api/v1/ar-assets/{assetId}/model.glb  normalized GLB (ready assets only)
Backend ArPreviewService
  ReconstructionProvider (SidecarReconstructionProvider -> reconstruction_service, SF3D)
  -> load exactly one mesh -> validate -> normalize (center XZ, floor at y=0, footprint yaw
     straightened, bounds recorded) -> GLB -> re-validate -> atomic save in .cache/ar-assets/<id>/
  -> labeled axes directly, otherwise score every source-compatible W/D/H permutation against mesh
     aspect ratios; reject poor or ambiguous matches
  -> scale per axis = verified retailer meters / normalized extent
Android GlbParser -> TexturedMeshRenderer (OpenGL ES 2.0, existing ARCore renderer)
  model = anchor * yaw(measured-edge alignment + fit rotation + user ±15°, +90° if mesh axes swapped)
          * scale(x, y, z)
```

Failures are never cached (a failed product is retried after 20 s); successful assets are reused
across requests and restarts. Only generated asset ids are served; ids that are not 32 lowercase hex
characters, unknown ids and anything outside the asset store return 404. Model weights, generated
meshes, downloaded images and caches are git-ignored and never committed.

Setup of the SF3D sidecar (Python 3.11 env, weights on D:, Hugging Face license) is in
[reconstruction_service/README.md](reconstruction_service/README.md).

### Known limitations

- The only image used is the selected search result's image; retailer thumbnails can be low
  resolution or show a styled scene, which lowers reconstruction quality.
- SF3D reconstructs the unseen back side; it is plausible, not measured.
- Per-axis scaling can stretch the mesh when its proportions disagree with the verified box; the
  distortion is recorded as `maxAxisDistortion`.
- First generation of a product takes tens of seconds (model warm-up plus CPU texture baking);
  cached assets load in about a second.

### Manual test procedure

1. Start Ollama, the backend and `reconstruction_service\run_service.ps1`; `adb reverse tcp:8000 tcp:8000`.
2. Analyze an object, find products, select one, Check fit.
3. Tap **Measure space**, tap A and B for the first footprint edge, then press either endpoint and
   drag across the horizontal surface to set breadth. Confirm the translucent rectangle and four corners.
4. With three verified measurements, tap **View in my space**; the product should appear automatically
   centered in the rectangle. The fill disappears while the outline remains. Without a rectangle, the app
   tries a tracked plane near screen center and falls back to tap placement when no confident point exists.
5. Rotate with ±15°, tap another surface to move the product and outline together, then Reset to clear both.
6. Compare with a tape measure against the verified dimensions shown in the info panel.
7. Select a product without verified height: the preview must be unavailable, with no chair.

## Milestone 5.5: local vision + multi-query retrieval

Gemini is no longer on the active path. `/api/v1/analyze` now calls a **local Ollama vision model** (default `qwen3-vl:8b` for demo latency), and product search runs **several searches per object** and merges them.

### Local vision architecture

Android → FastAPI `/api/v1/analyze` → `OllamaVisionService` → Ollama `POST /api/chat` on the same PC (`OLLAMA_BASE_URL`, default `http://127.0.0.1:11434`). The request uses `stream: false`, `think: false`, `temperature: 0.1`, `keep_alive: 15m`, the JPEG as a base64 image, and `format` set to the analysis JSON schema so Ollama constrains the output. The backend then validates the JSON against `VisualProductAnalysis` and applies deterministic guards:

- a detection below `VISION_MIN_CONFIDENCE` (0.2) becomes a safe "no product" result;
- a detection with no product type and no queries is rejected as incomplete (retryable 502);
- `visibleSpecifications` are kept only if their value appears in the legible `visibleText`;
- a brand/model hypothesis that claims `visible_text` evidence but is not in `visibleText` is downgraded to `design_resemblance` (its confidence is kept).

Ollama unreachable, model not installed, timeouts, malformed JSON and schema mismatches all become retryable 502/504 errors. `/health` reports whether Ollama is reachable and whether the configured model is installed. Each analysis logs model, outcome, duration and Ollama's own timings; the response carries `X-Vision-Model` and `X-Analysis-Duration-Ms` headers. There is no automatic fallback to Gemini.

Required setup: install Ollama, `ollama pull qwen3-vl:8b` (the larger `qwen3-vl:30b` remains an optional accuracy/latency tradeoff), and keep Ollama running. Environment variables: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS` (180), `OLLAMA_KEEP_ALIVE`, `VISION_MIN_CONFIDENCE`, `PRODUCT_SEARCH_MAX_QUERIES` (3), `PRODUCT_SEARCH_RESULTS_PER_QUERY` (10).

### Richer analysis contract

`VisualProductAnalysis` now carries `category`, `subcategory`, `brand` and `modelFamily` as optional hypotheses (`value`, `confidence`, `evidence` = `visible_text` | `logo` | `design_resemblance`), `visibleText`, `color`, `materials`, `style`, `shape`, `distinctiveFeatures`, text-backed `visibleSpecifications`, 2–6 `searchQueries`, `confidence`, `uncertaintyNotes`, and `message`. Unknown fields stay null or empty. The prompt tells the model its output feeds a shopping search, asks it to keep distinguishing details and printed text, and forbids guessed dimensions, wattage, connector standards, model numbers or brands. **Exact technical specs are never inferred from images.** The old `searchKeywords` name is still accepted as input.

### Multi-query retrieval and reranking

`QueryPlanner` takes the model's queries (most specific first) plus the deterministic Milestone 4 query as a fallback, removes near-duplicates, and caps the list (`PRODUCT_SEARCH_MAX_QUERIES`, default 3). The searches run concurrently through the existing `ProductSearchProvider` (SerpApi). Results are merged and deduplicated by provider product ID, product URL, and normalized title + retailer, keeping each product's best-ranked occurrence and recording every query that returned it (`matchedQueries`).

`ProductReranker` scores each candidate with explicit, testable terms: provider rank, extra queries, brand/type/identity/feature agreement, Lens exact/product/visual rank, agreement across text and Lens, and purchasability. The same photographed ARCore frame is uploaded directly to SerpApi's Image API (maximum 500 KB prepared copy) and its short-lived `image_id` drives concurrent Google Lens `products`, `visual_matches`, and `exact_matches` searches—no public image hosting. Results are deduplicated by canonical retailer URL, provider/merchant/global identifiers, Google offer identity, and normalized title + source.

The top merged shortlist (default 5—the same number the UI can display) is compared in one local `qwen3-vl` batch call against the original photo. Structured exact-product, visual-similarity, and category-match scores augment the deterministic ranking. Candidate-image download or local-model failure is fail-open: Lens/text results remain usable. Text and Lens candidate generation also degrade independently. Responses keep `retrievalSources`, text/visual ranks, local similarity, `combinedScore`, and a latency breakdown for debugging; the Android cards only show the final products and a compact “Visual + product search” status.

Milestone 5 compatibility: merged products keep their SerpApi product ID, detail token (server-side), links, retailer, price and image, and are recorded in the product cache, so **Check fit** works on them unchanged.

### Known limitations

- A cold local model load is slower than later requests. The 8B demo default avoids the multi-minute 30B path on the event laptop.
- Candidate images that are unavailable or block backend downloads keep their Lens/text score but skip local visual comparison.
- A visual search uses one SerpApi image upload plus up to three concurrent Lens modes, alongside up to 3 Google Shopping searches (and one detail call per **Check fit**).

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

### Dimension resolution update: structured page semantics

The active M6 pipeline is retailer page → generic structured page representation → `qwen3:4b-instruct` semantic source selection → deterministic provenance validation → meter normalization. It does not use regular expressions to decide which page fields the model may see.

`page_representation.py` preserves the page title, selected variant/options, product JSON-LD, embedded product/application JSON, specification tables, `<dl>` name/value pairs, headings with their following content, and compact visible text blocks. It removes navigation, footer, cookie/consent UI, advertisements, recommendations, reviews, Q&A, tracking/media blobs and other obvious noise. Each retained entry has a stable ID, source type, human-readable path, scope and owning item ID. Exact retailer identity is applied before model extraction, so sibling variants are not mixed into the selected item's evidence.

Resolution order, without mixing values between sources:

1. Fetch and scope the exact selected variant's retailer page when its identity is known.
2. Use a no-model fast path only when one structured section explicitly names Width, Depth and Height.
3. Otherwise give the compact structured representation to `qwen3:4b-instruct` with a strict JSON response schema.
4. Deterministically verify that every cited entry was actually shown to the model, every value and unit occurs there, the cited sources belong to one compatible section/variant, and any axis mapping is supported by source labels.
5. Convert the verified source values to meters in code. Unlabeled triples such as `13.39 x 4.13 x 4.13 in` remain verified in source order with unresolved axis indices; they are not discarded.
6. If the exact page does not resolve, apply the same representation/model/validation pipeline to provider family specs and then other retailer pages.

The prompt forbids invented or estimated values, package/shipping dimensions, and seat/arm/back/shelf/interior or other component measurements. Multiple plausible overall sets return uncertain rather than being guessed. The model identifies meaning; software establishes provenance. Regex is used only after the model chooses evidence, for exact number/unit/axis verification and deterministic conversion.

Setup: `ollama pull qwen3:4b-instruct`. Env: `OLLAMA_DIMENSION_MODEL`, `OLLAMA_DIMENSION_TIMEOUT_SECONDS`, `OLLAMA_DIMENSION_NUM_CTX`, `DIMENSION_LLM_ENABLED`, and `DIMENSION_PAGE_MAX_CHARS`. `/health` reports `dimensionModel.installed`. Only successful results are cached. Run `backend/scripts/dimension_debug.py` to print the selected variant, full structured model input, raw Qwen output, provenance validation, normalized dimensions, axis mapping, and fit/M6 eligibility.

Known retailer limitations: no browser automation is used, so pages that block plain requests or omit product data from server HTML remain unavailable. An unlabeled triple can be trusted as measurements but cannot be assigned to width/depth/height until M6 compares its proportions with the generated mesh; ambiguous or poor matches remain unavailable instead of being guessed.

### Selected-variant matching

SerpApi's product-detail specs describe Google's product *family*; a probe of seven retailers showed they can describe a different variant than the offer the user picked (e.g. "Color: Black" while the selected Target/Walmart offers were Gray). The resolver therefore builds a `VariantContext` for the selected result (`variant_context.py`) from real identifiers only:

- Google IDs parsed from the result's `product_link` (`catalogid`, `productid`, `headlineOfferDocid`, `gpcid`, `mid`);
- the concrete offer: the product-detail `stores[]` entry with the selected retailer name **and** the selected price (ambiguous → no offer);
- retailer IDs read from that offer's URL by a small rule table (Target `A-<item>`, Walmart `/ip/<item>` + `selectedOfferId`, Wayfair `~<sku>` + `PiID`, Home Depot, Kohl's `skuid`, Staples, Best Buy `/sku/`, Office Depot, Quill, Lowe's, Costco, Amazon) plus generic query parameters (`variant`, `sku`, `skuid`, `itemId`, `pid`, `preselect`, `offerId`, …);
- explicitly selected options: SerpApi `variants[].items[].selected == true`, or an explicit `Color: X`/`Size: Y` in the offer title.

Identity is `exact_item` (retailer item/SKU/variant ID), `exact_offer` (offer ID), `options_only`, or `product_only`.

Policy:

1. **Exact identity**: the selected offer's page is fetched and embedded product data is scoped to the object whose identifier key (`tcin`, `usItemId`, `sku`, `variantId`, `id`, …) *exactly* equals the retailer ID. Sibling records are excluded before the structured representation is given to the model. Explicit separate W/D/H fields can use the fast path; every other label/value shape is interpreted by the local model and then provenance-validated.
2. **Options only**: options scope a record only if they match exactly one item ID in the retailer's own data; zero or several matches leave the variant unresolved.
3. A complete exact-variant result wins immediately and is never mixed with family data. If the exact page has no complete three-value result, the resolver may fall back to a complete product-family source rather than combine partial values.

`ResolvedDimensions` now includes `variantScope` (`exact_variant`, `exact_variant_page`, `product_family`, `retailer_page`) and `variantIdentity`. Search deduplication no longer merges two results with the same title and retailer when their Google offer IDs (`headlineOfferDocid`) differ, so variants are not merged. AI-generated retailer summaries (e.g. `genAi…Summary`) are excluded from all dimension evidence.

### Dimension trust rules

- Values are only taken from explicit source data. Exact selected-variant evidence outranks product-family or generic retailer-page evidence; values from different sources are never mixed.
- A three-value overall measurement wins over a partial result at the same trust level. Unlabeled triples remain verified measurements with unresolved axes.
- Never used as measurement truth: images, mesh size, category averages, Gemini, product titles, or model world knowledge. Embedded JSON is usable only through the scoped structured representation and the same provenance rules.
- A number needs an explicit unit (mm, cm, m, in/″, ft, UN/CEFACT `MMT`/`CMT`/`MTR`/`INH`/`FOT`) or a unit declared by its label, e.g. `Width (in)`. Bare numbers are rejected.
- Source axes are mapped only from explicit labels: `Width`/`Overall Width`/…, `W`/`D`/`H` letters on each value, or a declared order such as `Dimensions (W x D x H)`. Unlabeled three-value measurements are retained with null indices. During M6 only, mesh proportions may infer their axis permutation when one mapping is clearly better; the physical meter values still come exclusively from the retailer. Ambiguous or badly mismatched meshes are rejected.
- Values outside 0.01–10 m are rejected as misparses.

### Dimension provenance model

`ResolvedDimensions` includes `dimensionsMeters` in source order, an explicit `axisMapping`, separately populated `widthMeters`/`depthMeters`/`heightMeters` only when the source labels those axes, `status`, source type/name/URL/path, raw evidence, extraction method, variant scope/identity, retryability and failure message. `verified` means three product measurements are provenance-verified; it does not imply their axes were labeled.

### Fit formula

```
fits ⇔ (productWidth ≤ availableWidth AND productDepth ≤ availableDepth)
    OR (productDepth ≤ availableWidth AND productWidth ≤ availableDepth)
widthRemaining = availableWidth − productWidth − clearance   (not clamped; negative = over)
depthRemaining = availableDepth − productDepth − clearance
```

The app evaluates both 0° and 90° footprint orientations and reports when rotation is required. Missing or unresolved product width/depth → `UNKNOWN`; missing available width/depth → `NEEDS_MEASUREMENT`. No LLM participates in the fit calculation.

### Manual test procedure

1. Start the backend and `adb reverse tcp:8000 tcp:8000`; launch the app.
2. Analyze a real object, tap **Find similar products**, select a result, tap **Check fit**.
3. Confirm the panel shows either product dimensions with their source, or **UNKNOWN — dimensions unavailable** with a reason (never invented values).
4. Tap **Measure space**: tap A and B for the first footprint edge, then drag from either endpoint across the surface to set breadth. Confirm the rectangle and two footprint readouts.
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
