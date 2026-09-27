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

## Milestone 5: spatial fit (baseline; physically verified end to end together with Milestone 5.5)

Selected real product → trustworthy product dimensions → measured available space → deterministic fit result.

### Architecture

- `POST /api/v1/products/dimensions` takes only `{"productId", "productUrl"}` of a product the backend itself returned from `/products/search`. The backend looks the product up in its own record (the product-search cache, which also stores the provider's server-side detail reference); unknown or mismatched products get a 404, and client-supplied dimensions or URLs are never trusted.
- `DimensionResolver` gathers evidence from replaceable sources:
  - `ProductDetailSource` → `SerpApiImmersiveProductDetailSource` (SerpApi Google Immersive Product API, keyed by the search result's `immersive_product_page_token`): explicit spec fields (`about_the_product.features`) and the list of stores with direct retailer links.
  - `PageFetcher` → `HttpPageFetcher`: one plain HTTP GET per retailer page (no JavaScript), at most two pages, fetched concurrently, the result's own retailer first. Only public `http(s)` hosts are fetched (no IP literals, localhost, or Google URLs).
  - `dimension_extraction` parses each page into per-source findings; `dimension_parsing` holds the pure, conservative parsing and unit rules.
- Android adds **Measure space** mode, a **Check fit** action on the selected product, a `FitPanel`, and the pure `FitEngine`.

### Dimension trust rules

- Values are only taken from explicit source data, in this priority order: (1) schema.org JSON-LD on the retailer page (`width`/`depth`/`height` QuantitativeValues and `additionalProperty`), (2) structured metadata (provider product-detail spec fields and page microdata), (3) retailer spec tables (`<table>` rows and `<dl>` pairs), (4) explicitly labeled dimensions in retailer page text near a dimensions keyword.
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
