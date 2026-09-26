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

For visual analysis:

1. Start the local backend and establish `adb reverse` as described above.
2. Point the AR camera at one clear physical product.
3. Tap **Analyze object** once and hold the phone steady briefly.
4. Review the structured attributes and generated shopping-search phrases in the lower overlay.
5. Tap **Analyze object** again to analyze the current view, or **Retry analysis** after a recoverable error.
