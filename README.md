# Spatial Commerce

HackGT 13 project exploring spatially aware conversational commerce. The current working checkpoint is intentionally limited to the Android/ARCore foundation.

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
