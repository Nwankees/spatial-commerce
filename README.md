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

The launcher label is **Spatial Measure**.

## Interaction

1. Move the phone slowly until ARCore tracks the environment.
2. In **Measure** mode, tap point A and point B on tracked physical surfaces.
3. Read the 3D distance shown in centimeters and meters.
4. Tap **Reset** to clear all anchors.
5. Select **Place cube**, tap a tracked surface, and move around to confirm that the cube remains anchored.

