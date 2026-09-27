package com.hackgt.spatialcommerce

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import com.google.ar.core.Anchor
import com.google.ar.core.Coordinates2d
import com.google.ar.core.DepthPoint
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sqrt

class ArRenderer(
    private val activity: MainActivity,
    private val product: ProductPreview,
    private val tapQueue: ConcurrentLinkedQueue<TapEvent>,
    private val onStatus: (String) -> Unit,
    private val onInfo: (String) -> Unit,
    private val onCameraFrame: (CapturedCameraFrame) -> Unit,
    private val onCameraCaptureError: (String) -> Unit,
    /** Available width/depth in meters (null = not measured yet). Called on the GL thread. */
    private val onSpaceMeasured: (Float?, Float?) -> Unit,
) : GLSurfaceView.Renderer {
    private val backgroundRenderer = CameraBackgroundRenderer()
    private val boxRenderer = BoxRenderer()
    private val productRenderer = ProductRenderer(boxRenderer)
    private val measurementAnchors = mutableListOf<Anchor>()
    // Measure-space mode: [widthA, widthB, depthA, depthB], same ARCore hit points as Measure.
    private val spaceAnchors = mutableListOf<Anchor>()
    private var productAnchor: Anchor? = null
    private var session: Session? = null
    @Volatile
    private var mode = InteractionMode.MEASURE
    @Volatile
    private var productPlaced = false
    private val pendingRotationDegrees = AtomicInteger(0)
    private var productRotationDegrees = 0f
    private var depthSupported = false
    private var textureBoundToSession = false
    private var viewportWidth = 1
    private var viewportHeight = 1
    private var lastStatus = ""
    private var lastInfo = ""
    @Volatile
    private var cameraCaptureRequested = false
    private var cameraCaptureAttempts = 0
    private val viewMatrix = FloatArray(16)
    private val projectionMatrix = FloatArray(16)

    @Synchronized
    fun setSession(newSession: Session, supportsDepth: Boolean) {
        session = newSession
        depthSupported = supportsDepth
        textureBoundToSession = false
    }

    fun setMode(newMode: InteractionMode) {
        mode = newMode
        when (newMode) {
            InteractionMode.MEASURE -> {
                forcePublishInfo(if (measurementAnchors.size == 2) formatDistance(distanceMeters(measurementAnchors[0], measurementAnchors[1])) else "Tap two points")
                publishStatus(measurementPrompt())
            }
            InteractionMode.PREVIEW_PRODUCT -> {
                forcePublishInfo(product.overlayText())
                publishStatus(previewPrompt())
            }
            InteractionMode.MEASURE_SPACE -> {
                forcePublishInfo(spaceInfoText())
                publishStatus(spacePrompt())
            }
        }
    }

    fun requestCameraCapture() {
        cameraCaptureAttempts = 0
        cameraCaptureRequested = true
    }

    fun reset() {
        resetPending = true
        when (mode) {
            InteractionMode.MEASURE -> {
                publishInfo("Tap two points")
                publishStatus(measurementPrompt(0))
            }
            InteractionMode.PREVIEW_PRODUCT -> {
                publishInfo(product.overlayText())
                publishStatus(previewPrompt(false))
            }
            InteractionMode.MEASURE_SPACE -> {
                publishInfo(spaceInfoText(0))
                publishStatus(spacePrompt(0))
            }
        }
    }

    fun rotateProduct(degrees: Int) {
        if (!productPlaced) {
            publishStatus("Place the ${product.name} on a tracked floor or tabletop first.")
            return
        }
        pendingRotationDegrees.addAndGet(degrees)
    }

    @Volatile
    private var resetPending = false

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        backgroundRenderer.createOnGlThread()
        boxRenderer.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        viewportWidth = width
        viewportHeight = height
        GLES20.glViewport(0, 0, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        if (resetPending) {
            clearAnchors()
            resetPending = false
        }

        val arSession = session ?: return
        arSession.setDisplayGeometry(activity.currentDisplayRotation(), viewportWidth, viewportHeight)
        if (!textureBoundToSession) {
            arSession.setCameraTextureName(backgroundRenderer.textureId)
            textureBoundToSession = true
        }

        val frame = try {
            arSession.update()
        } catch (_: CameraNotAvailableException) {
            publishStatus("Camera unavailable. Reopen the app after closing other camera apps.")
            return
        }

        backgroundRenderer.draw(frame)
        captureCameraFrameIfRequested(frame, arSession)
        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING) {
            publishStatus("Move the phone slowly so ARCore can find surfaces.")
            tapQueue.clear()
            return
        }

        processTap(frame)
        camera.getViewMatrix(viewMatrix, 0)
        camera.getProjectionMatrix(projectionMatrix, 0, 0.1f, 100f)

        if (mode == InteractionMode.MEASURE) {
            measurementAnchors.forEachIndexed { index, anchor ->
                if (anchor.trackingState == TrackingState.TRACKING) {
                    val color = if (index == 0) FIRST_MARKER_COLOR else SECOND_MARKER_COLOR
                    boxRenderer.drawCube(anchor.pose, viewMatrix, projectionMatrix, 0.035f, color, liftByHalf = true)
                }
            }
        }
        if (mode == InteractionMode.MEASURE_SPACE) {
            spaceAnchors.forEachIndexed { index, anchor ->
                if (anchor.trackingState == TrackingState.TRACKING) {
                    val color = if (index < 2) SPACE_WIDTH_COLOR else SPACE_DEPTH_COLOR
                    boxRenderer.drawCube(anchor.pose, viewMatrix, projectionMatrix, 0.035f, color, liftByHalf = true)
                }
            }
        }
        val rotationDelta = pendingRotationDegrees.getAndSet(0)
        if (rotationDelta != 0) {
            productRotationDegrees = (productRotationDegrees + rotationDelta) % 360f
            publishStatus("${product.name} rotated to ${productRotationDegrees.toInt()}°. Tap another surface to reposition it.")
        }
        if (mode == InteractionMode.PREVIEW_PRODUCT) {
            productAnchor?.takeIf { it.trackingState == TrackingState.TRACKING }?.let { anchor ->
                productRenderer.draw(product, anchor.pose, productRotationDegrees, viewMatrix, projectionMatrix)
            }
        }

        if (measurementAnchors.size == 2) {
            if (mode == InteractionMode.MEASURE) {
                publishInfo(formatDistance(distanceMeters(measurementAnchors[0], measurementAnchors[1])))
            }
        } else if (mode == InteractionMode.MEASURE) {
            publishStatus(measurementPrompt())
        }
    }

    fun releaseAnchors() = clearAnchors()

    private fun captureCameraFrameIfRequested(frame: Frame, arSession: Session) {
        if (!cameraCaptureRequested) return
        cameraCaptureAttempts += 1
        try {
            val jpegBytes = frame.acquireCameraImage().use(CameraFrameEncoder::toJpeg)
            val rotationDegrees = activity.cameraImageRotationDegrees(arSession.cameraConfig.cameraId)
            cameraCaptureRequested = false
            cameraCaptureAttempts = 0
            onCameraFrame(CapturedCameraFrame(jpegBytes, rotationDegrees))
        } catch (_: NotYetAvailableException) {
            if (cameraCaptureAttempts >= MAX_CAMERA_CAPTURE_ATTEMPTS) {
                cameraCaptureRequested = false
                cameraCaptureAttempts = 0
                onCameraCaptureError("A camera frame was not available. Keep the phone steady and retry.")
            }
        } catch (exception: Exception) {
            cameraCaptureRequested = false
            cameraCaptureAttempts = 0
            onCameraCaptureError(
                exception.message ?: "The current AR camera frame could not be captured.",
            )
        }
    }

    private fun processTap(frame: Frame) {
        val tap = tapQueue.poll() ?: return
        val hit = frame.hitTest(tap.x, tap.y).firstOrNull { result ->
            when (val trackable = result.trackable) {
                is Plane -> trackable.isPoseInPolygon(result.hitPose)
                is Point -> trackable.orientationMode == Point.OrientationMode.ESTIMATED_SURFACE_NORMAL
                is DepthPoint -> true
                else -> false
            }
        }

        if (hit == null) {
            publishStatus("No tracked surface there yet. Move slowly, aim at a textured surface, and tap again.")
            return
        }

        when (mode) {
            InteractionMode.MEASURE -> {
                if (measurementAnchors.size >= 2) {
                    publishStatus("Measurement complete. Tap Reset to measure again.")
                    return
                }
                measurementAnchors += hit.createAnchor()
                if (measurementAnchors.size == 2) {
                    publishInfo(formatDistance(distanceMeters(measurementAnchors[0], measurementAnchors[1])))
                    publishStatus("Measurement complete. Reset to start over, or choose Preview product.")
                } else {
                    publishStatus("First point placed. Tap the second physical point.")
                }
            }
            InteractionMode.MEASURE_SPACE -> {
                if (spaceAnchors.size >= 4) {
                    publishStatus("Space measured. Tap Reset to measure it again.")
                    return
                }
                spaceAnchors += hit.createAnchor()
                publishInfo(spaceInfoText())
                publishStatus(spacePrompt())
                when (spaceAnchors.size) {
                    2 -> onSpaceMeasured(spaceWidthMeters(), null)
                    4 -> onSpaceMeasured(spaceWidthMeters(), spaceDepthMeters())
                }
            }
            InteractionMode.PREVIEW_PRODUCT -> {
                val productHit = frame.hitTest(tap.x, tap.y).firstOrNull { result ->
                    val plane = result.trackable as? Plane
                    plane?.type == Plane.Type.HORIZONTAL_UPWARD_FACING && plane.isPoseInPolygon(result.hitPose)
                }
                if (productHit == null) {
                    publishStatus("Aim at a tracked floor or tabletop, move slowly, and tap again.")
                    return
                }
                productAnchor?.detach()
                productAnchor = productHit.createAnchor()
                productPlaced = true
                publishInfo(product.overlayText())
                publishStatus("${product.name} placed at true scale. Rotate it or tap another surface to reposition.")
            }
        }
    }

    @Synchronized
    private fun clearAnchors() {
        measurementAnchors.forEach(Anchor::detach)
        measurementAnchors.clear()
        val hadSpace = spaceAnchors.isNotEmpty()
        spaceAnchors.forEach(Anchor::detach)
        spaceAnchors.clear()
        if (hadSpace) onSpaceMeasured(null, null)
        productAnchor?.detach()
        productAnchor = null
        productPlaced = false
        productRotationDegrees = 0f
        pendingRotationDegrees.set(0)
        tapQueue.clear()
    }

    private fun distanceMeters(first: Anchor, second: Anchor): Float {
        val a = first.pose.translation
        val b = second.pose.translation
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    private fun spaceWidthMeters(): Float? =
        if (spaceAnchors.size >= 2) distanceMeters(spaceAnchors[0], spaceAnchors[1]) else null

    private fun spaceDepthMeters(): Float? =
        if (spaceAnchors.size >= 4) distanceMeters(spaceAnchors[2], spaceAnchors[3]) else null

    private fun spaceInfoText(count: Int = spaceAnchors.size): String {
        val width = if (count >= 2) spaceWidthMeters()?.let { String.format(Locale.US, "%.2f m", it) } else null
        val depth = if (count >= 4) spaceDepthMeters()?.let { String.format(Locale.US, "%.2f m", it) } else null
        return "Available width: ${width ?: "not measured"}\nAvailable depth: ${depth ?: "not measured"}"
    }

    private fun spacePrompt(count: Int = spaceAnchors.size): String {
        val depth = if (depthSupported) "Depth: ON" else "Depth: unavailable"
        return when (count) {
            0 -> "$depth  •  Measure space: tap the LEFT edge of the available width."
            1 -> "$depth  •  Tap the RIGHT edge of the available width."
            2 -> "$depth  •  Width recorded. Tap the FRONT edge of the available depth."
            3 -> "$depth  •  Tap the BACK edge of the available depth."
            else -> "$depth  •  Available space recorded. Reset to measure again."
        }
    }

    private fun formatDistance(meters: Float): String = if (meters < 1f) {
        String.format(Locale.US, "%.1f cm  •  %.3f m", meters * 100f, meters)
    } else {
        String.format(Locale.US, "%.3f m  •  %.1f cm", meters, meters * 100f)
    }

    private fun measurementPrompt(count: Int = measurementAnchors.size): String {
        val depth = if (depthSupported) "Depth: ON" else "Depth: unavailable"
        return when (count) {
            0 -> "$depth  •  Move slowly to find a surface, then tap point A."
            1 -> "$depth  •  Point A placed. Tap point B."
            else -> "$depth  •  Measurement complete."
        }
    }

    private fun previewPrompt(isPlaced: Boolean = productPlaced): String {
        val depth = if (depthSupported) "Depth: ON" else "Depth: unavailable"
        return if (isPlaced) {
            "$depth  •  Rotate the product or tap another floor/table surface to reposition."
        } else {
            "$depth  •  Move slowly to find a floor or tabletop, then tap to preview."
        }
    }

    private fun publishStatus(message: String) {
        if (message != lastStatus) {
            lastStatus = message
            onStatus(message)
        }
    }

    private fun publishInfo(message: String) {
        if (message != lastInfo) {
            lastInfo = message
            onInfo(message)
        }
    }

    private fun forcePublishInfo(message: String) {
        lastInfo = message
        onInfo(message)
    }

    companion object {
        private val FIRST_MARKER_COLOR = floatArrayOf(0.18f, 0.82f, 1f, 1f)
        private val SECOND_MARKER_COLOR = floatArrayOf(1f, 0.35f, 0.55f, 1f)
        private val SPACE_WIDTH_COLOR = floatArrayOf(1f, 0.78f, 0.18f, 1f)
        private val SPACE_DEPTH_COLOR = floatArrayOf(0.35f, 0.9f, 0.45f, 1f)
        private const val MAX_CAMERA_CAPTURE_ATTEMPTS = 90
    }
}
