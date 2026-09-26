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
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sqrt

class ArRenderer(
    private val activity: MainActivity,
    private val tapQueue: ConcurrentLinkedQueue<TapEvent>,
    private val onStatus: (String) -> Unit,
    private val onDistance: (String) -> Unit,
) : GLSurfaceView.Renderer {
    private val backgroundRenderer = CameraBackgroundRenderer()
    private val cubeRenderer = CubeRenderer()
    private val measurementAnchors = mutableListOf<Anchor>()
    private var cubeAnchor: Anchor? = null
    private var session: Session? = null
    private var mode = InteractionMode.MEASURE
    private var depthSupported = false
    private var textureBoundToSession = false
    private var viewportWidth = 1
    private var viewportHeight = 1
    private var lastStatus = ""
    private var lastDistance = ""
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
        publishStatus(
            if (newMode == InteractionMode.MEASURE) {
                measurementPrompt()
            } else {
                "Cube mode: tap a tracked surface to place a 10 cm cube."
            },
        )
    }

    fun reset() {
        resetPending = true
        publishDistance("Tap two points")
        publishStatus(measurementPrompt(0))
    }

    @Volatile
    private var resetPending = false

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        backgroundRenderer.createOnGlThread()
        cubeRenderer.createOnGlThread()
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
        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING) {
            publishStatus("Move the phone slowly so ARCore can find surfaces.")
            tapQueue.clear()
            return
        }

        processTap(frame)
        camera.getViewMatrix(viewMatrix, 0)
        camera.getProjectionMatrix(projectionMatrix, 0, 0.1f, 100f)

        measurementAnchors.forEachIndexed { index, anchor ->
            if (anchor.trackingState == TrackingState.TRACKING) {
                val color = if (index == 0) FIRST_MARKER_COLOR else SECOND_MARKER_COLOR
                cubeRenderer.draw(anchor.pose, viewMatrix, projectionMatrix, 0.035f, color, liftByHalf = true)
            }
        }
        cubeAnchor?.takeIf { it.trackingState == TrackingState.TRACKING }?.let { anchor ->
            cubeRenderer.draw(anchor.pose, viewMatrix, projectionMatrix, 0.10f, CUBE_COLOR, liftByHalf = true)
        }

        if (measurementAnchors.size == 2) {
            publishDistance(formatDistance(distanceMeters(measurementAnchors[0], measurementAnchors[1])))
        } else if (mode == InteractionMode.MEASURE) {
            publishStatus(measurementPrompt())
        }
    }

    fun releaseAnchors() = clearAnchors()

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
                    publishDistance(formatDistance(distanceMeters(measurementAnchors[0], measurementAnchors[1])))
                    publishStatus("Measurement complete. Reset to start over, or choose Place cube.")
                } else {
                    publishStatus("First point placed. Tap the second physical point.")
                }
            }
            InteractionMode.PLACE_CUBE -> {
                cubeAnchor?.detach()
                cubeAnchor = hit.createAnchor()
                publishStatus("10 cm cube placed. Tap another surface to move it.")
            }
        }
    }

    @Synchronized
    private fun clearAnchors() {
        measurementAnchors.forEach(Anchor::detach)
        measurementAnchors.clear()
        cubeAnchor?.detach()
        cubeAnchor = null
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

    private fun publishStatus(message: String) {
        if (message != lastStatus) {
            lastStatus = message
            onStatus(message)
        }
    }

    private fun publishDistance(message: String) {
        if (message != lastDistance) {
            lastDistance = message
            onDistance(message)
        }
    }

    companion object {
        private val FIRST_MARKER_COLOR = floatArrayOf(0.18f, 0.82f, 1f, 1f)
        private val SECOND_MARKER_COLOR = floatArrayOf(1f, 0.35f, 0.55f, 1f)
        private val CUBE_COLOR = floatArrayOf(0.55f, 0.30f, 1f, 1f)
    }
}
