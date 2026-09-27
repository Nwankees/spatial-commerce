package com.hackgt.spatialcommerce

import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.Matrix
import com.google.ar.core.Anchor
import com.google.ar.core.Coordinates2d
import com.google.ar.core.DepthPoint
import com.google.ar.core.Frame
import com.google.ar.core.HitResult
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.Pose
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.NotYetAvailableException
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicInteger
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.abs
import kotlin.math.sqrt

class ArRenderer(
    private val activity: MainActivity,
    private val product: ProductPreview,
    private val touchQueue: ConcurrentLinkedQueue<SurfaceTouchEvent>,
    private val onStatus: (String) -> Unit,
    private val onInfo: (String) -> Unit,
    private val onCameraFrame: (CapturedCameraFrame) -> Unit,
    private val onCameraCaptureError: (String) -> Unit,
    /** Available width/depth in meters (null = not measured yet). Called on the GL thread. */
    private val onSpaceMeasured: (Float?, Float?) -> Unit,
    /** Milestone 6: the real-product model could not be shown (GL thread). */
    private val onRealProductError: (String) -> Unit = {},
    /** First frame where the real product is actually drawable (GL thread). */
    private val onRealProductVisible: (placement: String, uploadMillis: Long) -> Unit = { _, _ -> },
) : GLSurfaceView.Renderer {
    private val backgroundRenderer = CameraBackgroundRenderer()
    private val boxRenderer = BoxRenderer()
    private val productRenderer = ProductRenderer(boxRenderer)
    private val footprintRenderer = FootprintOverlayRenderer()
    private val measurementAnchors = mutableListOf<Anchor>()
    // Measure-space mode: A/B establish the first edge; a drag derives the perpendicular breadth.
    private val spaceAnchors = mutableListOf<Anchor>()
    private var spaceRectangle: FootprintRectangle? = null
    private var spaceRectangleAnchor: Anchor? = null
    private var spaceMeasurementComplete = false
    private var spaceDragEndpoint: FootprintEndpoint? = null
    // Milestone 6: selected real product, reconstructed and scaled to verified dimensions.
    private val realMeshRenderer = TexturedMeshRenderer()
    @Volatile private var pendingRealMesh: GlbMesh? = null
    @Volatile private var realMesh: GlbMesh? = null
    @Volatile private var clearRealRequested = false
    @Volatile private var realScale: ArScale? = null
    @Volatile private var realTitle: String = "Selected product"
    private var realAnchor: Anchor? = null
    private var realPlacementYawDegrees = 0f
    private var realRotationDegrees = 0f
    private var realFitRotationDegrees = 0f
    @Volatile private var automaticRealPlacementPending = false
    private var realVisibilityReportPending = false
    private var realPlacementKind = "tap"
    private var realUploadMillis = 0L
    private val pendingRealRotation = AtomicInteger(0)
    private val realModelMatrix = FloatArray(16)
    private val anchorMatrix = FloatArray(16)
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
            InteractionMode.PREVIEW_REAL_PRODUCT -> {
                automaticRealPlacementPending = realAnchor == null
                forcePublishInfo(realInfoText())
                publishStatus(realPrompt())
            }
        }
    }

    /** Hands a decoded real-product mesh to the GL thread; it replaces any previous one. */
    fun setRealProduct(mesh: GlbMesh, scale: ArScale, title: String, rotateToFit: Boolean) {
        realTitle = title
        realScale = scale
        realFitRotationDegrees = ArPreviewMath.footprintPlacementYaw(0.0, rotateToFit)
        realMesh = mesh
        pendingRealMesh = mesh
    }

    /** Forgets the real product (another product was selected); GL objects are freed on the GL thread. */
    fun clearRealProduct() {
        realScale = null
        realMesh = null
        pendingRealMesh = null
        clearRealRequested = true
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
            InteractionMode.PREVIEW_REAL_PRODUCT -> {
                publishInfo(realInfoText())
                publishStatus("AR placement cleared. Tap View in my space to place the product again.")
            }
        }
    }

    fun rotateProduct(degrees: Int) {
        if (mode == InteractionMode.PREVIEW_REAL_PRODUCT) {
            if (realAnchor == null) {
                publishStatus("Place the product on a tracked floor or tabletop first.")
            } else {
                pendingRealRotation.addAndGet(degrees)
            }
            return
        }
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
        footprintRenderer.createOnGlThread()
        realMeshRenderer.invalidate()
        realMesh?.let { pendingRealMesh = it }
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
            touchQueue.clear()
            return
        }

        processTouchEvents(frame)
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
        if (clearRealRequested) {
            clearRealRequested = false
            realMeshRenderer.release()
            realAnchor?.detach()
            realAnchor = null
            realPlacementYawDegrees = 0f
            realRotationDegrees = 0f
            automaticRealPlacementPending = false
            realVisibilityReportPending = false
        }
        pendingRealMesh?.let { mesh ->
            pendingRealMesh = null
            realAnchor?.detach()
            realAnchor = null
            realPlacementYawDegrees = 0f
            realRotationDegrees = 0f
            try {
                val uploadStarted = android.os.SystemClock.elapsedRealtime()
                realMeshRenderer.upload(mesh)
                realUploadMillis = android.os.SystemClock.elapsedRealtime() - uploadStarted
                automaticRealPlacementPending = mode == InteractionMode.PREVIEW_REAL_PRODUCT
                publishStatus(realPrompt())
            } catch (e: Exception) {
                onRealProductError(e.message ?: "The 3D model could not be displayed.")
            }
        }
        if (mode == InteractionMode.PREVIEW_REAL_PRODUCT && automaticRealPlacementPending && realMeshRenderer.isLoaded) {
            attemptAutomaticRealPlacement(frame)
        }
        if (mode == InteractionMode.MEASURE_SPACE || mode == InteractionMode.PREVIEW_REAL_PRODUCT) {
            drawMeasuredFootprint()
        }
        val realDelta = pendingRealRotation.getAndSet(0)
        if (realDelta != 0) {
            realRotationDegrees = (realRotationDegrees + realDelta) % 360f
            publishStatus("Rotated to ${realRotationDegrees.toInt()}°. Tap another surface to reposition.")
        }
        if (mode == InteractionMode.PREVIEW_REAL_PRODUCT) {
            val scale = realScale
            realAnchor?.takeIf { it.trackingState == TrackingState.TRACKING && scale != null && realMeshRenderer.isLoaded }?.let { anchor ->
                anchor.pose.toMatrix(anchorMatrix, 0)
                ArPreviewMath.modelMatrix(
                    realModelMatrix,
                    anchorMatrix,
                    realPlacementYawDegrees + realRotationDegrees,
                    scale!!,
                )
                realMeshRenderer.draw(realModelMatrix, viewMatrix, projectionMatrix)
                if (realVisibilityReportPending) {
                    realVisibilityReportPending = false
                    onRealProductVisible(realPlacementKind, realUploadMillis)
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

    private fun processTouchEvents(frame: Frame) {
        while (true) {
            val event = touchQueue.poll() ?: return
            if (mode == InteractionMode.MEASURE_SPACE) {
                processSpaceTouch(frame, event)
            } else if (event.action == SurfaceTouchAction.UP) {
                processTap(frame, event)
            }
        }
    }

    private fun processTap(frame: Frame, tap: SurfaceTouchEvent) {
        val hit = trackedHit(frame, tap.x, tap.y)
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
            InteractionMode.PREVIEW_REAL_PRODUCT -> placeRealProductAtHit(horizontalHit(frame, tap.x, tap.y))
            InteractionMode.PREVIEW_PRODUCT -> {
                val productHit = horizontalHit(frame, tap.x, tap.y)
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
            InteractionMode.MEASURE_SPACE -> Unit
        }
    }

    private fun processSpaceTouch(frame: Frame, event: SurfaceTouchEvent) {
        if (event.action == SurfaceTouchAction.CANCEL) {
            spaceDragEndpoint = null
            if (!spaceMeasurementComplete) spaceRectangle = null
            return
        }
        if (spaceMeasurementComplete) {
            if (event.action == SurfaceTouchAction.UP) publishStatus("Footprint measured. Tap Reset to measure again.")
            return
        }
        if (spaceAnchors.size < 2) {
            if (event.action != SurfaceTouchAction.UP) return
            val hit = horizontalHit(frame, event.x, event.y)
            if (hit == null) {
                publishStatus("Aim at one tracked floor or tabletop, then tap again.")
                return
            }
            if (spaceAnchors.size == 1 && abs(spaceAnchors[0].pose.ty() - hit.hitPose.ty()) > MAX_SPACE_Y_DRIFT_METERS) {
                publishStatus("Point B must be on the same horizontal surface as point A.")
                return
            }
            val anchor = hit.createAnchor()
            if (spaceAnchors.size == 1 && horizontalDistance(spaceAnchors[0].pose, anchor.pose) < MIN_SPACE_EDGE_METERS) {
                anchor.detach()
                publishStatus("Move point B farther from point A to define the first edge.")
                return
            }
            spaceAnchors += anchor
            publishInfo(spaceInfoText())
            publishStatus(spacePrompt())
            if (spaceAnchors.size == 2) onSpaceMeasured(spaceWidthMeters(), null)
            return
        }

        val hit = horizontalHit(frame, event.x, event.y)
        when (event.action) {
            SurfaceTouchAction.DOWN -> {
                spaceRectangle = null
                if (hit == null) {
                    publishStatus("Start the breadth drag on corner A or B.")
                    return
                }
                val point = hit.hitPose.toFootprintPoint()
                val a = spaceAnchors[0].pose.toFootprintPoint()
                val b = spaceAnchors[1].pose.toFootprintPoint()
                val threshold = (horizontalDistance(spaceAnchors[0].pose, spaceAnchors[1].pose) * 0.22f)
                    .coerceIn(MIN_ENDPOINT_TOUCH_METERS, MAX_ENDPOINT_TOUCH_METERS)
                spaceDragEndpoint = when {
                    point.horizontalDistanceTo(a) <= threshold -> FootprintEndpoint.A
                    point.horizontalDistanceTo(b) <= threshold -> FootprintEndpoint.B
                    else -> null
                }
                if (spaceDragEndpoint == null) {
                    publishStatus("Start the breadth drag on either end of the cyan A–B edge.")
                } else {
                    publishStatus("Drag across the surface to set the footprint breadth, then release.")
                }
            }
            SurfaceTouchAction.MOVE, SurfaceTouchAction.UP -> {
                val endpoint = spaceDragEndpoint ?: return
                if (hit == null) {
                    if (event.action == SurfaceTouchAction.UP) {
                        spaceDragEndpoint = null
                        publishStatus("The drag left the tracked surface. Start again from A or B.")
                    }
                    return
                }
                val edge = FootprintEdge(
                    spaceAnchors[0].pose.toFootprintPoint(),
                    spaceAnchors[1].pose.toFootprintPoint(),
                )
                val rectangle = FootprintGeometry.rectangleFromDrag(
                    edge,
                    endpoint,
                    hit.hitPose.toFootprintPoint(),
                    minimumBreadthMeters = MIN_SPACE_BREADTH_METERS.toDouble(),
                )
                if (rectangle != null) {
                    spaceRectangle = rectangle
                    publishInfo(spaceInfoText())
                }
                if (event.action == SurfaceTouchAction.UP) {
                    spaceDragEndpoint = null
                    if (rectangle == null) {
                        spaceRectangle = null
                        publishStatus("Drag farther across the surface to set the footprint breadth.")
                    } else {
                        spaceMeasurementComplete = true
                        spaceRectangleAnchor?.detach()
                        spaceRectangleAnchor = session?.createAnchor(
                            Pose.makeTranslation(
                                rectangle.center.x.toFloat(),
                                rectangle.center.y.toFloat(),
                                rectangle.center.z.toFloat(),
                            ),
                        )
                        onSpaceMeasured(rectangle.widthMeters.toFloat(), rectangle.depthMeters.toFloat())
                        publishStatus("Footprint measured. The shaded rectangle is the space being tested.")
                    }
                }
            }
            SurfaceTouchAction.CANCEL -> Unit
        }
    }

    private fun trackedHit(frame: Frame, x: Float, y: Float): HitResult? =
        frame.hitTest(x, y).firstOrNull { result ->
            when (val trackable = result.trackable) {
                is Plane -> trackable.isPoseInPolygon(result.hitPose)
                is Point -> trackable.orientationMode == Point.OrientationMode.ESTIMATED_SURFACE_NORMAL
                is DepthPoint -> true
                else -> false
            }
        }

    private fun horizontalHit(frame: Frame, x: Float, y: Float): HitResult? =
        frame.hitTest(x, y).firstOrNull { result ->
            val plane = result.trackable as? Plane
            plane?.type == Plane.Type.HORIZONTAL_UPWARD_FACING && plane.isPoseInPolygon(result.hitPose)
        }

    private fun attemptAutomaticRealPlacement(frame: Frame) {
        automaticRealPlacementPending = false
        val measured = currentSpaceRectangle()?.takeIf { spaceMeasurementComplete }
        if (measured != null) {
            val arSession = session
            if (arSession != null) {
                runCatching {
                    val center = measured.center
                    replaceRealAnchor(
                        arSession.createAnchor(
                            Pose.makeTranslation(center.x.toFloat(), center.y.toFloat(), center.z.toFloat()),
                        ),
                        placementYawDegrees = measured.alignmentYawDegrees.toFloat() + realFitRotationDegrees,
                        placementKind = "measured-footprint",
                    )
                }.onSuccess {
                    publishInfo(realInfoText())
                    publishStatus("$realTitle placed automatically in the measured footprint at verified scale.")
                    return
                }
            }
        }

        val samples = arrayOf(
            floatArrayOf(viewportWidth * 0.50f, viewportHeight * 0.52f),
            floatArrayOf(viewportWidth * 0.50f, viewportHeight * 0.62f),
            floatArrayOf(viewportWidth * 0.40f, viewportHeight * 0.56f),
            floatArrayOf(viewportWidth * 0.60f, viewportHeight * 0.56f),
        )
        val hit = samples.asSequence()
            .mapNotNull { horizontalHit(frame, it[0], it[1]) }
            .firstOrNull { it.distance in MIN_AUTO_PLACE_DISTANCE_METERS..MAX_AUTO_PLACE_DISTANCE_METERS }
        if (hit != null) {
            replaceRealAnchor(hit.createAnchor(), placementYawDegrees = 0f, placementKind = "screen-center-plane")
            publishInfo(realInfoText())
            publishStatus("$realTitle placed automatically at verified scale. Rotate it or tap another surface to reposition.")
        } else {
            publishStatus("Automatic placement needs a clearer horizontal surface. Tap a tracked floor or tabletop to place it.")
        }
    }

    private fun placeRealProductAtHit(hit: HitResult?) {
        if (!realMeshRenderer.isLoaded) {
            publishStatus("The 3D preview is still loading.")
            return
        }
        if (hit == null) {
            publishStatus("Aim at a tracked floor or tabletop, move slowly, and tap again.")
            return
        }
        automaticRealPlacementPending = false
        val firstPlacement = realAnchor == null
        val measured = currentSpaceRectangle()?.takeIf { spaceMeasurementComplete }
        if (measured != null) {
            val moved = measured.translatedTo(hit.hitPose.toFootprintPoint())
            spaceRectangle = moved
            spaceRectangleAnchor?.detach()
            spaceRectangleAnchor = session?.createAnchor(
                Pose.makeTranslation(
                    moved.center.x.toFloat(),
                    moved.center.y.toFloat(),
                    moved.center.z.toFloat(),
                ),
            )
            val anchor = session?.createAnchor(
                Pose.makeTranslation(
                    moved.center.x.toFloat(),
                    moved.center.y.toFloat(),
                    moved.center.z.toFloat(),
                ),
            ) ?: hit.createAnchor()
            replaceRealAnchor(
                anchor,
                placementYawDegrees = moved.alignmentYawDegrees.toFloat() + realFitRotationDegrees,
                placementKind = "tap-reposition",
                reportVisibility = firstPlacement,
            )
        } else {
            replaceRealAnchor(
                hit.createAnchor(),
                placementYawDegrees = 0f,
                placementKind = "tap",
                reportVisibility = firstPlacement,
            )
        }
        publishInfo(realInfoText())
        publishStatus("$realTitle repositioned at verified scale. Rotate it or tap another surface to move it again.")
    }

    private fun replaceRealAnchor(
        anchor: Anchor,
        placementYawDegrees: Float,
        placementKind: String,
        reportVisibility: Boolean = true,
    ) {
        realAnchor?.detach()
        realAnchor = anchor
        realPlacementYawDegrees = placementYawDegrees
        realPlacementKind = placementKind
        realVisibilityReportPending = reportVisibility
    }

    private fun drawMeasuredFootprint() {
        val rectangle = currentSpaceRectangle()
        if (rectangle == null) {
            spaceAnchors.forEachIndexed { index, anchor ->
                if (anchor.trackingState == TrackingState.TRACKING) {
                    val color = if (index == 0) SPACE_WIDTH_COLOR else SPACE_DEPTH_COLOR
                    boxRenderer.drawCube(anchor.pose, viewMatrix, projectionMatrix, 0.035f, color, liftByHalf = true)
                }
            }
            if (spaceAnchors.size == 2 && spaceAnchors.all { it.trackingState == TrackingState.TRACKING }) {
                footprintRenderer.drawEdge(
                    spaceAnchors[0].pose.translation.withOverlayLift(),
                    spaceAnchors[1].pose.translation.withOverlayLift(),
                    viewMatrix,
                    projectionMatrix,
                )
            }
            return
        }

        val corners = rectangle.corners.map { it.toFloatArray(FOOTPRINT_OVERLAY_LIFT_METERS) }
        footprintRenderer.drawFootprint(
            corners[0], corners[1], corners[2], corners[3],
            viewMatrix, projectionMatrix,
            showFill = realAnchor == null,
        )
        rectangle.corners.forEachIndexed { index, point ->
            val color = if (index < 2) SPACE_WIDTH_COLOR else SPACE_DEPTH_COLOR
            boxRenderer.drawCube(
                Pose.makeTranslation(point.x.toFloat(), point.y.toFloat(), point.z.toFloat()),
                viewMatrix,
                projectionMatrix,
                0.035f,
                color,
                liftByHalf = true,
            )
        }
    }

    private fun currentSpaceRectangle(): FootprintRectangle? {
        val rectangle = spaceRectangle ?: return null
        val trackingAnchor = realAnchor?.takeIf { it.trackingState == TrackingState.TRACKING }
            ?: spaceRectangleAnchor?.takeIf { it.trackingState == TrackingState.TRACKING }
        return trackingAnchor?.pose?.toFootprintPoint()?.let(rectangle::translatedTo) ?: rectangle
    }

    @Synchronized
    private fun clearAnchors() {
        measurementAnchors.forEach(Anchor::detach)
        measurementAnchors.clear()
        val hadSpace = spaceAnchors.isNotEmpty() || spaceRectangle != null
        spaceAnchors.forEach(Anchor::detach)
        spaceAnchors.clear()
        spaceRectangleAnchor?.detach()
        spaceRectangleAnchor = null
        spaceRectangle = null
        spaceMeasurementComplete = false
        spaceDragEndpoint = null
        if (hadSpace) onSpaceMeasured(null, null)
        productAnchor?.detach()
        productAnchor = null
        realAnchor?.detach()
        realAnchor = null
        realPlacementYawDegrees = 0f
        realRotationDegrees = 0f
        automaticRealPlacementPending = false
        realVisibilityReportPending = false
        pendingRealRotation.set(0)
        productPlaced = false
        productRotationDegrees = 0f
        pendingRotationDegrees.set(0)
        touchQueue.clear()
    }

    private fun distanceMeters(first: Anchor, second: Anchor): Float {
        val a = first.pose.translation
        val b = second.pose.translation
        val dx = a[0] - b[0]
        val dy = a[1] - b[1]
        val dz = a[2] - b[2]
        return sqrt(dx * dx + dy * dy + dz * dz)
    }

    private fun horizontalDistance(first: Pose, second: Pose): Float {
        val dx = first.tx() - second.tx()
        val dz = first.tz() - second.tz()
        return sqrt(dx * dx + dz * dz)
    }

    private fun spaceWidthMeters(): Float? =
        spaceRectangle?.widthMeters?.toFloat()
            ?: if (spaceAnchors.size >= 2) horizontalDistance(spaceAnchors[0].pose, spaceAnchors[1].pose) else null

    private fun spaceInfoText(count: Int = spaceAnchors.size): String {
        val width = if (count >= 2) spaceWidthMeters()?.let { String.format(Locale.US, "%.2f m", it) } else null
        val breadth = if (count >= 2) {
            spaceRectangle?.depthMeters?.let { String.format(Locale.US, "%.2f m", it) }
        } else {
            null
        }
        val breadthLabel = breadth ?: if (count >= 2) "drag from A or B" else "not measured"
        return "Footprint length: ${width ?: "not measured"}\nFootprint breadth: $breadthLabel"
    }

    private fun spacePrompt(count: Int = spaceAnchors.size): String {
        val depth = if (depthSupported) "Depth: ON" else "Depth: unavailable"
        if (spaceMeasurementComplete) return "$depth  •  Footprint rectangle ready. Check Fit or View in my space."
        return when (count) {
            0 -> "$depth  •  Footprint: tap corner A on a floor or tabletop."
            1 -> "$depth  •  Tap corner B to define the first edge."
            else -> "$depth  •  Press on A or B, drag across the surface for breadth, then release."
        }
    }

    private fun realInfoText(): String {
        val scale = realScale ?: return "Real product preview"
        return String.format(
            Locale.US,
            "%s\n%.2f × %.2f × %.2f m (W × D × H, verified)\nAI-generated 3D preview scaled to verified product dimensions.",
            realTitle.take(60), scale.widthMeters, scale.depthMeters, scale.heightMeters,
        )
    }

    private fun realPrompt(): String {
        val depth = if (depthSupported) "Depth: ON" else "Depth: unavailable"
        return when {
            realAnchor != null -> "$depth  •  Rotate the product or tap another surface to reposition."
            automaticRealPlacementPending -> "$depth  •  Finding the best tracked surface for automatic placement…"
            else -> "$depth  •  Tap a tracked floor or tabletop to place the product."
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
        private const val MIN_SPACE_EDGE_METERS = 0.08f
        private const val MIN_SPACE_BREADTH_METERS = 0.08f
        private const val MAX_SPACE_Y_DRIFT_METERS = 0.08f
        private const val MIN_ENDPOINT_TOUCH_METERS = 0.10f
        private const val MAX_ENDPOINT_TOUCH_METERS = 0.25f
        private const val FOOTPRINT_OVERLAY_LIFT_METERS = 0.004f
        private const val MIN_AUTO_PLACE_DISTANCE_METERS = 0.45f
        private const val MAX_AUTO_PLACE_DISTANCE_METERS = 4.0f
        private const val MAX_CAMERA_CAPTURE_ATTEMPTS = 90
    }
}

private fun Pose.toFootprintPoint() = FootprintPoint(tx().toDouble(), ty().toDouble(), tz().toDouble())

private fun FootprintPoint.toFloatArray(yLift: Float = 0f) =
    floatArrayOf(x.toFloat(), y.toFloat() + yLift, z.toFloat())

private fun FloatArray.withOverlayLift(): FloatArray =
    floatArrayOf(this[0], this[1] + 0.004f, this[2])
