package com.hackgt.spatialcommerce

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.view.Gravity
import android.view.MotionEvent
import android.view.Surface
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import com.google.ar.core.ArCoreApk
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableException
import java.util.concurrent.ConcurrentLinkedQueue

class MainActivity : Activity() {
    private lateinit var surfaceView: GLSurfaceView
    private lateinit var renderer: ArRenderer
    private lateinit var statusText: TextView
    private lateinit var infoText: TextView
    private lateinit var measureButton: Button
    private lateinit var previewButton: Button
    private lateinit var measureSpaceButton: Button
    private lateinit var fitPanel: FitPanel
    private lateinit var analyzeButton: Button
    private lateinit var findProductsButton: Button
    private lateinit var resultsPanel: ProductResultsPanel
    private lateinit var rotationControls: LinearLayout
    private val previewProduct = PreviewProducts.lighthouseLoungeChair
    private val backendClient = GeminiBackendClient()
    private val productSearchClient = ProductSearchClient()

    // Milestone 4 state. Only touched on the UI thread.
    private var lastAnalysis: VisualProductAnalysis? = null
    private var analysisGeneration = 0
    private var productSearchInFlight = false
    private var lastProductSearch: ProductSearchResult? = null
    /** The candidate the user picked; the handoff point for later milestones. */
    private var selectedProduct: ProductCandidate? = null

    // Milestone 5 state. Only touched on the UI thread.
    private var availableWidthMeters: Double? = null
    private var availableDepthMeters: Double? = null
    private var fitProduct: ProductCandidate? = null
    private var fitDimensions: ResolvedDimensions? = null
    private var fitGeneration = 0
    private var fitLookupInFlight = false

    private var session: Session? = null
    private var installRequested = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildUi()
    }

    private fun buildUi() {
        val root = FrameLayout(this)
        surfaceView = TapSurfaceView(this).apply {
            setEGLContextClientVersion(2)
            preserveEGLContextOnPause = true
        }

        statusText = TextView(this).apply {
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(190, 17, 19, 24))
            textSize = 16f
            gravity = Gravity.CENTER
            setPadding(dp(16), dp(12), dp(16), dp(12))
            text = "Starting ARCore..."
        }

        infoText = TextView(this).apply {
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.argb(210, 26, 29, 36))
            textSize = 20f
            gravity = Gravity.CENTER
            setPadding(dp(18), dp(12), dp(18), dp(12))
            text = "Tap two points"
        }

        measureButton = modeButton("Measure")
        previewButton = modeButton("Preview product")
        measureSpaceButton = modeButton("Measure space")
        val resetButton = modeButton("Reset")
        val rotateLeftButton = modeButton("Rotate -15°")
        val rotateRightButton = modeButton("Rotate +15°")
        analyzeButton = modeButton("Analyze object").apply {
            setPadding(dp(14), 0, dp(14), 0)
        }

        measureButton.setOnClickListener {
            renderer.setMode(InteractionMode.MEASURE)
            updateModeUi(InteractionMode.MEASURE)
        }
        previewButton.setOnClickListener {
            renderer.setMode(InteractionMode.PREVIEW_PRODUCT)
            updateModeUi(InteractionMode.PREVIEW_PRODUCT)
        }
        measureSpaceButton.setOnClickListener { enterMeasureSpace() }
        resetButton.setOnClickListener { renderer.reset() }
        rotateLeftButton.setOnClickListener { renderer.rotateProduct(-15) }
        rotateRightButton.setOnClickListener { renderer.rotateProduct(15) }
        analyzeButton.setOnClickListener { beginObjectAnalysis() }
        findProductsButton = modeButton("Find similar products").apply {
            setPadding(dp(14), 0, dp(14), 0)
            visibility = View.GONE
            setOnClickListener { beginProductSearch() }
        }
        resultsPanel = ProductResultsPanel(
            context = this,
            onProductSelected = { product -> selectProduct(product) },
            onRetry = { beginProductSearch() },
            onCheckFit = { beginFitCheck() },
        )
        fitPanel = FitPanel(
            context = this,
            onMeasureSpace = { enterMeasureSpace() },
            onRetry = { beginFitCheck() },
        )

        rotationControls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(12), dp(4), dp(12), 0)
            addView(rotateLeftButton, weightedButtonParams())
            addView(rotateRightButton, weightedButtonParams())
        }

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            setPadding(dp(8), dp(8), dp(8), dp(12))
            addView(measureButton, weightedButtonParams())
            addView(previewButton, weightedButtonParams())
            addView(measureSpaceButton, weightedButtonParams())
            addView(resetButton, weightedButtonParams())
        }

        val bottomPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.argb(170, 17, 19, 24))
            addView(resultsPanel, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
            addView(fitPanel, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
            addView(infoText, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
            addView(rotationControls, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
            addView(controls, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }

        root.addView(surfaceView, FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.MATCH_PARENT)
        root.addView(
            statusText,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.TOP),
        )
        root.addView(
            analyzeButton,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, dp(50), Gravity.TOP or Gravity.END).apply {
                topMargin = dp(64)
                marginEnd = dp(12)
            },
        )
        root.addView(
            findProductsButton,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, dp(50), Gravity.TOP or Gravity.END).apply {
                topMargin = dp(122)
                marginEnd = dp(12)
            },
        )
        root.addView(
            bottomPanel,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM),
        )
        setContentView(root)

        val tapQueue = ConcurrentLinkedQueue<TapEvent>()
        renderer = ArRenderer(
            activity = this,
            product = previewProduct,
            tapQueue = tapQueue,
            onStatus = { text -> statusText.post { statusText.text = text } },
            onInfo = { text -> infoText.post { infoText.text = text } },
            onCameraFrame = { frame -> onCameraFrameCaptured(frame) },
            onCameraCaptureError = { message -> showAnalysisError(message) },
            onSpaceMeasured = { width, depth -> runOnUiThread { onAvailableSpaceMeasured(width, depth) } },
        )
        surfaceView.setRenderer(renderer)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        surfaceView.setOnTouchListener { view, event ->
            if (event.action == MotionEvent.ACTION_UP) {
                tapQueue.add(TapEvent(event.x, event.y))
                view.performClick()
                true
            } else {
                true
            }
        }
        updateModeUi(InteractionMode.MEASURE)
    }

    override fun onResume() {
        super.onResume()
        if (!hasCameraPermission()) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_REQUEST)
            return
        }

        if (session == null) {
            try {
                when (ArCoreApk.getInstance().requestInstall(this, !installRequested)) {
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED -> {
                        installRequested = true
                        statusText.text = "Install or update Google Play Services for AR, then return here."
                        return
                    }
                    ArCoreApk.InstallStatus.INSTALLED -> Unit
                }
                session = Session(this).also { arSession ->
                    val depthSupported = arSession.isDepthModeSupported(Config.DepthMode.AUTOMATIC)
                    val config = arSession.config.apply {
                        planeFindingMode = Config.PlaneFindingMode.HORIZONTAL_AND_VERTICAL
                        updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                        focusMode = Config.FocusMode.AUTO
                        depthMode = if (depthSupported) Config.DepthMode.AUTOMATIC else Config.DepthMode.DISABLED
                    }
                    arSession.configure(config)
                    renderer.setSession(arSession, depthSupported)
                }
            } catch (exception: UnavailableException) {
                showFatal("ARCore is unavailable: ${exception.javaClass.simpleName}")
                return
            }
        }

        try {
            session?.resume()
        } catch (_: CameraNotAvailableException) {
            showFatal("The camera is unavailable. Close other camera apps and reopen this app.")
            session = null
            return
        }
        surfaceView.onResume()
    }

    override fun onPause() {
        super.onPause()
        surfaceView.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        renderer.releaseAnchors()
        backendClient.close()
        productSearchClient.close()
        session?.close()
        session = null
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != CAMERA_PERMISSION_REQUEST) return
        if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            onResume()
        } else {
            showFatal("Camera permission is required for the AR experience.")
        }
    }

    @Suppress("DEPRECATION")
    fun currentDisplayRotation(): Int = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
        display?.rotation ?: Surface.ROTATION_0
    } else {
        windowManager.defaultDisplay.rotation
    }

    fun cameraImageRotationDegrees(cameraId: String): Int {
        val cameraManager = getSystemService(CameraManager::class.java)
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        val sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        val deviceOrientation = when (currentDisplayRotation()) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return if (characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT) {
            (sensorOrientation + deviceOrientation) % 360
        } else {
            (sensorOrientation - deviceOrientation + 360) % 360
        }
    }

    private fun beginObjectAnalysis() {
        // A new analysis supersedes the previous object; stale search results are ignored.
        analysisGeneration += 1
        lastAnalysis = null
        productSearchInFlight = false
        findProductsButton.visibility = View.GONE
        resultsPanel.visibility = View.GONE
        analyzeButton.isEnabled = false
        analyzeButton.text = "Capturing…"
        infoText.text = "Hold steady while the current camera view is captured."
        statusText.text = "Capturing one AR camera frame…"
        renderer.requestCameraCapture()
    }

    private fun onCameraFrameCaptured(frame: CapturedCameraFrame) {
        runOnUiThread {
            analyzeButton.text = "Analyzing…"
            infoText.text = "The local vision model is identifying the product. This can take a while on first use."
            statusText.text = "Analyzing object…"
        }
        backendClient.analyze(frame) { result ->
            runOnUiThread {
                result.fold(
                    onSuccess = { analysis ->
                        infoText.text = analysis.overlayText()
                        statusText.text = if (analysis.objectDetected) {
                            "Analysis complete. Tap Find similar products, or Analyze object to inspect again."
                        } else {
                            "No obvious product found. Center one object and retry."
                        }
                        if (analysis.objectDetected) {
                            lastAnalysis = analysis
                            findProductsButton.text = "Find similar products"
                            findProductsButton.isEnabled = true
                            findProductsButton.visibility = View.VISIBLE
                        }
                        analyzeButton.text = "Analyze object"
                        analyzeButton.isEnabled = true
                    },
                    onFailure = { exception ->
                        finishAnalysisWithError(
                            exception.message ?: "Analysis failed. Please retry.",
                        )
                    },
                )
            }
        }
    }

    private fun beginProductSearch() {
        val analysis = lastAnalysis ?: return
        if (productSearchInFlight) return
        productSearchInFlight = true
        val generation = analysisGeneration
        findProductsButton.isEnabled = false
        findProductsButton.text = "Searching…"
        resultsPanel.showLoading()
        statusText.text = "Searching for real purchasable products…"

        productSearchClient.search(analysis) { result ->
            runOnUiThread {
                if (generation != analysisGeneration) return@runOnUiThread
                productSearchInFlight = false
                findProductsButton.isEnabled = true
                result.fold(
                    onSuccess = { search ->
                        lastProductSearch = search
                        findProductsButton.text = "Find similar products"
                        resultsPanel.showResults(search, selectedProduct?.id)
                        statusText.text = when {
                            search.products.isEmpty() -> "No purchasable matches found. Retry or analyze another object."
                            search.isCached -> "Showing cached results; live search is unavailable. Tap a product to select it."
                            else -> "Found ${search.products.size} real products. Tap one to select it."
                        }
                    },
                    onFailure = { exception ->
                        findProductsButton.text = "Retry search"
                        val message = exception.message ?: "Product search failed. Please retry."
                        resultsPanel.showError(message)
                        statusText.text = "Product search did not complete."
                    },
                )
            }
        }
    }

    private fun selectProduct(product: ProductCandidate) {
        if (product.id != fitProduct?.id) clearFitCheck()
        selectedProduct = product
        resultsPanel.markSelected(product.id)
        val retailer = product.retailer?.let { " at $it" }.orEmpty()
        statusText.text = "Selected: ${product.title} — ${product.displayPrice()}$retailer. " +
            "AR preview still shows only the Lighthouse chair."
    }

    private fun beginFitCheck() {
        val product = selectedProduct ?: return
        if (fitLookupInFlight && fitProduct?.id == product.id) return
        fitGeneration += 1
        val generation = fitGeneration
        fitLookupInFlight = true
        fitProduct = product
        fitDimensions = null
        fitPanel.showLoading(product)
        statusText.text = "Looking up explicit product dimensions…"

        productSearchClient.resolveDimensions(product) { result ->
            runOnUiThread {
                if (generation != fitGeneration) return@runOnUiThread
                fitLookupInFlight = false
                result.fold(
                    onSuccess = { dimensions ->
                        fitDimensions = dimensions
                        refreshFit()
                    },
                    onFailure = { exception ->
                        fitPanel.showError(exception.message ?: "Dimension lookup failed. Please retry.")
                        statusText.text = "Fit check did not complete."
                    },
                )
            }
        }
    }

    /** Re-evaluates the deterministic fit whenever dimensions or measurements change. */
    private fun refreshFit() {
        val dimensions = fitDimensions ?: return
        val result = FitEngine.evaluate(
            productWidthMeters = dimensions.widthMeters,
            productDepthMeters = dimensions.depthMeters,
            availableWidthMeters = availableWidthMeters,
            availableDepthMeters = availableDepthMeters,
        )
        fitPanel.show(result, dimensions)
        statusText.text = when (result.verdict) {
            FitVerdict.FITS -> "Fit check: FITS the measured space."
            FitVerdict.DOES_NOT_FIT -> "Fit check: DOES NOT FIT the measured space."
            FitVerdict.UNKNOWN -> "Fit check: UNKNOWN — explicit width and depth are unavailable."
            FitVerdict.NEEDS_MEASUREMENT -> "Fit check: measure the available width and depth."
        }
    }

    private fun onAvailableSpaceMeasured(width: Float?, depth: Float?) {
        availableWidthMeters = width?.toDouble()
        availableDepthMeters = depth?.toDouble()
        refreshFit()
    }

    private fun enterMeasureSpace() {
        renderer.setMode(InteractionMode.MEASURE_SPACE)
        updateModeUi(InteractionMode.MEASURE_SPACE)
        resultsPanel.visibility = View.GONE
    }

    private fun clearFitCheck() {
        fitGeneration += 1
        fitLookupInFlight = false
        fitProduct = null
        fitDimensions = null
        fitPanel.visibility = View.GONE
    }

    private fun showAnalysisError(message: String) {
        runOnUiThread { finishAnalysisWithError(message) }
    }

    private fun finishAnalysisWithError(message: String) {
        infoText.text = message
        statusText.text = "Object analysis did not complete."
        analyzeButton.text = "Retry analysis"
        analyzeButton.isEnabled = true
    }

    private fun updateModeUi(mode: InteractionMode) {
        measureButton.isEnabled = mode != InteractionMode.MEASURE
        previewButton.isEnabled = mode != InteractionMode.PREVIEW_PRODUCT
        measureSpaceButton.isEnabled = mode != InteractionMode.MEASURE_SPACE
        rotationControls.visibility = if (mode == InteractionMode.PREVIEW_PRODUCT) View.VISIBLE else View.GONE
    }

    private fun showFatal(message: String) {
        statusText.text = message
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
    }

    private fun hasCameraPermission() = checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    private fun modeButton(label: String) = Button(this).apply {
        text = label
        textSize = 14f
        isAllCaps = false
    }

    private fun weightedButtonParams() = LinearLayout.LayoutParams(0, dp(52), 1f).apply {
        marginStart = dp(4)
        marginEnd = dp(4)
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    companion object {
        private const val CAMERA_PERMISSION_REQUEST = 1001
    }
}

private class TapSurfaceView(context: Context) : GLSurfaceView(context) {
    override fun performClick(): Boolean {
        super.performClick()
        return true
    }
}

data class TapEvent(val x: Float, val y: Float)

enum class InteractionMode {
    MEASURE,
    PREVIEW_PRODUCT,
    MEASURE_SPACE,
}
