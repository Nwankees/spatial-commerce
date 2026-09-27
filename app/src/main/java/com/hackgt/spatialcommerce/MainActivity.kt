package com.hackgt.spatialcommerce

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.opengl.GLSurfaceView
import android.net.Uri
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
import java.util.UUID

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
    private lateinit var conversationPanel: ConversationPanel
    private lateinit var chatButton: Button
    private val previewProduct = PreviewProducts.lighthouseLoungeChair
    private val backendClient = GeminiBackendClient()
    private val productSearchClient = ProductSearchClient()
    private val arPreviewClient = ArPreviewClient()
    private val conversationClient = ConversationClient()
    private val sponsorVoiceClient = SponsorVoiceClient()
    private val shopperId: String by lazy { stableShopperId() }
    private var currentMode = InteractionMode.MEASURE

    // Milestone 6 state. Only touched on the UI thread.
    private var arGeneration = 0
    private var arRequestInFlight = false

    // Milestone 4 state. Only touched on the UI thread.
    private var lastAnalysis: VisualProductAnalysis? = null
    /** Exact ARCore camera frame used for lastAnalysis and subsequent visual search. */
    private var lastAnalyzedFrame: CapturedCameraFrame? = null
    private var analysisGeneration = 0
    private var productSearchInFlight = false
    private var conversationGeneration = 0
    private var conversationRequestInFlight = false
    private var conversationContextVersion = 0
    private var lastProductSearch: ProductSearchResult? = null
    /** The candidate the user picked; the handoff point for later milestones. */
    private var selectedProduct: ProductCandidate? = null

    // Milestone 5 state. Only touched on the UI thread.
    private var availableWidthMeters: Double? = null
    private var availableDepthMeters: Double? = null
    private var fitProduct: ProductCandidate? = null
    private var fitDimensions: ResolvedDimensions? = null
    private var latestFitResult: FitResult? = null
    private var fitGeneration = 0
    private var fitLookupInFlight = false
    private var pendingConversationArProductId: String? = null
    @Volatile private var arPreviewStartedElapsedRealtime = 0L

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
        resetButton.setOnClickListener { resetArScene() }
        rotateLeftButton.setOnClickListener { renderer.rotateProduct(-15) }
        rotateRightButton.setOnClickListener { renderer.rotateProduct(15) }
        analyzeButton.setOnClickListener { beginObjectAnalysis() }
        findProductsButton = modeButton("Find similar products").apply {
            setPadding(dp(14), 0, dp(14), 0)
            visibility = View.GONE
            setOnClickListener { beginProductSearch() }
        }
        chatButton = modeButton("Chat").apply {
            setPadding(dp(14), 0, dp(14), 0)
        }
        conversationPanel = ConversationPanel(
            context = this,
            onSend = { message -> sendConversationMessage(message) },
            onClose = {
                sponsorVoiceClient.stop()
                conversationPanel.close()
            },
            onVoiceEnabledChanged = { enabled -> if (!enabled) sponsorVoiceClient.stop() },
            onStopVoice = { sponsorVoiceClient.stop() },
        )
        chatButton.setOnClickListener { conversationPanel.open() }
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
            onViewInSpace = { beginRealProductPreview() },
            onViewSource = { url ->
                runCatching { startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url))) }
                    .onFailure { Toast.makeText(this, "Could not open the source page.", Toast.LENGTH_SHORT).show() }
            },
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
            chatButton,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.WRAP_CONTENT, dp(50), Gravity.TOP or Gravity.START).apply {
                topMargin = dp(64)
                marginStart = dp(12)
            },
        )
        root.addView(
            bottomPanel,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, FrameLayout.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM),
        )
        root.addView(
            conversationPanel,
            FrameLayout.LayoutParams(FrameLayout.LayoutParams.MATCH_PARENT, dp(390), Gravity.BOTTOM),
        )
        setContentView(root)

        val touchQueue = ConcurrentLinkedQueue<SurfaceTouchEvent>()
        renderer = ArRenderer(
            activity = this,
            product = previewProduct,
            touchQueue = touchQueue,
            onStatus = { text -> statusText.post { statusText.text = text } },
            onInfo = { text -> infoText.post { infoText.text = text } },
            onCameraFrame = { frame -> onCameraFrameCaptured(frame) },
            onCameraCaptureError = { message -> showAnalysisError(message) },
            onSpaceMeasured = { width, depth -> runOnUiThread { onAvailableSpaceMeasured(width, depth) } },
            onRealProductError = { message -> runOnUiThread { failRealProductPreview(message, retryable = false) } },
            onRealProductVisible = { placement, uploadMillis ->
                val totalMillis = android.os.SystemClock.elapsedRealtime() - arPreviewStartedElapsedRealtime
                android.util.Log.i(
                    "SpatialCommerceM6",
                    "visible placement=$placement textureDecodeUpload=${uploadMillis}ms perceivedTotal=${totalMillis}ms",
                )
            },
        )
        surfaceView.setRenderer(renderer)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        surfaceView.setOnTouchListener { view, event ->
            val action = when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> SurfaceTouchAction.DOWN
                MotionEvent.ACTION_MOVE -> SurfaceTouchAction.MOVE
                MotionEvent.ACTION_UP -> SurfaceTouchAction.UP
                MotionEvent.ACTION_CANCEL -> SurfaceTouchAction.CANCEL
                else -> null
            }
            if (action != null) {
                touchQueue.add(SurfaceTouchEvent(action, event.x, event.y))
            }
            if (event.actionMasked == MotionEvent.ACTION_UP) {
                view.performClick()
            }
            true
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
        arPreviewClient.close()
        conversationClient.close()
        sponsorVoiceClient.close()
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
        conversationContextVersion += 1
        // A new analysis supersedes the previous object; stale search results are ignored.
        analysisGeneration += 1
        lastAnalysis = null
        lastAnalyzedFrame = null
        productSearchInFlight = false
        lastProductSearch = null
        selectedProduct = null
        clearFitCheck()
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
                            lastAnalyzedFrame = frame
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
        val frame = lastAnalyzedFrame ?: return
        if (productSearchInFlight) return
        productSearchInFlight = true
        val generation = analysisGeneration
        findProductsButton.isEnabled = false
        findProductsButton.text = "Searching…"
        resultsPanel.showLoading()
        statusText.text = "Running visual + product search…"

        productSearchClient.search(analysis, frame) { result ->
            runOnUiThread {
                if (generation != analysisGeneration) return@runOnUiThread
                productSearchInFlight = false
                findProductsButton.isEnabled = true
                result.fold(
                    onSuccess = { search ->
                        lastProductSearch = search
                        conversationContextVersion += 1
                        findProductsButton.text = "Find similar products"
                        resultsPanel.showResults(search, selectedProduct?.id)
                        statusText.text = when {
                            search.products.isEmpty() -> "No purchasable matches found. Retry or analyze another object."
                            search.isCached -> "Showing cached results; live search is unavailable. Tap a product to select it."
                            else -> "Found ${search.products.size} visually ranked products. Tap one to select it."
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

    private fun sendConversationMessage(message: String) {
        sponsorVoiceClient.stop()
        conversationGeneration += 1
        val generation = conversationGeneration
        val contextVersion = conversationContextVersion
        conversationRequestInFlight = true
        val context = ConversationContext(
            shopperId = shopperId,
            analysis = lastAnalysis,
            products = lastProductSearch?.products.orEmpty(),
            selectedProductId = selectedProduct?.id,
            availableWidthMeters = availableWidthMeters,
            availableDepthMeters = availableDepthMeters,
            frame = lastAnalyzedFrame,
        )
        conversationClient.send(
            message = message,
            context = context,
            onProgress = { progress ->
                runOnUiThread {
                    if (
                        generation != conversationGeneration ||
                        contextVersion != conversationContextVersion ||
                        !conversationRequestInFlight
                    ) return@runOnUiThread
                    val label = progress.label()
                    conversationPanel.showWorking(label)
                    statusText.text = label
                }
            },
        ) { result ->
            runOnUiThread {
                if (generation != conversationGeneration) return@runOnUiThread
                conversationRequestInFlight = false
                if (contextVersion != conversationContextVersion) {
                    conversationPanel.showError(
                        "The camera, product, or measurement changed while I was working, so I ignored that older result. Please ask again.",
                    )
                    return@runOnUiThread
                }
                result.fold(
                    onSuccess = { reply -> applyConversationReply(reply) },
                    onFailure = { error ->
                        conversationPanel.showError(error.message ?: "The shopping assistant is unavailable. Please retry.")
                    },
                )
            }
        }
    }

    private fun applyConversationReply(reply: ConversationReply) {
        conversationPanel.showReply(reply.message)
        conversationPanel.showPurchase(reply.purchase)
        if (conversationPanel.isVoiceEnabled()) {
            sponsorVoiceClient.speak(reply.message) { reason ->
                if (conversationPanel.isVoiceEnabled()) {
                    Toast.makeText(this, reason, Toast.LENGTH_SHORT).show()
                }
            }
        }
        var contextChanged = false
        if (reply.products.isNotEmpty()) {
            val search = reply.searchResult
                ?: lastProductSearch?.copy(products = reply.products, message = reply.message)
                ?: ProductSearchResult(
                    query = "Conversation results",
                    provider = "agent",
                    resultSource = "live",
                    cachedAt = null,
                    products = reply.products,
                    message = reply.message,
                )
            lastProductSearch = search
            contextChanged = true
            resultsPanel.showResults(search, reply.selectedProductId)
        }
        if (
            reply.action in setOf("find_similar_products", "refine_search") &&
            reply.selectedProductId == null &&
            selectedProduct != null
        ) {
            clearFitCheck()
            selectedProduct = null
            resultsPanel.markSelected(null)
            contextChanged = true
        }
        val selected = reply.selectedProduct
            ?: lastProductSearch?.products?.firstOrNull { it.id == reply.selectedProductId }
        if (selected != null && selected.id != selectedProduct?.id) {
            selectProduct(selected)
            contextChanged = false // selectProduct already advances the context version.
        }
        if (contextChanged) conversationContextVersion += 1
        when (reply.uiDirective) {
            "analyze_object" -> {
                conversationPanel.close()
                beginObjectAnalysis()
            }
            "show_products" -> {
                conversationPanel.close()
                resultsPanel.visibility = View.VISIBLE
                statusText.text = reply.message
            }
            "measure_space" -> {
                conversationPanel.close()
                enterMeasureSpace()
            }
            "show_fit" -> statusText.text = reply.message
            "enter_ar_preview" -> {
                conversationPanel.close()
                beginConversationArPreview(selected ?: selectedProduct)
            }
            "open_checkout" -> {
                val url = reply.checkoutUrl ?: reply.purchase?.checkoutUrl
                if (url != null) {
                    runCatching { startActivity(android.content.Intent(android.content.Intent.ACTION_VIEW, android.net.Uri.parse(url))) }
                        .onFailure { Toast.makeText(this, "Could not open the merchant page.", Toast.LENGTH_SHORT).show() }
                }
            }
            else -> statusText.text = reply.message
        }
    }

    private fun selectProduct(product: ProductCandidate) {
        val changed = selectedProduct?.id != product.id
        if (product.id != fitProduct?.id) clearFitCheck()
        selectedProduct = product
        if (changed) conversationContextVersion += 1
        resultsPanel.markSelected(product.id)
        val retailer = product.retailer?.let { " at $it" }.orEmpty()
        statusText.text = "Selected: ${product.title} — ${product.displayPrice()}$retailer. " +
            "Check fit, then View in my space for a real-scale 3D preview."
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
                        if (pendingConversationArProductId == product.id) {
                            pendingConversationArProductId = null
                            beginRealProductPreview()
                        }
                    },
                    onFailure = { exception ->
                        if (pendingConversationArProductId == product.id) {
                            pendingConversationArProductId = null
                        }
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
        val footprint = dimensions.verifiedFootprint()
        val result = FitEngine.evaluate(
            productWidthMeters = footprint?.first,
            productDepthMeters = footprint?.second,
            availableWidthMeters = availableWidthMeters,
            availableDepthMeters = availableDepthMeters,
        )
        latestFitResult = result
        fitPanel.show(result, dimensions)
        statusText.text = when (result.verdict) {
            FitVerdict.FITS -> "Fit check: FITS the measured space."
            FitVerdict.DOES_NOT_FIT -> "Fit check: DOES NOT FIT the measured space."
            FitVerdict.UNKNOWN -> if (dimensions.hasVerifiedTriple()) {
                "Retailer dimensions found; axis order is unresolved until the 3D mesh is generated."
            } else {
                "No trustworthy product footprint dimensions were found."
            }
            FitVerdict.NEEDS_MEASUREMENT -> "Fit check: measure the available width and depth."
        }
    }

    private fun ResolvedDimensions.hasVerifiedTriple(): Boolean =
        status == "verified" && dimensionsMeters.size == 3 && dimensionsMeters.all { it > 0.0 }

    /**
     * Returns two retailer-backed horizontal values when the source establishes
     * them. Their order is irrelevant because FitEngine checks a 90° rotation.
     */
    private fun ResolvedDimensions.verifiedFootprint(): Pair<Double, Double>? {
        if (widthMeters != null && depthMeters != null) return widthMeters to depthMeters
        if (!hasVerifiedTriple()) return null
        val heightIndex = axisMapping?.heightIndex ?: return null
        val horizontal = dimensionsMeters.filterIndexed { index, _ -> index != heightIndex }
        return if (horizontal.size == 2) horizontal[0] to horizontal[1] else null
    }

    private fun onAvailableSpaceMeasured(width: Float?, depth: Float?) {
        val newWidth = width?.toDouble()
        val newDepth = depth?.toDouble()
        if (newWidth != availableWidthMeters || newDepth != availableDepthMeters) {
            conversationContextVersion += 1
        }
        availableWidthMeters = newWidth
        availableDepthMeters = newDepth
        refreshFit()
    }

    private fun enterMeasureSpace() {
        renderer.setMode(InteractionMode.MEASURE_SPACE)
        updateModeUi(InteractionMode.MEASURE_SPACE)
        resultsPanel.visibility = View.GONE
    }

    /** Stable random app-install identity; it contains no account, device, or contact information. */
    private fun stableShopperId(): String {
        val preferences = getSharedPreferences(SHOPPER_PREFERENCES, Context.MODE_PRIVATE)
        preferences.getString(SHOPPER_ID_KEY, null)?.trim()?.takeIf { it.isNotEmpty() }?.let { return it }
        return "shopper_${UUID.randomUUID()}".also { generated ->
            preferences.edit().putString(SHOPPER_ID_KEY, generated).apply()
        }
    }

    private fun beginConversationArPreview(product: ProductCandidate?) {
        if (product == null) {
            statusText.text = "Choose a product before opening it in your space."
            return
        }
        if (selectedProduct?.id != product.id) selectProduct(product)
        pendingConversationArProductId = product.id
        if (fitProduct?.id == product.id && fitDimensions != null) {
            pendingConversationArProductId = null
            beginRealProductPreview()
            return
        }
        statusText.text = "Confirming verified dimensions before opening the 3D preview…"
        beginFitCheck()
    }

    private fun resetArScene() {
        pendingConversationArProductId = null
        arGeneration += 1
        arRequestInFlight = false
        arPreviewClient.cancel()
        renderer.reset()
        fitDimensions?.takeIf(FitPanel::hasVerifiedSize)?.let {
            fitPanel.setArState("AR placement reset. The verified preview is ready to load again.", "View in my space")
        }
    }

    private fun beginRealProductPreview() {
        val product = fitProduct ?: return
        val dimensions = fitDimensions ?: return
        if (arRequestInFlight) return
        if (!FitPanel.hasVerifiedSize(dimensions)) {
            fitPanel.setArState(FitPanel.REAL_SCALE_UNAVAILABLE, null)
            return
        }
        arGeneration += 1
        val generation = arGeneration
        arRequestInFlight = true
        fitPanel.setArState("Generating 3D preview… The first run for a product can take a minute.", "Generating…", enabled = false)
        statusText.text = "Generating 3D preview of ${product.title.take(50)}…"
        arPreviewStartedElapsedRealtime = android.os.SystemClock.elapsedRealtime()
        val started = System.currentTimeMillis()
        arPreviewClient.request(
            product,
            onProgress = { _ ->
                runOnUiThread {
                    if (generation != arGeneration) return@runOnUiThread
                    val seconds = (System.currentTimeMillis() - started) / 1000
                    statusText.text = "Generating 3D preview… ${seconds}s"
                }
            },
        ) { result ->
            runOnUiThread {
                if (generation != arGeneration) return@runOnUiThread
                arRequestInFlight = false
                result.fold(
                    onSuccess = { asset -> showRealProduct(product, asset, System.currentTimeMillis() - started) },
                    onFailure = { e ->
                        val status = (e as? ArPreviewClient.ArPreviewException)?.status
                        failRealProductPreview(
                            e.message ?: "3D preview failed. Please retry.",
                            retryable = status?.retryable ?: (status == null),
                            unavailable = status?.status == "unavailable",
                        )
                    },
                )
            }
        }
    }

    private fun showRealProduct(product: ProductCandidate, asset: LoadedArAsset, elapsedMillis: Long) {
        val scale = asset.status.scale
        if (scale == null) {
            failRealProductPreview(FitPanel.REAL_SCALE_UNAVAILABLE, retryable = false, unavailable = true)
            return
        }
        val b = asset.mesh.bounds()
        val size = floatArrayOf(b[3] - b[0], b[4] - b[1], b[5] - b[2])
        if (!ArPreviewMath.matchesVerified(size, scale)) {
            failRealProductPreview("The 3D preview did not match the verified dimensions, so it is not shown.", retryable = false)
            return
        }
        // Mesh-assisted axis mapping can make W/D known only after reconstruction. Re-run the
        // deterministic footprint check so automatic placement uses the correct 90° orientation.
        val placementFit = FitEngine.evaluate(
            productWidthMeters = scale.widthMeters,
            productDepthMeters = scale.depthMeters,
            availableWidthMeters = availableWidthMeters,
            availableDepthMeters = availableDepthMeters,
        )
        latestFitResult = placementFit
        fitDimensions?.let { fitPanel.show(placementFit, it) }
        renderer.setRealProduct(
            asset.mesh,
            scale,
            product.title,
            rotateToFit = placementFit.rotatedToFit,
        )
        renderer.setMode(InteractionMode.PREVIEW_REAL_PRODUCT)
        updateModeUi(InteractionMode.PREVIEW_REAL_PRODUCT)
        resultsPanel.visibility = View.GONE
        val timing = String.format(
            java.util.Locale.US, "%s in %.1fs",
            if (asset.status.cached) "Cached" else "Generated", elapsedMillis / 1000.0,
        )
        fitPanel.setArState("${asset.status.previewLabel} ($timing)", "View in my space")
        val serverBreakdown = asset.status.timings.entries
            .sortedBy { it.key }
            .joinToString(" ") { (key, value) -> "$key=${String.format(java.util.Locale.US, "%.3f", value)}s" }
        android.util.Log.i(
            "SpatialCommerceM6",
            "asset=${asset.status.assetId} cache=${asset.status.cacheOutcome} cached=${asset.status.cached} " +
                "dimensionPath=${asset.status.dimensionLookupPath} dimensionCache=${asset.status.dimensionLookupCacheHit} " +
                "phonePrepare=${asset.preparationMillis}ms phoneDownload=${asset.downloadMillis}ms " +
                "phoneParse=${asset.parseMillis}ms callbackTotal=${elapsedMillis}ms " +
                "bytes=${asset.status.glbBytes} vertices=${asset.mesh.vertexCount} server=[$serverBreakdown]",
        )
    }

    private fun failRealProductPreview(message: String, retryable: Boolean, unavailable: Boolean = false) {
        arRequestInFlight = false
        // Never fall back to the built-in chair for a selected real product.
        if (currentMode == InteractionMode.PREVIEW_REAL_PRODUCT) {
            renderer.clearRealProduct()
            renderer.setMode(InteractionMode.MEASURE)
            updateModeUi(InteractionMode.MEASURE)
        }
        val text = if (unavailable) message else "3D preview unavailable: $message"
        fitPanel.setArState(text, if (retryable) "Retry 3D preview" else null)
        statusText.text = "3D preview did not complete."
    }

    private fun clearFitCheck() {
        pendingConversationArProductId = null
        arGeneration += 1
        arRequestInFlight = false
        arPreviewClient.cancel()
        renderer.clearRealProduct()
        if (currentMode == InteractionMode.PREVIEW_REAL_PRODUCT) {
            renderer.setMode(InteractionMode.MEASURE)
            updateModeUi(InteractionMode.MEASURE)
        }
        fitGeneration += 1
        fitLookupInFlight = false
        fitProduct = null
        fitDimensions = null
        latestFitResult = null
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
        currentMode = mode
        measureButton.isEnabled = mode != InteractionMode.MEASURE
        previewButton.isEnabled = mode != InteractionMode.PREVIEW_PRODUCT
        measureSpaceButton.isEnabled = mode != InteractionMode.MEASURE_SPACE
        rotationControls.visibility =
            if (mode == InteractionMode.PREVIEW_PRODUCT || mode == InteractionMode.PREVIEW_REAL_PRODUCT) View.VISIBLE else View.GONE
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
        private const val SHOPPER_PREFERENCES = "spatial_commerce_identity"
        private const val SHOPPER_ID_KEY = "pseudonymous_shopper_id"
    }
}

private class TapSurfaceView(context: Context) : GLSurfaceView(context) {
    override fun performClick(): Boolean {
        super.performClick()
        return true
    }
}

enum class SurfaceTouchAction { DOWN, MOVE, UP, CANCEL }

data class SurfaceTouchEvent(val action: SurfaceTouchAction, val x: Float, val y: Float)

enum class InteractionMode {
    MEASURE,
    PREVIEW_PRODUCT,
    MEASURE_SPACE,
    /** Milestone 6: the selected real product, reconstructed and scaled to verified dimensions. */
    PREVIEW_REAL_PRODUCT,
}
