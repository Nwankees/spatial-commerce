package com.hackgt.spatialcommerce

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Color
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
    private lateinit var rotationControls: LinearLayout
    private val previewProduct = PreviewProducts.lighthouseLoungeChair

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
        val resetButton = modeButton("Reset")
        val rotateLeftButton = modeButton("Rotate -15°")
        val rotateRightButton = modeButton("Rotate +15°")

        measureButton.setOnClickListener {
            renderer.setMode(InteractionMode.MEASURE)
            updateModeUi(InteractionMode.MEASURE)
        }
        previewButton.setOnClickListener {
            renderer.setMode(InteractionMode.PREVIEW_PRODUCT)
            updateModeUi(InteractionMode.PREVIEW_PRODUCT)
        }
        resetButton.setOnClickListener { renderer.reset() }
        rotateLeftButton.setOnClickListener { renderer.rotateProduct(-15) }
        rotateRightButton.setOnClickListener { renderer.rotateProduct(15) }

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
            addView(resetButton, weightedButtonParams())
        }

        val bottomPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.argb(170, 17, 19, 24))
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

    private fun updateModeUi(mode: InteractionMode) {
        measureButton.isEnabled = mode != InteractionMode.MEASURE
        previewButton.isEnabled = mode != InteractionMode.PREVIEW_PRODUCT
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
}
