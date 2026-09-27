package com.hackgt.spatialcommerce

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import java.util.Locale
import kotlin.math.abs

/** Compact fit-check card. Consumes touches so they never reach AR hit testing. */
class FitPanel(
    context: Context,
    private val onMeasureSpace: () -> Unit,
    private val onRetry: () -> Unit,
    private val onViewInSpace: () -> Unit = {},
    private val onViewSource: (String) -> Unit = {},
) : LinearLayout(context) {
    private val verdictText: TextView
    private val detailText: TextView
    private val arText: TextView
    private val arButton: Button
    private val measureButton: Button
    private val retryButton: Button
    private val sourceButton: Button
    private var sourceUrl: String? = null

    init {
        orientation = VERTICAL
        isClickable = true
        isFocusable = true
        setBackgroundColor(Color.argb(240, 16, 18, 24))
        setPadding(dp(12), dp(8), dp(12), dp(8))
        visibility = View.GONE

        verdictText = TextView(context).apply {
            textSize = 18f
            setTypeface(typeface, Typeface.BOLD)
        }
        detailText = TextView(context).apply {
            setTextColor(Color.rgb(214, 218, 228))
            textSize = 13f
        }
        arText = TextView(context).apply {
            setTextColor(Color.rgb(170, 205, 255))
            textSize = 13f
            visibility = View.GONE
        }
        arButton = smallButton("View in my space") { onViewInSpace() }.apply { visibility = View.GONE }
        measureButton = smallButton("Measure space") { onMeasureSpace() }
        retryButton = smallButton("Retry") { onRetry() }
        sourceButton = smallButton("View source") { sourceUrl?.let(onViewSource) }.apply {
            visibility = View.GONE
        }
        val closeButton = smallButton("Close") { visibility = View.GONE }
        val buttons = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.END
            addView(arButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(44)))
            addView(measureButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(44)))
            addView(retryButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(44)))
            addView(sourceButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(44)))
            addView(closeButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(44)))
        }
        addView(verdictText, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        addView(detailText, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        addView(arText, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        addView(buttons, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
    }

    fun showLoading(product: ProductCandidate) {
        setVerdict("Checking fit…", Color.WHITE)
        detailText.text = "Looking up explicit dimensions for\n${product.title}"
        measureButton.visibility = View.GONE
        retryButton.visibility = View.GONE
        sourceUrl = null
        sourceButton.visibility = View.GONE
        setArState(null, null)
        visibility = View.VISIBLE
    }

    /** Milestone 6 line + button. [buttonLabel] null hides the button. */
    fun setArState(text: String?, buttonLabel: String?, enabled: Boolean = true) {
        arText.text = text.orEmpty()
        arText.visibility = if (text == null) View.GONE else View.VISIBLE
        arButton.text = buttonLabel.orEmpty()
        arButton.isEnabled = enabled
        arButton.visibility = if (buttonLabel == null) View.GONE else View.VISIBLE
    }

    fun showError(message: String) {
        setVerdict("Fit check did not complete", Color.rgb(255, 196, 120))
        detailText.text = message
        measureButton.visibility = View.GONE
        retryButton.visibility = View.VISIBLE
        sourceUrl = null
        sourceButton.visibility = View.GONE
        setArState(null, null)
        visibility = View.VISIBLE
    }

    fun show(result: FitResult, dimensions: ResolvedDimensions) {
        val lines = mutableListOf<String>()
        when (result.verdict) {
            FitVerdict.FITS -> setVerdict("FITS", Color.rgb(120, 230, 150))
            FitVerdict.DOES_NOT_FIT -> setVerdict("DOES NOT FIT", Color.rgb(255, 110, 110))
            FitVerdict.UNKNOWN -> setVerdict(
                if (hasVerifiedTriple(dimensions)) {
                    "DIMENSIONS FOUND — AXIS ORDER UNRESOLVED"
                } else {
                    "NO TRUSTWORTHY PRODUCT DIMENSIONS"
                },
                Color.rgb(255, 196, 120),
            )
            FitVerdict.NEEDS_MEASUREMENT -> setVerdict("NEEDS MEASUREMENT", Color.rgb(140, 190, 255))
        }
        if (result.productWidthMeters != null && result.productDepthMeters != null) {
            lines += "Product footprint: ${axis(result.productWidthMeters, "A")} × ${axis(result.productDepthMeters, "B")}" +
                (dimensions.heightMeters?.let { " · ${meters(it)} H" } ?: "")
        } else if (hasVerifiedTriple(dimensions)) {
            lines += "Verified measurements: ${dimensions.dimensionsMeters.joinToString(" × ") { meters(it) }}"
        } else {
            lines += "Product footprint: unavailable"
        }
        lines += "Available: ${axis(result.availableWidthMeters, "W")} × ${axis(result.availableDepthMeters, "D")}"
        if (result.widthRemainingMeters != null && result.depthRemainingMeters != null) {
            lines += "Width: ${remaining(result.widthRemainingMeters)}   Depth: ${remaining(result.depthRemainingMeters)}"
        }
        when (result.verdict) {
            FitVerdict.UNKNOWN -> {
                lines += if (hasVerifiedTriple(dimensions)) {
                    "All three retailer values are verified. The 3D mesh will only map those values to axes if its proportions agree."
                } else {
                    "No measurements are estimated."
                }
                dimensions.message?.let { lines += it }
            }
            FitVerdict.NEEDS_MEASUREMENT ->
                lines += "Tap Measure space: tap A and B for one edge, then drag from either endpoint to set the footprint breadth."
            FitVerdict.FITS ->
                if (result.rotatedToFit) lines += "Fits when the product footprint is rotated 90°."
            FitVerdict.DOES_NOT_FIT ->
                if (result.rotatedToFit) lines += "The closest orientation is rotated 90°, but it still does not fit."
        }
        if (dimensions.status != "unavailable") {
            val source = listOfNotNull(dimensions.sourceName, sourceLabel(dimensions.sourceType)).joinToString(" · ")
            val trust = if (dimensions.axisMapping?.source == "none") {
                "Verified retailer measurements; axis order unresolved"
            } else {
                "Verified retailer dimensions"
            }
            lines += "$trust · $source"
            dimensions.sourcePath?.let { lines += "Field: $it" }
            dimensions.rawDimensions?.let { lines += "“${it.take(120)}”" }
        }
        detailText.text = lines.joinToString("\n")
        measureButton.visibility = if (result.verdict == FitVerdict.UNKNOWN) View.GONE else View.VISIBLE
        retryButton.visibility = if (result.verdict == FitVerdict.UNKNOWN && dimensions.retryable) View.VISIBLE else View.GONE
        sourceUrl = dimensions.sourceUrl
        sourceButton.visibility = if (sourceUrl != null) View.VISIBLE else View.GONE
        if (hasVerifiedSize(dimensions)) {
            setArState(null, "View in my space")
        } else {
            setArState(REAL_SCALE_UNAVAILABLE, null)
        }
        visibility = View.VISIBLE
    }

    private fun setVerdict(text: String, color: Int) {
        verdictText.text = text
        verdictText.setTextColor(color)
    }

    private fun smallButton(label: String, onClick: () -> Unit) = Button(context).apply {
        text = label
        textSize = 13f
        isAllCaps = false
        setOnClickListener { onClick() }
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()

    companion object {
        const val REAL_SCALE_UNAVAILABLE =
            "Real-scale preview unavailable — three trustworthy product measurements could not be verified."

        /** M6 may map a verified unlabeled triple to axes from mesh proportions server-side. */
        fun hasVerifiedSize(d: ResolvedDimensions) =
            ((d.widthMeters ?: 0.0) > 0.0 && (d.depthMeters ?: 0.0) > 0.0 && (d.heightMeters ?: 0.0) > 0.0) ||
                hasVerifiedTriple(d)

        private fun hasVerifiedTriple(d: ResolvedDimensions) =
            d.status == "verified" && d.dimensionsMeters.size == 3 && d.dimensionsMeters.all { it > 0.0 }

        fun meters(value: Double) = String.format(Locale.US, "%.2f m", value)

        private fun axis(value: Double?, label: String) = value?.let { "${meters(it)} $label" } ?: "? $label"

        private fun remaining(value: Double): String = if (value >= 0.0) {
            String.format(Locale.US, "%.2f m to spare", value)
        } else {
            String.format(Locale.US, "%.2f m too large", abs(value))
        }

        private fun sourceLabel(sourceType: String) = when (sourceType) {
            "json_ld" -> "JSON-LD"
            "structured_metadata" -> "structured metadata"
            "spec_table" -> "spec table"
            "page_text_llm" -> "retailer specifications"
            "page_text" -> "page text"
            else -> null
        }
    }
}
