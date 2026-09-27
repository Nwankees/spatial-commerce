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
) : LinearLayout(context) {
    private val verdictText: TextView
    private val detailText: TextView
    private val measureButton: Button
    private val retryButton: Button

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
        measureButton = smallButton("Measure space") { onMeasureSpace() }
        retryButton = smallButton("Retry") { onRetry() }
        val closeButton = smallButton("Close") { visibility = View.GONE }
        val buttons = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.END
            addView(measureButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(44)))
            addView(retryButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(44)))
            addView(closeButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(44)))
        }
        addView(verdictText, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        addView(detailText, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        addView(buttons, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
    }

    fun showLoading(product: ProductCandidate) {
        setVerdict("Checking fit…", Color.WHITE)
        detailText.text = "Looking up explicit dimensions for\n${product.title}"
        measureButton.visibility = View.GONE
        retryButton.visibility = View.GONE
        visibility = View.VISIBLE
    }

    fun showError(message: String) {
        setVerdict("Fit check did not complete", Color.rgb(255, 196, 120))
        detailText.text = message
        measureButton.visibility = View.GONE
        retryButton.visibility = View.VISIBLE
        visibility = View.VISIBLE
    }

    fun show(result: FitResult, dimensions: ResolvedDimensions) {
        val lines = mutableListOf<String>()
        when (result.verdict) {
            FitVerdict.FITS -> setVerdict("FITS", Color.rgb(120, 230, 150))
            FitVerdict.DOES_NOT_FIT -> setVerdict("DOES NOT FIT", Color.rgb(255, 110, 110))
            FitVerdict.UNKNOWN -> setVerdict("UNKNOWN — dimensions unavailable", Color.rgb(255, 196, 120))
            FitVerdict.NEEDS_MEASUREMENT -> setVerdict("NEEDS MEASUREMENT", Color.rgb(140, 190, 255))
        }
        lines += "Product: ${axis(result.productWidthMeters, "W")} × ${axis(result.productDepthMeters, "D")}" +
            (dimensions.heightMeters?.let { " × ${meters(it)} H" } ?: "")
        lines += "Available: ${axis(result.availableWidthMeters, "W")} × ${axis(result.availableDepthMeters, "D")}"
        if (result.widthRemainingMeters != null && result.depthRemainingMeters != null) {
            lines += "Width: ${remaining(result.widthRemainingMeters)}   Depth: ${remaining(result.depthRemainingMeters)}"
        }
        when (result.verdict) {
            FitVerdict.UNKNOWN -> {
                lines += "Width and depth must both be stated explicitly by a source; nothing is estimated."
                dimensions.message?.let { lines += it }
            }
            FitVerdict.NEEDS_MEASUREMENT ->
                lines += "Tap Measure space, then measure the available width and depth with two points each."
            else -> Unit
        }
        if (dimensions.status != "unavailable") {
            val source = listOfNotNull(dimensions.sourceName, sourceLabel(dimensions.sourceType)).joinToString(" · ")
            lines += "Source: $source · ${dimensions.status}"
            dimensions.rawDimensions?.let { lines += "“${it.take(120)}”" }
        }
        detailText.text = lines.joinToString("\n")
        measureButton.visibility = if (result.verdict == FitVerdict.UNKNOWN) View.GONE else View.VISIBLE
        retryButton.visibility = if (result.verdict == FitVerdict.UNKNOWN && dimensions.retryable) View.VISIBLE else View.GONE
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
            "page_text" -> "page text"
            else -> null
        }
    }
}
