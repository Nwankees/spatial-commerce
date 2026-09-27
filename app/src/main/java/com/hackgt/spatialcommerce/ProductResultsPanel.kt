package com.hackgt.spatialcommerce

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.ColorDrawable
import android.graphics.drawable.GradientDrawable
import android.text.TextUtils
import android.view.Gravity
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import coil3.load
import coil3.request.crossfade
import coil3.request.error
import coil3.request.placeholder

/**
 * Programmatic, scrollable list of real product candidates. It consumes touches
 * so taps on the list never fall through to AR hit testing.
 */
class ProductResultsPanel(
    context: Context,
    private val onProductSelected: (ProductCandidate) -> Unit,
    private val onRetry: () -> Unit,
    private val onCheckFit: () -> Unit,
) : LinearLayout(context) {
    private val headerText: TextView
    private val detailText: TextView
    private val retryButton: Button
    private val checkFitButton: Button
    private val list: LinearLayout
    private val rowsById = mutableMapOf<String, View>()
    private val selectedLabelsById = mutableMapOf<String, TextView>()

    init {
        orientation = VERTICAL
        isClickable = true
        isFocusable = true
        setBackgroundColor(Color.argb(235, 20, 22, 28))
        setPadding(dp(12), dp(8), dp(12), dp(8))
        visibility = View.GONE

        headerText = TextView(context).apply {
            setTextColor(Color.WHITE)
            textSize = 16f
            setTypeface(typeface, Typeface.BOLD)
        }
        val closeButton = Button(context).apply {
            text = "Hide"
            textSize = 13f
            isAllCaps = false
            setOnClickListener { this@ProductResultsPanel.visibility = View.GONE }
        }
        val header = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            addView(headerText, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
            addView(closeButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(44)))
        }
        detailText = TextView(context).apply {
            setTextColor(Color.rgb(190, 196, 208))
            textSize = 13f
        }
        retryButton = Button(context).apply {
            text = "Retry search"
            textSize = 14f
            isAllCaps = false
            visibility = View.GONE
            setOnClickListener { onRetry() }
        }
        checkFitButton = Button(context).apply {
            text = "Check fit"
            textSize = 14f
            isAllCaps = false
            visibility = View.GONE
            setOnClickListener { onCheckFit() }
        }
        list = LinearLayout(context).apply { orientation = VERTICAL }
        val scroll = MaxHeightScrollView(context, maxHeightPx = (resources.displayMetrics.heightPixels * 0.34f).toInt()).apply {
            isFillViewport = false
            addView(list, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        }

        addView(header, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        addView(detailText, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        addView(retryButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(48)))
        addView(checkFitButton, LayoutParams(LayoutParams.WRAP_CONTENT, dp(48)))
        addView(scroll, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
    }

    fun showLoading() {
        clearRows()
        headerText.text = "Finding similar products…"
        detailText.text = "Searching real retailers for the analyzed object."
        retryButton.visibility = View.GONE
        visibility = View.VISIBLE
    }

    fun showError(message: String) {
        clearRows()
        headerText.text = "Product search did not complete"
        detailText.text = message
        retryButton.visibility = View.VISIBLE
        visibility = View.VISIBLE
    }

    fun showResults(result: ProductSearchResult, selectedId: String?) {
        clearRows()
        val sourceLabel = if (result.isCached) "Cached" else "Live"
        headerText.text = if (result.products.isEmpty()) {
            "No similar products found"
        } else {
            "${result.products.size} similar products  •  $sourceLabel"
        }
        val lines = if (result.queries.size > 1) {
            val failed = result.queries.count { it.status == "failed" }
            mutableListOf(
                "${result.queries.size} searches via ${result.provider}" +
                    (if (failed > 0) " ($failed failed)" else "") + ", merged and ranked:",
                result.queries.joinToString("\n") { "• “${it.query}”" },
            )
        } else {
            mutableListOf("Search: “${result.query}”  via ${result.provider}")
        }
        if (result.isCached) {
            lines += "Live search unavailable — saved from an earlier live search" +
                result.cachedAt?.let { " (${it.take(16).replace('T', ' ')} UTC)" }.orEmpty() + "."
        }
        result.message?.takeIf { !result.isCached }?.let { lines += it }
        detailText.text = lines.joinToString("\n")
        retryButton.visibility = if (result.products.isEmpty() || result.isCached) View.VISIBLE else View.GONE
        result.products.forEach { product -> list.addView(buildRow(product)) }
        markSelected(selectedId)
        visibility = View.VISIBLE
    }

    fun markSelected(selectedId: String?) {
        rowsById.forEach { (id, row) ->
            val selected = id == selectedId
            row.background = rowBackground(selected)
            selectedLabelsById[id]?.visibility = if (selected) View.VISIBLE else View.GONE
        }
        checkFitButton.visibility = if (selectedId != null && rowsById.containsKey(selectedId)) View.VISIBLE else View.GONE
    }

    private fun clearRows() {
        checkFitButton.visibility = View.GONE
        list.removeAllViews()
        rowsById.clear()
        selectedLabelsById.clear()
    }

    private fun buildRow(product: ProductCandidate): View {
        val image = ImageView(context).apply {
            scaleType = ImageView.ScaleType.CENTER_CROP
            setBackgroundColor(Color.rgb(48, 52, 62))
            contentDescription = product.title
        }
        val imageUrl = product.imageUrl
        if (imageUrl != null) {
            // A failed image only affects this row; the rest of the list still renders.
            image.load(imageUrl) {
                crossfade(true)
                placeholder(ColorDrawable(Color.rgb(48, 52, 62)))
                error(ColorDrawable(Color.rgb(70, 52, 58)))
            }
        }

        val title = TextView(context).apply {
            text = product.title
            setTextColor(Color.WHITE)
            textSize = 14f
            maxLines = 2
            ellipsize = TextUtils.TruncateAt.END
        }
        val price = TextView(context).apply {
            text = product.displayPrice()
            setTextColor(Color.rgb(140, 230, 170))
            textSize = 15f
            setTypeface(typeface, Typeface.BOLD)
        }
        val retailer = TextView(context).apply {
            text = product.displayRetailerAndRating()
            setTextColor(Color.rgb(190, 196, 208))
            textSize = 12f
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
        }
        val selectedLabel = TextView(context).apply {
            text = "✓ Selected"
            setTextColor(Color.rgb(186, 160, 255))
            textSize = 12f
            setTypeface(typeface, Typeface.BOLD)
            visibility = View.GONE
        }
        val textColumn = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(dp(10), 0, 0, 0)
            addView(title)
            addView(price)
            addView(retailer)
            addView(selectedLabel)
        }
        val row = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(8), dp(8), dp(8), dp(8))
            isClickable = true
            isFocusable = true
            background = rowBackground(false)
            addView(image, LayoutParams(dp(72), dp(72)))
            addView(textColumn, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
            setOnClickListener { onProductSelected(product) }
        }
        rowsById[product.id] = row
        selectedLabelsById[product.id] = selectedLabel
        return LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(0, dp(4), 0, dp(4))
            addView(row, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT))
        }
    }

    private fun rowBackground(selected: Boolean) = GradientDrawable().apply {
        cornerRadius = dp(8).toFloat()
        setColor(if (selected) Color.argb(255, 58, 44, 104) else Color.argb(255, 32, 35, 44))
        if (selected) setStroke(dp(2), Color.rgb(160, 130, 255))
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density).toInt()
}

private class MaxHeightScrollView(context: Context, private val maxHeightPx: Int) : ScrollView(context) {
    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        super.onMeasure(widthMeasureSpec, MeasureSpec.makeMeasureSpec(maxHeightPx, MeasureSpec.AT_MOST))
    }
}
