package com.hackgt.spatialcommerce

/** An uncertain identification (e.g. brand) with the vision model's confidence. */
data class Hypothesis(
    val value: String,
    val confidence: Double,
    /** "visible_text", "logo", or "design_resemblance". */
    val evidence: String,
)

data class VisualProductAnalysis(
    val objectDetected: Boolean,
    val category: String?,
    val subcategory: String?,
    val brand: Hypothesis?,
    val modelFamily: Hypothesis?,
    val visibleText: List<String>,
    val color: String?,
    val materials: List<String>,
    val style: List<String>,
    val shape: String?,
    val distinctiveFeatures: List<String>,
    val searchQueries: List<String>,
    val confidence: Double,
    val message: String?,
    /**
     * The backend's validated analysis JSON, sent back unchanged for product search so
     * no field is lost or re-interpreted on the device.
     */
    val rawJson: String,
) {
    fun overlayText(): String {
        if (!objectDetected) {
            return message ?: "No obvious product detected. Point at one object and retry."
        }

        val productType = listOfNotNull(subcategory ?: category)
            .joinToString()
            .ifBlank { "Product" }
        val lines = mutableListOf(productType)
        brand?.let { lines += "Likely brand: ${it.value} (${percent(it.confidence)}, ${evidenceLabel(it.evidence)})" }
        modelFamily?.let { lines += "Possible line: ${it.value} (${percent(it.confidence)})" }
        val look = listOfNotNull(color, materials.takeIf { it.isNotEmpty() }?.joinToString()).joinToString(" • ")
        if (look.isNotBlank()) lines += look
        if (distinctiveFeatures.isNotEmpty()) lines += "Features: ${distinctiveFeatures.take(3).joinToString(", ")}"
        if (searchQueries.isNotEmpty()) lines += "Searches: ${searchQueries.take(3).joinToString("  •  ")}"
        lines += "Confidence: ${percent(confidence)}"
        return lines.joinToString("\n")
    }

    private fun percent(value: Double) = "${(value * 100).toInt()}%"

    private fun evidenceLabel(evidence: String) = when (evidence) {
        "visible_text" -> "printed"
        "logo" -> "logo"
        else -> "by design"
    }
}
