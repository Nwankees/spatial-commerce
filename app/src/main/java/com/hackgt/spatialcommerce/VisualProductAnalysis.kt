package com.hackgt.spatialcommerce

data class VisualProductAnalysis(
    val objectDetected: Boolean,
    val category: String?,
    val subcategory: String?,
    val color: String?,
    val materials: List<String>,
    val style: List<String>,
    val shape: String?,
    val searchKeywords: List<String>,
    val confidence: Double,
    val message: String?,
) {
    fun overlayText(): String {
        if (!objectDetected) {
            return message ?: "No obvious product detected. Point at one object and retry."
        }

        val productType = listOfNotNull(category, subcategory)
            .distinct()
            .joinToString(" • ")
            .ifBlank { "Product" }
        val lines = mutableListOf(productType)
        color?.let { lines += "Color: $it" }
        if (materials.isNotEmpty()) lines += "Materials: ${materials.joinToString()}"
        if (style.isNotEmpty()) lines += "Style: ${style.joinToString()}"
        if (searchKeywords.isNotEmpty()) {
            lines += "Search: ${searchKeywords.joinToString("  •  ")}"
        }
        lines += "Confidence: ${(confidence * 100).toInt()}%"
        return lines.joinToString("\n")
    }
}
