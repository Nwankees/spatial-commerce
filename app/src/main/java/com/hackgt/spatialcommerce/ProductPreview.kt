package com.hackgt.spatialcommerce

import java.util.Locale

data class ProductPreview(
    val id: String,
    val name: String,
    val modelAsset: String,
    val widthMeters: Float,
    val depthMeters: Float,
    val heightMeters: Float,
) {
    init {
        require(id.isNotBlank())
        require(name.isNotBlank())
        require(modelAsset.isNotBlank())
        require(widthMeters > 0f && depthMeters > 0f && heightMeters > 0f)
    }

    fun overlayText(): String {
        val inchesPerMeter = 39.3701f
        return String.format(
            Locale.US,
            "%s\n%.2f × %.2f × %.2f m  (W × D × H)\n%.1f × %.1f × %.1f in",
            name,
            widthMeters,
            depthMeters,
            heightMeters,
            widthMeters * inchesPerMeter,
            depthMeters * inchesPerMeter,
            heightMeters * inchesPerMeter,
        )
    }
}

object PreviewProducts {
    val lighthouseLoungeChair = ProductPreview(
        id = "lighthouse-lounge-chair",
        name = "Lighthouse Lounge Chair",
        modelAsset = "primitive://lighthouse-lounge-chair-v1",
        widthMeters = 0.68f,
        depthMeters = 0.74f,
        heightMeters = 0.84f,
    )
}
