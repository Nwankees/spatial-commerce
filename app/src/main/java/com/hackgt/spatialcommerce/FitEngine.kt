package com.hackgt.spatialcommerce

enum class FitVerdict {
    FITS,
    DOES_NOT_FIT,
    /** Required product dimensions (width and depth) are not explicitly known. */
    UNKNOWN,
    /** Available width and/or depth has not been measured yet. */
    NEEDS_MEASUREMENT,
}

data class FitResult(
    val verdict: FitVerdict,
    val productWidthMeters: Double?,
    val productDepthMeters: Double?,
    val availableWidthMeters: Double?,
    val availableDepthMeters: Double?,
    /** available - product - clearance; negative means over by that much. Null when not computable. */
    val widthRemainingMeters: Double?,
    val depthRemainingMeters: Double?,
    val clearanceMeters: Double,
    /** True when swapping the two horizontal product axes is the fitting orientation. */
    val rotatedToFit: Boolean,
)

/**
 * Deterministic footprint check. No estimation, no defaults for missing values.
 *
 * A footprint may rotate on the floor, so both A×B and B×A are checked. Milestone 5
 * uses clearance = 0 (exact footprint comparison).
 */
object FitEngine {
    fun evaluate(
        productWidthMeters: Double?,
        productDepthMeters: Double?,
        availableWidthMeters: Double?,
        availableDepthMeters: Double?,
        clearanceMeters: Double = 0.0,
    ): FitResult {
        require(clearanceMeters >= 0.0) { "Clearance cannot be negative." }
        val verdict: FitVerdict
        var widthRemaining: Double? = null
        var depthRemaining: Double? = null
        var rotated = false
        if (productWidthMeters == null || productDepthMeters == null) {
            verdict = FitVerdict.UNKNOWN
        } else if (availableWidthMeters == null || availableDepthMeters == null) {
            verdict = FitVerdict.NEEDS_MEASUREMENT
        } else {
            val directWidth = availableWidthMeters - productWidthMeters - clearanceMeters
            val directDepth = availableDepthMeters - productDepthMeters - clearanceMeters
            val rotatedWidth = availableWidthMeters - productDepthMeters - clearanceMeters
            val rotatedDepth = availableDepthMeters - productWidthMeters - clearanceMeters
            val directFits = directWidth >= 0.0 && directDepth >= 0.0
            val rotatedFits = rotatedWidth >= 0.0 && rotatedDepth >= 0.0
            rotated = !directFits && rotatedFits
            if (!directFits && !rotatedFits) {
                // Report the less-bad orientation so the shortfall is useful.
                rotated = minOf(rotatedWidth, rotatedDepth) > minOf(directWidth, directDepth)
            }
            widthRemaining = if (rotated) rotatedWidth else directWidth
            depthRemaining = if (rotated) rotatedDepth else directDepth
            verdict = if (directFits || rotatedFits) FitVerdict.FITS else FitVerdict.DOES_NOT_FIT
        }
        return FitResult(
            verdict = verdict,
            productWidthMeters = productWidthMeters,
            productDepthMeters = productDepthMeters,
            availableWidthMeters = availableWidthMeters,
            availableDepthMeters = availableDepthMeters,
            widthRemainingMeters = widthRemaining,
            depthRemainingMeters = depthRemaining,
            clearanceMeters = clearanceMeters,
            rotatedToFit = rotated,
        )
    }
}
