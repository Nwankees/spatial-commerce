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
)

/**
 * Deterministic footprint check. No estimation, no defaults for missing values.
 *
 * fits  <=>  productWidth + clearance <= availableWidth  AND  productDepth + clearance <= availableDepth
 *
 * Milestone 5 uses clearance = 0 (exact footprint comparison). The product is
 * compared in its stated orientation only (width against width, depth against depth).
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
        if (productWidthMeters == null || productDepthMeters == null) {
            verdict = FitVerdict.UNKNOWN
        } else if (availableWidthMeters == null || availableDepthMeters == null) {
            verdict = FitVerdict.NEEDS_MEASUREMENT
        } else {
            widthRemaining = availableWidthMeters - productWidthMeters - clearanceMeters
            depthRemaining = availableDepthMeters - productDepthMeters - clearanceMeters
            verdict = if (widthRemaining >= 0.0 && depthRemaining >= 0.0) FitVerdict.FITS else FitVerdict.DOES_NOT_FIT
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
        )
    }
}
