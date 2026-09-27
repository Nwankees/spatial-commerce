package com.hackgt.spatialcommerce

import android.opengl.Matrix

/** Server-computed scale of a normalized asset to M5 verified meters (see backend ArScaleModel). */
data class ArScale(
    val widthMeters: Double,
    val depthMeters: Double,
    val heightMeters: Double,
    val scaleX: Float,
    val scaleY: Float,
    val scaleZ: Float,
    val axesSwapped: Boolean,
    val maxAxisDistortion: Double,
)

object ArPreviewMath {
    /** Aligns product width to the measured A-B edge, adding the deterministic 90° fit choice. */
    fun footprintPlacementYaw(alignmentYawDegrees: Double, rotateToFit: Boolean): Float =
        (alignmentYawDegrees + if (rotateToFit) 90.0 else 0.0).toFloat()

    /**
     * Physical extents (width, height, depth in meters) the asset will occupy once scaled,
     * given the normalized mesh extents. Used to double-check the server's scale factors
     * against the verified dimensions before showing a "real-scale" preview.
     */
    fun scaledExtents(meshSize: FloatArray, scale: ArScale): Triple<Double, Double, Double> {
        val x = meshSize[0] * scale.scaleX.toDouble()
        val y = meshSize[1] * scale.scaleY.toDouble()
        val z = meshSize[2] * scale.scaleZ.toDouble()
        return if (scale.axesSwapped) Triple(z, y, x) else Triple(x, y, z)
    }

    /** True when the scaled asset matches the verified W/H/D within [tolerance] (fraction). */
    fun matchesVerified(meshSize: FloatArray, scale: ArScale, tolerance: Double = 0.02): Boolean {
        val (w, h, d) = scaledExtents(meshSize, scale)
        fun close(a: Double, b: Double) = b > 0 && kotlin.math.abs(a - b) / b <= tolerance
        return close(w, scale.widthMeters) && close(h, scale.heightMeters) && close(d, scale.depthMeters)
    }

    /** anchor * yaw(user) * yaw(90° if swapped) * scale — writes into [out] (column-major). */
    fun modelMatrix(out: FloatArray, anchorMatrix: FloatArray, userYawDegrees: Float, scale: ArScale) {
        System.arraycopy(anchorMatrix, 0, out, 0, 16)
        Matrix.rotateM(out, 0, userYawDegrees + if (scale.axesSwapped) 90f else 0f, 0f, 1f, 0f)
        Matrix.scaleM(out, 0, scale.scaleX, scale.scaleY, scale.scaleZ)
    }
}
