package com.hackgt.spatialcommerce

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ArPreviewMathTest {
    private fun scale(sx: Float, sy: Float, sz: Float, swapped: Boolean = false) =
        ArScale(0.60, 0.50, 0.90, sx, sy, sz, swapped, 0.1)

    @Test
    fun scaledExtentsMatchVerifiedDimensions() {
        // Normalized mesh 1.2 (x) × 1.8 (y) × 1.0 (z) → 0.60 W × 0.90 H × 0.50 D.
        val size = floatArrayOf(1.2f, 1.8f, 1.0f)
        val s = scale(0.5f, 0.5f, 0.5f)
        val (w, h, d) = ArPreviewMath.scaledExtents(size, s)
        assertEquals(0.60, w, 1e-6)
        assertEquals(0.90, h, 1e-6)
        assertEquals(0.50, d, 1e-6)
        assertTrue(ArPreviewMath.matchesVerified(size, s))
    }

    @Test
    fun swappedAxesMapMeshXToDepth() {
        // Mesh long axis is z; server scales x to depth and z to width, then the model is yawed 90°.
        val size = floatArrayOf(1.0f, 1.8f, 1.2f)
        val s = scale(0.5f, 0.5f, 0.5f, swapped = true)
        val (w, _, d) = ArPreviewMath.scaledExtents(size, s)
        assertEquals(0.60, w, 1e-6)
        assertEquals(0.50, d, 1e-6)
        assertTrue(ArPreviewMath.matchesVerified(size, s))
    }

    @Test
    fun mismatchedScaleIsRejected() {
        val size = floatArrayOf(1.2f, 1.8f, 1.0f)
        assertFalse(ArPreviewMath.matchesVerified(size, scale(0.6f, 0.5f, 0.5f)))
        assertFalse(ArPreviewMath.matchesVerified(size, scale(0.5f, 0.4f, 0.5f)))
    }

    @Test
    fun footprintPlacementAddsNinetyDegreesOnlyWhenFitRequiresIt() {
        assertEquals(32f, ArPreviewMath.footprintPlacementYaw(32.0, rotateToFit = false), 1e-6f)
        assertEquals(122f, ArPreviewMath.footprintPlacementYaw(32.0, rotateToFit = true), 1e-6f)
    }
}
