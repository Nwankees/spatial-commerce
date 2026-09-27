package com.hackgt.spatialcommerce

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FitEngineTest {
    @Test
    fun fitsWhenBothAxesFit() {
        val result = FitEngine.evaluate(0.68, 0.74, 0.80, 0.90)
        assertEquals(FitVerdict.FITS, result.verdict)
        assertEquals(0.12, result.widthRemainingMeters!!, 1e-9)
        assertEquals(0.16, result.depthRemainingMeters!!, 1e-9)
    }

    @Test
    fun exactFitCounts() {
        assertEquals(FitVerdict.FITS, FitEngine.evaluate(0.5, 0.5, 0.5, 0.5).verdict)
    }

    @Test
    fun doesNotFitKeepsNegativeRemaining() {
        val result = FitEngine.evaluate(0.97, 0.74, 0.80, 0.90)
        assertEquals(FitVerdict.DOES_NOT_FIT, result.verdict)
        assertEquals(-0.17, result.widthRemainingMeters!!, 1e-9)
        assertEquals(0.16, result.depthRemainingMeters!!, 1e-9)
    }

    @Test
    fun missingProductDimensionIsUnknownEvenWithMeasurements() {
        val result = FitEngine.evaluate(0.97, null, 0.80, 0.90)
        assertEquals(FitVerdict.UNKNOWN, result.verdict)
        assertNull(result.widthRemainingMeters)
        assertEquals(FitVerdict.UNKNOWN, FitEngine.evaluate(null, null, null, null).verdict)
    }

    @Test
    fun incompleteMeasurementNeedsMeasurement() {
        assertEquals(FitVerdict.NEEDS_MEASUREMENT, FitEngine.evaluate(0.5, 0.5, 0.8, null).verdict)
        assertEquals(FitVerdict.NEEDS_MEASUREMENT, FitEngine.evaluate(0.5, 0.5, null, null).verdict)
    }

    @Test
    fun clearanceIsAppliedWhenConfigured() {
        assertEquals(FitVerdict.DOES_NOT_FIT, FitEngine.evaluate(0.70, 0.70, 0.75, 0.90, clearanceMeters = 0.1).verdict)
    }
}
