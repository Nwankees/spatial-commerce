package com.hackgt.spatialcommerce

import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

class FootprintGeometryTest {
    private fun point(x: Double, z: Double, y: Double = 0.0) = FootprintPoint(x, y, z)

    @Test
    fun twoTapsThenDragFromAProduceRectangleAndAlignment() {
        val afterA = FootprintGeometry.tap(FootprintGeometry.reset(), point(1.0, 2.0))
        val afterB = FootprintGeometry.tap(afterA, point(3.0, 2.0))
        val complete = FootprintGeometry.drag(
            afterB,
            FootprintEndpoint.A,
            // The along-edge component (0.4 m) is ignored; perpendicular breadth is 1.0 m.
            point(1.4, 3.0),
        ) as FootprintMeasurementState.Complete

        val rectangle = complete.rectangle
        assertPoint(rectangle.a, 1.0, 0.0, 2.0)
        assertPoint(rectangle.b, 3.0, 0.0, 2.0)
        assertPoint(rectangle.c, 3.0, 0.0, 3.0)
        assertPoint(rectangle.d, 1.0, 0.0, 3.0)
        assertPoint(rectangle.center, 2.0, 0.0, 2.5)
        assertEquals(2.0, rectangle.widthMeters, EPSILON)
        assertEquals(1.0, rectangle.depthMeters, EPSILON)
        assertEquals(0.0, rectangle.alignmentYawDegrees, EPSILON)
    }

    @Test
    fun dragFromBSelectsTheOtherSideAndDerivesSameFourCornerShape() {
        val edge = FootprintEdge(point(0.0, 0.0), point(2.0, 0.0))
        val rectangle = FootprintGeometry.rectangleFromDrag(
            edge,
            FootprintEndpoint.B,
            // A large along-edge component must not skew the rectangle.
            point(0.5, -0.75),
        )!!

        assertPoint(rectangle.a, 0.0, 0.0, 0.0)
        assertPoint(rectangle.b, 2.0, 0.0, 0.0)
        assertPoint(rectangle.c, 2.0, 0.0, -0.75)
        assertPoint(rectangle.d, 0.0, 0.0, -0.75)
        assertPoint(rectangle.center, 1.0, 0.0, -0.375)
        assertEquals(0.75, rectangle.depthMeters, EPSILON)
        assertEquals(FootprintEndpoint.B, rectangle.draggedFrom)
    }

    @Test
    fun alignmentYawMapsLocalWidthAxisToVerticalWorldEdge() {
        val towardNegativeZ = FootprintGeometry.rectangleFromDrag(
            FootprintEdge(point(0.0, 0.0), point(0.0, -2.0)),
            FootprintEndpoint.A,
            point(1.0, 0.0),
        )!!
        val towardPositiveZ = FootprintGeometry.rectangleFromDrag(
            FootprintEdge(point(0.0, 0.0), point(0.0, 2.0)),
            FootprintEndpoint.A,
            point(-1.0, 0.0),
        )!!

        assertEquals(90.0, towardNegativeZ.alignmentYawDegrees, EPSILON)
        assertEquals(-90.0, towardPositiveZ.alignmentYawDegrees, EPSILON)
    }

    @Test
    fun tooShortEdgeOrBreadthKeepsMeasurementAwaitingUsefulInput() {
        val waitingForB = FootprintGeometry.tap(FootprintGeometry.reset(), point(0.0, 0.0))
        val stillWaitingForB = FootprintGeometry.tap(waitingForB, point(0.01, 0.0))
        assertSame(waitingForB, stillWaitingForB)

        val waitingForBreadth = FootprintGeometry.tap(waitingForB, point(1.0, 0.0))
        val stillWaitingForBreadth = FootprintGeometry.drag(
            waitingForBreadth,
            FootprintEndpoint.A,
            point(0.5, 0.01),
        )
        assertSame(waitingForBreadth, stillWaitingForBreadth)
    }

    @Test
    fun translatingToNewProductCenterMovesEveryCornerWithoutChangingGeometry() {
        val original = FootprintGeometry.rectangleFromDrag(
            FootprintEdge(point(0.0, 0.0, 0.02), point(2.0, 0.0, 0.04)),
            FootprintEndpoint.A,
            point(0.0, 1.0, 0.03),
        )!!
        val moved = original.translatedTo(point(10.0, -4.0, 1.2))

        assertPoint(moved.center, 10.0, 1.2, -4.0)
        assertPoint(moved.a, 9.0, 1.2, -4.5)
        assertPoint(moved.c, 11.0, 1.2, -3.5)
        assertEquals(original.widthMeters, moved.widthMeters, EPSILON)
        assertEquals(original.depthMeters, moved.depthMeters, EPSILON)
        assertEquals(original.alignmentYawDegrees, moved.alignmentYawDegrees, EPSILON)
        assertTrue(moved.c.horizontalDistanceTo(moved.d) > 1.99)
    }

    private fun assertPoint(
        actual: FootprintPoint,
        expectedX: Double,
        expectedY: Double,
        expectedZ: Double,
    ) {
        assertEquals(expectedX, actual.x, EPSILON)
        assertEquals(expectedY, actual.y, EPSILON)
        assertEquals(expectedZ, actual.z, EPSILON)
    }

    companion object {
        private const val EPSILON = 1e-9
    }
}
