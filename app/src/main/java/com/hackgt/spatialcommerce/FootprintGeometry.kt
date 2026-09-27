package com.hackgt.spatialcommerce

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot

/** A point in ARCore world space. Rectangle calculations use the horizontal X/Z plane. */
data class FootprintPoint(
    val x: Double,
    val y: Double,
    val z: Double,
) {
    fun translatedBy(dx: Double, dy: Double, dz: Double) =
        FootprintPoint(x + dx, y + dy, z + dz)

    internal fun horizontalDistanceTo(other: FootprintPoint): Double =
        hypot(other.x - x, other.z - z)
}

/** The endpoint where the user began the breadth drag. */
enum class FootprintEndpoint { A, B }

data class FootprintEdge(
    val a: FootprintPoint,
    val b: FootprintPoint,
) {
    val lengthMeters: Double get() = a.horizontalDistanceTo(b)
}

/**
 * A measured horizontal rectangle in perimeter order A -> B -> C -> D.
 *
 * A/B are the two tapped points. C/D are derived from the perpendicular component of the
 * breadth drag. [alignmentYawDegrees] is the OpenGL +Y rotation that aligns model-local +X
 * with A -> B, so a product can be centered and aligned without guessing from the camera.
 */
data class FootprintRectangle(
    val a: FootprintPoint,
    val b: FootprintPoint,
    val c: FootprintPoint,
    val d: FootprintPoint,
    val center: FootprintPoint,
    val widthMeters: Double,
    val depthMeters: Double,
    val alignmentYawDegrees: Double,
    val draggedFrom: FootprintEndpoint,
) {
    val corners: List<FootprintPoint> get() = listOf(a, b, c, d)

    /** Keeps the measured rectangle attached to a product that is repositioned. */
    fun translatedBy(dx: Double, dy: Double, dz: Double) = copy(
        a = a.translatedBy(dx, dy, dz),
        b = b.translatedBy(dx, dy, dz),
        c = c.translatedBy(dx, dy, dz),
        d = d.translatedBy(dx, dy, dz),
        center = center.translatedBy(dx, dy, dz),
    )

    /** Moves the rectangle's center to [newCenter] without changing its size or alignment. */
    fun translatedTo(newCenter: FootprintPoint) = translatedBy(
        dx = newCenter.x - center.x,
        dy = newCenter.y - center.y,
        dz = newCenter.z - center.z,
    )
}

/** Immutable state for the two-taps-then-drag footprint measurement interaction. */
sealed class FootprintMeasurementState {
    object AwaitingFirstPoint : FootprintMeasurementState()
    data class AwaitingSecondPoint(val a: FootprintPoint) : FootprintMeasurementState()
    data class AwaitingBreadth(val edge: FootprintEdge) : FootprintMeasurementState()
    data class Complete(val rectangle: FootprintRectangle) : FootprintMeasurementState()
}

/** Pure footprint geometry; it has no Android, ARCore, or renderer dependency. */
object FootprintGeometry {
    const val DEFAULT_MINIMUM_EDGE_METERS = 0.03
    const val DEFAULT_MINIMUM_BREADTH_METERS = 0.03

    fun reset(): FootprintMeasurementState = FootprintMeasurementState.AwaitingFirstPoint

    /**
     * Records A and then B. Further taps are ignored because breadth must come from a drag.
     * A too-short second edge leaves the state waiting for a useful B point.
     */
    fun tap(
        state: FootprintMeasurementState,
        point: FootprintPoint,
        minimumEdgeMeters: Double = DEFAULT_MINIMUM_EDGE_METERS,
    ): FootprintMeasurementState {
        requirePoint(point)
        require(minimumEdgeMeters > 0.0 && minimumEdgeMeters.isFinite()) {
            "Minimum edge must be a positive finite distance."
        }
        return when (state) {
            FootprintMeasurementState.AwaitingFirstPoint ->
                FootprintMeasurementState.AwaitingSecondPoint(point)

            is FootprintMeasurementState.AwaitingSecondPoint -> {
                if (state.a.horizontalDistanceTo(point) < minimumEdgeMeters) state
                else FootprintMeasurementState.AwaitingBreadth(FootprintEdge(state.a, point))
            }

            is FootprintMeasurementState.AwaitingBreadth,
            is FootprintMeasurementState.Complete,
            -> state
        }
    }

    /** Completes an edge-ready measurement, or keeps waiting when the drag is too shallow. */
    fun drag(
        state: FootprintMeasurementState,
        from: FootprintEndpoint,
        current: FootprintPoint,
        minimumBreadthMeters: Double = DEFAULT_MINIMUM_BREADTH_METERS,
    ): FootprintMeasurementState {
        val edgeState = state as? FootprintMeasurementState.AwaitingBreadth ?: return state
        val rectangle = rectangleFromDrag(edgeState.edge, from, current, minimumBreadthMeters)
            ?: return state
        return FootprintMeasurementState.Complete(rectangle)
    }

    /**
     * Builds a rectangle from the perpendicular component of the drag.
     *
     * Along-edge finger movement is deliberately discarded. This makes an approximately
     * perpendicular gesture produce an exact rectangle and lets either A or B drive the same
     * breadth/side calculation. Returns null until the perpendicular breadth is large enough.
     */
    fun rectangleFromDrag(
        edge: FootprintEdge,
        from: FootprintEndpoint,
        current: FootprintPoint,
        minimumBreadthMeters: Double = DEFAULT_MINIMUM_BREADTH_METERS,
    ): FootprintRectangle? {
        requirePoint(edge.a)
        requirePoint(edge.b)
        requirePoint(current)
        require(minimumBreadthMeters > 0.0 && minimumBreadthMeters.isFinite()) {
            "Minimum breadth must be a positive finite distance."
        }

        val width = edge.lengthMeters
        if (width < DEFAULT_MINIMUM_EDGE_METERS) return null

        val unitX = (edge.b.x - edge.a.x) / width
        val unitZ = (edge.b.z - edge.a.z) / width
        // A stable perpendicular on the X/Z plane. The drag's sign selects which side of A-B.
        val perpendicularX = -unitZ
        val perpendicularZ = unitX
        val origin = if (from == FootprintEndpoint.A) edge.a else edge.b
        val signedBreadth =
            (current.x - origin.x) * perpendicularX +
                (current.z - origin.z) * perpendicularZ
        val depth = abs(signedBreadth)
        if (depth < minimumBreadthMeters) return null

        val offsetX = perpendicularX * signedBreadth
        val offsetZ = perpendicularZ * signedBreadth
        // All four corners should rest on one horizontal surface despite tiny hit-test Y noise.
        val surfaceY = (edge.a.y + edge.b.y + current.y) / 3.0
        val a = FootprintPoint(edge.a.x, surfaceY, edge.a.z)
        val b = FootprintPoint(edge.b.x, surfaceY, edge.b.z)
        val c = FootprintPoint(edge.b.x + offsetX, surfaceY, edge.b.z + offsetZ)
        val d = FootprintPoint(edge.a.x + offsetX, surfaceY, edge.a.z + offsetZ)
        val center = FootprintPoint(
            x = (a.x + c.x) / 2.0,
            y = surfaceY,
            z = (a.z + c.z) / 2.0,
        )
        // Android/OpenGL +Y rotation maps local +X toward -Z for positive angles.
        val yawDegrees = Math.toDegrees(atan2(-unitZ, unitX))

        return FootprintRectangle(
            a = a,
            b = b,
            c = c,
            d = d,
            center = center,
            widthMeters = width,
            depthMeters = depth,
            alignmentYawDegrees = yawDegrees,
            draggedFrom = from,
        )
    }

    private fun requirePoint(point: FootprintPoint) {
        require(point.x.isFinite() && point.y.isFinite() && point.z.isFinite()) {
            "Footprint points must contain finite coordinates."
        }
    }
}
