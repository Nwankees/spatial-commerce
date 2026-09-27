package com.hackgt.spatialcommerce

import android.opengl.GLES20
import android.opengl.Matrix
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * Draws a measured footprint directly from world-space points.
 *
 * Call [createOnGlThread] once when the GL surface is created. Corners supplied to
 * [drawFootprint] must follow the rectangle perimeter in A, B, C, D order.
 */
class FootprintOverlayRenderer {
    private var program = 0
    private var positionAttribute = 0
    private var mvpUniform = 0
    private var colorUniform = 0

    private val vertices: FloatBuffer = ByteBuffer.allocateDirect(FLOATS_PER_FOOTPRINT * FLOAT_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
    private val viewProjectionMatrix = FloatArray(16)

    private val savedDepthWriteMask = BooleanArray(1)
    private val savedBlendSourceRgb = IntArray(1)
    private val savedBlendDestinationRgb = IntArray(1)
    private val savedBlendSourceAlpha = IntArray(1)
    private val savedBlendDestinationAlpha = IntArray(1)
    private val savedLineWidth = FloatArray(1)

    fun createOnGlThread() {
        program = createProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        positionAttribute = GLES20.glGetAttribLocation(program, "aPosition")
        mvpUniform = GLES20.glGetUniformLocation(program, "uMvp")
        colorUniform = GLES20.glGetUniformLocation(program, "uColor")
    }

    /**
     * Draws a translucent interior when requested and always draws all four edges.
     * Each corner is a world-space `[x, y, z]` point.
     */
    fun drawFootprint(
        cornerA: FloatArray,
        cornerB: FloatArray,
        cornerC: FloatArray,
        cornerD: FloatArray,
        viewMatrix: FloatArray,
        projectionMatrix: FloatArray,
        showFill: Boolean,
    ) {
        requireWorldPoint(cornerA)
        requireWorldPoint(cornerB)
        requireWorldPoint(cornerC)
        requireWorldPoint(cornerD)

        vertices.clear()
        putPoint(cornerA)
        putPoint(cornerB)
        putPoint(cornerC)
        putPoint(cornerD)
        vertices.position(0)

        prepareProgram(viewMatrix, projectionMatrix)
        withOverlayState {
            if (showFill) {
                setColor(FILL_COLOR)
                vertices.position(0)
                GLES20.glVertexAttribPointer(positionAttribute, 3, GLES20.GL_FLOAT, false, 0, vertices)
                GLES20.glDrawArrays(GLES20.GL_TRIANGLE_FAN, 0, FOOTPRINT_VERTEX_COUNT)
            }

            setColor(OUTLINE_COLOR)
            vertices.position(0)
            GLES20.glVertexAttribPointer(positionAttribute, 3, GLES20.GL_FLOAT, false, 0, vertices)
            GLES20.glLineWidth(OUTLINE_WIDTH_PIXELS)
            GLES20.glDrawArrays(GLES20.GL_LINE_LOOP, 0, FOOTPRINT_VERTEX_COUNT)
        }
        GLES20.glDisableVertexAttribArray(positionAttribute)
    }

    /** Draws the first measured A-B edge before the full rectangle exists. */
    fun drawEdge(
        first: FloatArray,
        second: FloatArray,
        viewMatrix: FloatArray,
        projectionMatrix: FloatArray,
    ) {
        requireWorldPoint(first)
        requireWorldPoint(second)

        vertices.clear()
        putPoint(first)
        putPoint(second)
        vertices.position(0)

        prepareProgram(viewMatrix, projectionMatrix)
        withOverlayState {
            setColor(OUTLINE_COLOR)
            vertices.position(0)
            GLES20.glVertexAttribPointer(positionAttribute, 3, GLES20.GL_FLOAT, false, 0, vertices)
            GLES20.glLineWidth(OUTLINE_WIDTH_PIXELS)
            GLES20.glDrawArrays(GLES20.GL_LINES, 0, EDGE_VERTEX_COUNT)
        }
        GLES20.glDisableVertexAttribArray(positionAttribute)
    }

    private fun prepareProgram(viewMatrix: FloatArray, projectionMatrix: FloatArray) {
        check(program != 0) { "createOnGlThread must be called before drawing the footprint" }
        Matrix.multiplyMM(viewProjectionMatrix, 0, projectionMatrix, 0, viewMatrix, 0)
        GLES20.glUseProgram(program)
        GLES20.glUniformMatrix4fv(mvpUniform, 1, false, viewProjectionMatrix, 0)
        GLES20.glEnableVertexAttribArray(positionAttribute)
    }

    private fun setColor(color: FloatArray) {
        GLES20.glUniform4fv(colorUniform, 1, color, 0)
    }

    private fun putPoint(point: FloatArray) {
        vertices.put(point[0])
        vertices.put(point[1])
        vertices.put(point[2])
    }

    private fun requireWorldPoint(point: FloatArray) {
        require(point.size >= COMPONENTS_PER_POINT) { "A world-space point must contain x, y, and z" }
    }

    /**
     * The overlay participates in depth testing but never writes depth, so the product
     * can remain visually in front of it. All GL state changed here is restored.
     */
    private inline fun withOverlayState(draw: () -> Unit) {
        val depthTestWasEnabled = GLES20.glIsEnabled(GLES20.GL_DEPTH_TEST)
        val blendWasEnabled = GLES20.glIsEnabled(GLES20.GL_BLEND)
        GLES20.glGetBooleanv(GLES20.GL_DEPTH_WRITEMASK, savedDepthWriteMask, 0)
        GLES20.glGetIntegerv(GLES20.GL_BLEND_SRC_RGB, savedBlendSourceRgb, 0)
        GLES20.glGetIntegerv(GLES20.GL_BLEND_DST_RGB, savedBlendDestinationRgb, 0)
        GLES20.glGetIntegerv(GLES20.GL_BLEND_SRC_ALPHA, savedBlendSourceAlpha, 0)
        GLES20.glGetIntegerv(GLES20.GL_BLEND_DST_ALPHA, savedBlendDestinationAlpha, 0)
        GLES20.glGetFloatv(GLES20.GL_LINE_WIDTH, savedLineWidth, 0)

        GLES20.glEnable(GLES20.GL_DEPTH_TEST)
        GLES20.glDepthMask(false)
        GLES20.glEnable(GLES20.GL_BLEND)
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA)

        try {
            draw()
        } finally {
            GLES20.glLineWidth(savedLineWidth[0])
            GLES20.glBlendFuncSeparate(
                savedBlendSourceRgb[0],
                savedBlendDestinationRgb[0],
                savedBlendSourceAlpha[0],
                savedBlendDestinationAlpha[0],
            )
            if (!blendWasEnabled) GLES20.glDisable(GLES20.GL_BLEND)
            GLES20.glDepthMask(savedDepthWriteMask[0])
            if (!depthTestWasEnabled) GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        }
    }

    companion object {
        private const val COMPONENTS_PER_POINT = 3
        private const val FOOTPRINT_VERTEX_COUNT = 4
        private const val EDGE_VERTEX_COUNT = 2
        private const val FLOATS_PER_FOOTPRINT = FOOTPRINT_VERTEX_COUNT * COMPONENTS_PER_POINT
        private const val FLOAT_BYTES = 4
        private const val OUTLINE_WIDTH_PIXELS = 4f

        private val FILL_COLOR = floatArrayOf(0.10f, 0.72f, 0.95f, 0.20f)
        private val OUTLINE_COLOR = floatArrayOf(0.20f, 0.90f, 1.00f, 0.95f)

        private const val VERTEX_SHADER = """
            uniform mat4 uMvp;
            attribute vec4 aPosition;
            void main() {
                gl_Position = uMvp * aPosition;
            }
        """

        private const val FRAGMENT_SHADER = """
            precision mediump float;
            uniform vec4 uColor;
            void main() {
                gl_FragColor = uColor;
            }
        """
    }
}
